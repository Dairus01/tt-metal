# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Self-contained PyTorch reference for LLVC (Low-Latency Low-Resource Voice
Conversion) suitable as a parity target for the TTNN port.

This implementation is intentionally simplified compared to the full
KoeAI/LLVC training pipeline -- it captures the structural pieces that
matter for inference parity:

* Causal dilated 1-D convolution blocks with explicit input-side state
  buffers, enabling true streaming (chunk by chunk) inference.
* A WaveNet-style residual encoder / decoder stack.
* An *optional* F0 conditioning path. It is OFF by default (matching the
  primary LLVC "F0-free" mode the bounty calls out). When enabled, F0 is
  embedded and added to the encoder bottleneck.
* A tiny prenet that mirrors the reference repo's input projection. We
  document any deliberate deviations from the reference.

The model is intentionally small (default ~1M params) so it can be tested
end-to-end without external checkpoints. Loading the official KoeAI
weights is out of scope of this file; the structure is what matters for
the TTNN parity tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class LLVCConfig:
    """Configuration for the reference LLVC model.

    Attributes
    ----------
    in_channels:
        Number of input audio channels. Always ``1`` for mono speech.
    enc_dim:
        Hidden width of the encoder/decoder residual stacks.
    n_layers:
        Number of dilated residual blocks per stack.
    kernel_size:
        Kernel size for the dilated convolutions.
    dilation_base:
        Base used for exponential dilation: layer ``i`` uses
        ``dilation_base ** i``.
    chunk_size:
        Default streaming chunk size in samples (frame size). Used by the
        streaming helpers.
    use_f0:
        When ``True``, the model exposes an F0 conditioning input (one
        scalar per encoder frame). This is the optional pitch-conditioned
        path described in the bounty.
    sample_rate:
        Sample rate of the audio (informational; used by RTF helpers).
    """

    in_channels: int = 1
    enc_dim: int = 64
    n_layers: int = 6
    kernel_size: int = 3
    dilation_base: int = 2
    chunk_size: int = 512
    use_f0: bool = False
    sample_rate: int = 16000


# ---------------------------------------------------------------------------
# Streaming-friendly causal Conv1d
# ---------------------------------------------------------------------------


class CausalConv1d(nn.Module):
    """A 1-D convolution with explicit causal-padding state for streaming.

    The non-streaming forward pass left-pads with zeros (standard causal
    conv). The streaming forward pass instead consumes a state buffer of
    shape ``(B, C, (k-1)*d)`` representing the last ``(k-1)*d`` input
    samples, which the caller is expected to maintain across chunks.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1, bias: bool = True):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, bias=bias)

    @property
    def state_size(self) -> int:
        return self.pad

    def init_state(self, batch: int, device=None, dtype=None) -> torch.Tensor:
        return torch.zeros(batch, self.in_ch, self.state_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Non-streaming: left-pad with zeros so output length == input length.
        if self.pad > 0:
            x = F.pad(x, (self.pad, 0))
        return self.conv(x)

    def forward_streaming(self, x: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Streaming forward.

        Parameters
        ----------
        x:
            Tensor of shape ``(B, C, T)`` containing a new chunk of audio.
        state:
            Tensor of shape ``(B, C, (k-1)*d)`` carrying the previous
            samples needed to make the convolution causal.

        Returns
        -------
        (y, new_state):
            ``y`` has the same time length as ``x``. ``new_state`` is the
            updated buffer, ready to be passed to the next call.
        """
        if self.pad == 0:
            return self.conv(x), state
        x_full = torch.cat([state, x], dim=-1)
        y = self.conv(x_full)
        new_state = x_full[..., -self.pad :]
        return y, new_state


# ---------------------------------------------------------------------------
# Residual block (WaveNet-style, gated)
# ---------------------------------------------------------------------------


class DilatedResidualBlock(nn.Module):
    """A WaveNet-style gated residual block.

    Mirrors KoeAI/LLVC's residual unit: a single dilated causal Conv1d
    feeds two halves which are gated with ``tanh * sigmoid``, then a 1x1
    conv produces the residual and skip outputs.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv = CausalConv1d(channels, 2 * channels, kernel_size, dilation=dilation)
        self.res_proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.skip_proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x)
        a, b = h.chunk(2, dim=1)
        h = torch.tanh(a) * torch.sigmoid(b)
        return x + self.res_proj(h), self.skip_proj(h)

    def forward_streaming(
        self, x: torch.Tensor, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, new_state = self.conv.forward_streaming(x, state)
        a, b = h.chunk(2, dim=1)
        h = torch.tanh(a) * torch.sigmoid(b)
        return x + self.res_proj(h), self.skip_proj(h), new_state


# ---------------------------------------------------------------------------
# F0 embedding
# ---------------------------------------------------------------------------


class F0Embedding(nn.Module):
    """Optional pitch (F0) conditioning.

    F0 is provided as a scalar per encoder frame (Hz). We log-scale it,
    project it through a small MLP, and broadcast-add it to the encoder
    bottleneck. Disabled by default to match the F0-free LLVC mode.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        # f0: (B, T) in Hz, with 0 for unvoiced. Use log1p for stability.
        x = torch.log1p(f0.clamp(min=0.0)).unsqueeze(-1)
        emb = self.proj(x)  # (B, T, C)
        return emb.transpose(1, 2)  # (B, C, T)


# ---------------------------------------------------------------------------
# LLVC reference model
# ---------------------------------------------------------------------------


class LLVCRef(nn.Module):
    """Reference LLVC model.

    The forward pass takes a raw waveform ``(B, 1, T)`` and returns a
    converted waveform of the same shape. ``forward_streaming`` consumes
    a chunk plus an opaque ``state`` (a list of per-block buffers) and
    returns the next chunk plus the updated state.
    """

    def __init__(self, config: Optional[LLVCConfig] = None):
        super().__init__()
        self.config = config or LLVCConfig()
        c = self.config

        # Prenet: matches the reference repo's 7-tap causal input conv. We
        # keep this explicit so any deviation from KoeAI is easy to spot.
        self.prenet = CausalConv1d(c.in_channels, c.enc_dim, kernel_size=7, dilation=1)

        # Encoder: stack of dilated residual blocks.
        self.enc_blocks = nn.ModuleList(
            [
                DilatedResidualBlock(c.enc_dim, c.kernel_size, dilation=c.dilation_base ** i)
                for i in range(c.n_layers)
            ]
        )

        # Optional F0 conditioning
        self.f0_embedding: Optional[F0Embedding] = F0Embedding(c.enc_dim) if c.use_f0 else None

        # Decoder: another stack of dilated residual blocks with reset
        # dilation pattern (mirrors the WaveNet decoder in LLVC).
        self.dec_blocks = nn.ModuleList(
            [
                DilatedResidualBlock(c.enc_dim, c.kernel_size, dilation=c.dilation_base ** i)
                for i in range(c.n_layers)
            ]
        )

        # Output projection back to waveform
        self.postnet = nn.Sequential(
            nn.Conv1d(c.enc_dim, c.enc_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(c.enc_dim, c.in_channels, kernel_size=1),
        )

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, f0: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.prenet(x)

        if self.f0_embedding is not None:
            if f0 is None:
                raise ValueError("Model was configured with use_f0=True but no F0 was provided.")
            h = h + self.f0_embedding(f0)
        elif f0 is not None:
            raise ValueError("F0 was provided but the model was configured with use_f0=False.")

        skip_total = torch.zeros_like(h)
        for block in self.enc_blocks:
            h, s = block(h)
            skip_total = skip_total + s
        for block in self.dec_blocks:
            h, s = block(h)
            skip_total = skip_total + s

        return torch.tanh(self.postnet(skip_total))

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def init_streaming_state(self, batch: int = 1, device=None, dtype=None) -> List[torch.Tensor]:
        states: List[torch.Tensor] = [self.prenet.init_state(batch, device, dtype)]
        for blk in self.enc_blocks:
            states.append(blk.conv.init_state(batch, device, dtype))
        for blk in self.dec_blocks:
            states.append(blk.conv.init_state(batch, device, dtype))
        return states

    def forward_streaming(
        self,
        x: torch.Tensor,
        state: List[torch.Tensor],
        f0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        new_state: List[torch.Tensor] = []
        h, s0 = self.prenet.forward_streaming(x, state[0])
        new_state.append(s0)

        if self.f0_embedding is not None:
            if f0 is None:
                raise ValueError("Streaming with use_f0=True requires per-chunk F0.")
            h = h + self.f0_embedding(f0)

        idx = 1
        skip_total = torch.zeros_like(h)
        for blk in self.enc_blocks:
            h, sk, ns = blk.forward_streaming(h, state[idx])
            new_state.append(ns)
            skip_total = skip_total + sk
            idx += 1
        for blk in self.dec_blocks:
            h, sk, ns = blk.forward_streaming(h, state[idx])
            new_state.append(ns)
            skip_total = skip_total + sk
            idx += 1

        return torch.tanh(self.postnet(skip_total)), new_state


__all__ = ["LLVCConfig", "LLVCRef", "CausalConv1d", "DilatedResidualBlock", "F0Embedding"]
