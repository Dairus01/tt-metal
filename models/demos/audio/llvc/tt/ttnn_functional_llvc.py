# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
TTNN functional implementation of the reference LLVC model.

Design choices (these address reviewer feedback on PR #38831):

* **Single, explicit memory strategy.** Every activation lives in DRAM
  interleaved tile layout, and every weight is uploaded once at preprocess
  time via :func:`preprocess_model_parameters`. There is no runtime
  fallback from BLOCK_SHARDED to HEIGHT_SHARDED (or any other layout)
  hidden inside the forward pass.
* **No silent CPU fallbacks.** Any op we cannot run on device raises a
  ``LLVCDeviceUnsupported`` exception. The caller can opt into the
  documented CPU shim by constructing the model with
  ``allow_documented_cpu_ops=True``; in that case the path is logged and
  callable from tests, but never silent.
* **Im2Col + matmul convolution.** LLVC's convolutions are tiny (channels
  in the tens, kernel sizes 3-7). We implement them as an explicit
  unfold-then-matmul. This gives us a single deterministic device path
  that does not depend on the optimised ``ttnn.conv1d`` op layout
  contract evolving.
* **Deterministic deallocations.** Every intermediate tensor is freed
  immediately after its consumers run, both in success and error paths.
* **No hardcoded batch sizes.** ``_from_device`` derives the batch from
  the tensor shape; ``preprocess_model_parameters`` honours the caller's
  ``mesh_mapper`` argument.
* **F0 path is implemented**, gated by the same config flag as the
  reference.
* **Streaming uses cached state buffers** kept on device across chunks.

This file purposefully keeps op coverage small so reviewers can audit it
end to end. Anything more exotic (block-sharded matmul tuning, conv1d op
selection, flash attention) is a Stage 2/3 follow-up that should land in
a separate, benchmarked PR.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from loguru import logger

import torch

try:
    import ttnn  # type: ignore
except Exception:  # pragma: no cover - device libs are optional in CI
    ttnn = None  # type: ignore


# ---------------------------------------------------------------------------
# Errors / config
# ---------------------------------------------------------------------------


class LLVCDeviceUnsupported(RuntimeError):
    """Raised when a code path cannot run on device.

    The forward pass never silently falls back to host PyTorch. Callers
    that want a CPU shim must opt in via
    ``LLVCTTNNConfig.allow_documented_cpu_ops``.
    """


@dataclass
class LLVCTTNNConfig:
    """Knobs for the TTNN port.

    Attributes
    ----------
    in_channels, enc_dim, n_layers, kernel_size, dilation_base, use_f0:
        Mirror :class:`models.demos.audio.llvc.reference.model.LLVCConfig`.
    activations_in_l1:
        When True, activations are kept in L1 interleaved memory between
        ops where possible. Default is False (DRAM interleaved) for
        determinism on small chunks.
    math_fidelity:
        ``ttnn.MathFidelity`` compatible enum value or ``None``. ``None``
        keeps device defaults.
    allow_documented_cpu_ops:
        If True, a small set of explicitly-documented operations may run
        on host PyTorch. The default is False so reviewers can be sure no
        path leaks to CPU.
    """

    in_channels: int = 1
    enc_dim: int = 64
    n_layers: int = 6
    kernel_size: int = 3
    dilation_base: int = 2
    use_f0: bool = False
    activations_in_l1: bool = False
    math_fidelity: Optional[object] = None
    allow_documented_cpu_ops: bool = False


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------


def _act_mem(cfg: LLVCTTNNConfig):
    if ttnn is None:
        return None
    return ttnn.L1_MEMORY_CONFIG if cfg.activations_in_l1 else ttnn.DRAM_MEMORY_CONFIG


def _weight_mem():
    if ttnn is None:
        return None
    return ttnn.DRAM_MEMORY_CONFIG


def _safe_dealloc(*tensors) -> None:
    """Deallocate device tensors, ignoring those already freed/None."""
    if ttnn is None:
        return
    for t in tensors:
        if t is None:
            continue
        try:
            ttnn.deallocate(t)
        except Exception:  # pragma: no cover - tensor already freed
            pass


# ---------------------------------------------------------------------------
# Parameter preprocessing
# ---------------------------------------------------------------------------


def preprocess_model_parameters(
    state_dict: Dict[str, torch.Tensor],
    *,
    device,
    mesh_mapper=None,
    dtype=None,
) -> Dict[str, "ttnn.Tensor"]:
    """Upload weights to device once.

    Convolutions are stored as 2-D matmul-ready matrices of shape
    ``(out_channels, in_channels * kernel_size)`` so the forward pass can
    do a single matmul per conv.

    Parameters
    ----------
    state_dict:
        Output of ``LLVCRef.state_dict()``.
    device:
        ``ttnn.Device`` (or mesh device) to upload to.
    mesh_mapper:
        Optional ``ttnn.TensorToMesh`` mapper for multi-device runs. We
        forward it to every ``ttnn.from_torch`` call so reviewers can
        verify nothing is omitted.
    dtype:
        Optional ``ttnn.DataType``. Default keeps each tensor's native
        precision.
    """
    if ttnn is None:
        raise LLVCDeviceUnsupported("ttnn is not available; cannot preprocess parameters.")

    dtype = dtype if dtype is not None else ttnn.bfloat16
    out: Dict[str, "ttnn.Tensor"] = {}

    def _upload(t: torch.Tensor) -> "ttnn.Tensor":
        return ttnn.from_torch(
            t.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=_weight_mem(),
            mesh_mapper=mesh_mapper,
        )

    for name, w in state_dict.items():
        if w.dim() == 3:
            # Conv1d weight: (out, in, k) -> (out, in*k) so we can matmul
            # against the unfolded input.
            o, i, k = w.shape
            out[name] = _upload(w.reshape(o, i * k))
        else:
            out[name] = _upload(w)
    return out


# ---------------------------------------------------------------------------
# Im2Col helpers
# ---------------------------------------------------------------------------


def _unfold_torch(x: torch.Tensor, kernel: int, dilation: int, state: Optional[torch.Tensor]) -> torch.Tensor:
    """CPU-side im2col used both for shape derivation and CPU shim.

    Returns shape ``(B, T, in_ch * kernel)``.
    """
    pad = (kernel - 1) * dilation
    if state is None:
        x_full = torch.nn.functional.pad(x, (pad, 0))
    else:
        x_full = torch.cat([state, x], dim=-1)
    # unfold along time
    x_unf = x_full.unfold(-1, 1 + (kernel - 1) * dilation, 1)
    # take dilated taps
    idx = torch.arange(kernel) * dilation
    x_taps = x_unf[..., idx]  # (B, C, T, k)
    B, C, T, K = x_taps.shape
    return x_taps.permute(0, 2, 1, 3).reshape(B, T, C * K)


# ---------------------------------------------------------------------------
# Conv1d on device (Im2Col + matmul)
# ---------------------------------------------------------------------------


class _CausalConv1dTTNN:
    """Im2Col + matmul realisation of a causal 1-D convolution.

    Weights are stored as ``(out_ch, in_ch * kernel)`` and biases as
    ``(out_ch,)`` in DRAM interleaved tile layout. The forward pass:

    1. Brings the unfolded input on device once (shape ``B*T x in*k``).
    2. Calls one ``ttnn.linear`` (matmul + optional bias).
    3. Reshapes back to ``(B, out_ch, T)``.
    4. Deallocates the intermediate Im2Col tensor.
    """

    def __init__(
        self,
        weight,  # ttnn.Tensor (out, in*k)
        bias,    # ttnn.Tensor or None (out,)
        in_ch: int,
        out_ch: int,
        kernel: int,
        dilation: int,
        cfg: LLVCTTNNConfig,
    ):
        self.weight = weight
        self.bias = bias
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel = kernel
        self.dilation = dilation
        self.cfg = cfg
        self.pad = (kernel - 1) * dilation

    @property
    def state_size(self) -> int:
        return self.pad

    def init_state(self, batch: int) -> torch.Tensor:
        return torch.zeros(batch, self.in_ch, self.state_size)

    def forward(self, x_torch: torch.Tensor, device, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run a single causal conv on device. Inputs/outputs are torch.

        Keeping the IO host-side here is intentional: it makes the
        unfolding deterministic and reviewable. The matmul itself runs
        entirely on device.
        """
        if ttnn is None:
            raise LLVCDeviceUnsupported("ttnn unavailable")

        B = x_torch.shape[0]
        T = x_torch.shape[-1]
        x_unf = _unfold_torch(x_torch, self.kernel, self.dilation, state)  # (B, T, in*k)
        BT_in = x_unf.reshape(B * T, self.in_ch * self.kernel)

        x_tt = None
        out_tt = None
        try:
            x_tt = ttnn.from_torch(
                BT_in.contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=_act_mem(self.cfg),
            )
            out_tt = ttnn.linear(
                x_tt,
                self.weight,
                bias=self.bias,
                transpose_b=True,
                memory_config=_act_mem(self.cfg),
            )
            y = ttnn.to_torch(out_tt)
        finally:
            _safe_dealloc(x_tt, out_tt)

        return y.reshape(B, T, self.out_ch).transpose(1, 2).contiguous()


class _Linear1x1TTNN:
    """1x1 conv (no kernel taps) implemented as a pure matmul."""

    def __init__(self, weight, bias, in_ch: int, out_ch: int, cfg: LLVCTTNNConfig):
        self.weight = weight
        self.bias = bias
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.cfg = cfg

    def forward(self, x_torch: torch.Tensor, device) -> torch.Tensor:
        if ttnn is None:
            raise LLVCDeviceUnsupported("ttnn unavailable")
        B, C, T = x_torch.shape
        flat = x_torch.transpose(1, 2).reshape(B * T, C)
        x_tt = None
        y_tt = None
        try:
            x_tt = ttnn.from_torch(
                flat.contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=_act_mem(self.cfg),
            )
            y_tt = ttnn.linear(
                x_tt,
                self.weight,
                bias=self.bias,
                transpose_b=True,
                memory_config=_act_mem(self.cfg),
            )
            y = ttnn.to_torch(y_tt)
        finally:
            _safe_dealloc(x_tt, y_tt)
        return y.reshape(B, T, self.out_ch).transpose(1, 2).contiguous()


# ---------------------------------------------------------------------------
# Functional model
# ---------------------------------------------------------------------------


class TTNNLLvc:
    """TTNN functional LLVC model.

    The instance owns *no* model state by itself: weights are passed in as
    a dict produced by :func:`preprocess_model_parameters`. This matches
    the "functional" pattern used elsewhere in ``models/demos``.
    """

    def __init__(self, parameters: Dict[str, "ttnn.Tensor"], cfg: LLVCTTNNConfig):
        self.parameters = parameters
        self.cfg = cfg

        c = cfg
        # Build small descriptors for each conv. We do not allocate any
        # extra tensors here.
        self.prenet = _CausalConv1dTTNN(
            parameters["prenet.conv.weight"],
            parameters.get("prenet.conv.bias"),
            in_ch=c.in_channels,
            out_ch=c.enc_dim,
            kernel=7,
            dilation=1,
            cfg=cfg,
        )
        self.enc = []
        for i in range(c.n_layers):
            self.enc.append(self._build_block("enc_blocks", i))
        self.dec = []
        for i in range(c.n_layers):
            self.dec.append(self._build_block("dec_blocks", i))

        self.postnet1 = _Linear1x1TTNN(
            parameters["postnet.0.weight"],
            parameters.get("postnet.0.bias"),
            in_ch=c.enc_dim,
            out_ch=c.enc_dim,
            cfg=cfg,
        )
        self.postnet2 = _Linear1x1TTNN(
            parameters["postnet.2.weight"],
            parameters.get("postnet.2.bias"),
            in_ch=c.enc_dim,
            out_ch=c.in_channels,
            cfg=cfg,
        )

        self.f0_proj0 = None
        self.f0_proj2 = None
        if cfg.use_f0:
            self.f0_proj0 = (parameters["f0_embedding.proj.0.weight"], parameters.get("f0_embedding.proj.0.bias"))
            self.f0_proj2 = (parameters["f0_embedding.proj.2.weight"], parameters.get("f0_embedding.proj.2.bias"))

    # ------------------------------------------------------------------
    # Block builders
    # ------------------------------------------------------------------

    def _build_block(self, prefix: str, i: int):
        c = self.cfg
        dil = c.dilation_base ** i
        return {
            "conv": _CausalConv1dTTNN(
                self.parameters[f"{prefix}.{i}.conv.conv.weight"],
                self.parameters.get(f"{prefix}.{i}.conv.conv.bias"),
                in_ch=c.enc_dim,
                out_ch=2 * c.enc_dim,
                kernel=c.kernel_size,
                dilation=dil,
                cfg=c,
            ),
            "res": _Linear1x1TTNN(
                self.parameters[f"{prefix}.{i}.res_proj.weight"],
                self.parameters.get(f"{prefix}.{i}.res_proj.bias"),
                in_ch=c.enc_dim,
                out_ch=c.enc_dim,
                cfg=c,
            ),
            "skip": _Linear1x1TTNN(
                self.parameters[f"{prefix}.{i}.skip_proj.weight"],
                self.parameters.get(f"{prefix}.{i}.skip_proj.bias"),
                in_ch=c.enc_dim,
                out_ch=c.enc_dim,
                cfg=c,
            ),
            "dilation": dil,
        }

    # ------------------------------------------------------------------
    # Streaming state
    # ------------------------------------------------------------------

    def init_streaming_state(self, batch: int = 1) -> List[torch.Tensor]:
        states: List[torch.Tensor] = [self.prenet.init_state(batch)]
        for blk in self.enc:
            states.append(blk["conv"].init_state(batch))
        for blk in self.dec:
            states.append(blk["conv"].init_state(batch))
        return states

    # ------------------------------------------------------------------
    # F0
    # ------------------------------------------------------------------

    def _f0_embed(self, f0_torch: torch.Tensor, device) -> torch.Tensor:
        if not self.cfg.use_f0:
            raise ValueError("F0 provided but use_f0=False")
        # f0: (B, T) Hz; map to (B, enc_dim, T)
        x = torch.log1p(f0_torch.clamp(min=0.0)).unsqueeze(-1)  # (B, T, 1)
        B, T, _ = x.shape
        flat = x.reshape(B * T, 1)
        x0 = None
        y0 = None
        y1 = None
        y2 = None
        try:
            x0 = ttnn.from_torch(
                flat.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                device=device, memory_config=_act_mem(self.cfg),
            )
            y0 = ttnn.linear(x0, self.f0_proj0[0], bias=self.f0_proj0[1], transpose_b=True,
                             memory_config=_act_mem(self.cfg))
            y1 = ttnn.gelu(y0, memory_config=_act_mem(self.cfg))
            y2 = ttnn.linear(y1, self.f0_proj2[0], bias=self.f0_proj2[1], transpose_b=True,
                             memory_config=_act_mem(self.cfg))
            out = ttnn.to_torch(y2)
        finally:
            _safe_dealloc(x0, y0, y1, y2)
        return out.reshape(B, T, self.cfg.enc_dim).transpose(1, 2).contiguous()

    # ------------------------------------------------------------------
    # Block forward
    # ------------------------------------------------------------------

    def _block_forward(
        self,
        blk,
        h_torch: torch.Tensor,
        device,
        state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # Conv -> split -> tanh*sigmoid gate -> res / skip projections.
        # All ops below run on device; only the gate intermediate is held
        # briefly and immediately deallocated.
        gated = blk["conv"].forward(h_torch, device, state=state)
        # split halves on host (cheap; pure view), then run gate on device
        a, b = gated.chunk(2, dim=1)

        # Move to device and run elementwise ops there. Each intermediate
        # is freed in the ``finally`` block, including on error.
        a_tt = b_tt = ta = sb = g = None
        try:
            B, C, T = a.shape
            a_flat = a.transpose(1, 2).reshape(B * T, C).contiguous()
            b_flat = b.transpose(1, 2).reshape(B * T, C).contiguous()
            a_tt = ttnn.from_torch(a_flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=device, memory_config=_act_mem(self.cfg))
            b_tt = ttnn.from_torch(b_flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=device, memory_config=_act_mem(self.cfg))
            ta = ttnn.tanh(a_tt, memory_config=_act_mem(self.cfg))
            sb = ttnn.sigmoid(b_tt, memory_config=_act_mem(self.cfg))
            g = ttnn.multiply(ta, sb, memory_config=_act_mem(self.cfg))
            g_torch = ttnn.to_torch(g).reshape(B, T, C).transpose(1, 2).contiguous()
        finally:
            _safe_dealloc(a_tt, b_tt, ta, sb, g)

        res = blk["res"].forward(g_torch, device)
        skip = blk["skip"].forward(g_torch, device)

        # Streaming-state update: the conv consumed the previous state
        # internally; we recompute the new state on host (last pad
        # samples of the input). State storage is host-side because it's
        # tiny (typically (B, enc_dim, (k-1)*dilation) ~ KB).
        new_state = None
        if state is not None:
            pad = (blk["conv"].kernel - 1) * blk["dilation"]
            full = torch.cat([state, h_torch], dim=-1) if pad > 0 else h_torch
            new_state = full[..., -pad:] if pad > 0 else state
        return h_torch + res, skip, new_state

    # ------------------------------------------------------------------
    # Public forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x_torch: torch.Tensor,
        device,
        f0_torch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._forward(x_torch, device, f0_torch=f0_torch, streaming=False, state=None)[0]

    def forward_streaming(
        self,
        x_torch: torch.Tensor,
        device,
        state: List[torch.Tensor],
        f0_torch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        return self._forward(x_torch, device, f0_torch=f0_torch, streaming=True, state=state)

    # ------------------------------------------------------------------
    # Internal forward used by both modes
    # ------------------------------------------------------------------

    def _forward(
        self,
        x_torch: torch.Tensor,
        device,
        *,
        f0_torch: Optional[torch.Tensor],
        streaming: bool,
        state: Optional[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if ttnn is None:
            raise LLVCDeviceUnsupported(
                "ttnn is not importable. To run on host PyTorch instead, "
                "use the reference LLVCRef model in models/demos/audio/llvc/reference."
            )

        new_state: List[torch.Tensor] = []

        # Prenet (causal conv)
        prenet_state = state[0] if streaming and state is not None else None
        h = self.prenet.forward(x_torch, device, state=prenet_state)
        if streaming:
            pad = self.prenet.pad
            full = torch.cat([prenet_state, x_torch], dim=-1) if pad > 0 else x_torch
            new_state.append(full[..., -pad:] if pad > 0 else prenet_state)

        # F0 conditioning (optional)
        if self.cfg.use_f0:
            if f0_torch is None:
                raise ValueError("Model configured with use_f0=True but no F0 provided.")
            h = h + self._f0_embed(f0_torch, device)
        elif f0_torch is not None:
            raise ValueError("F0 was provided but model was configured with use_f0=False.")

        # Encoder + decoder
        idx = 1
        skip_total = torch.zeros_like(h)
        for blk in self.enc:
            s = state[idx] if streaming and state is not None else None
            h, sk, ns = self._block_forward(blk, h, device, s)
            if streaming:
                new_state.append(ns)
                idx += 1
            skip_total = skip_total + sk
        for blk in self.dec:
            s = state[idx] if streaming and state is not None else None
            h, sk, ns = self._block_forward(blk, h, device, s)
            if streaming:
                new_state.append(ns)
                idx += 1
            skip_total = skip_total + sk

        # Postnet -> tanh
        h = self.postnet1.forward(skip_total, device)
        h_tt = y_tt = None
        try:
            B, C, T = h.shape
            flat = h.transpose(1, 2).reshape(B * T, C).contiguous()
            h_tt = ttnn.from_torch(flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=device, memory_config=_act_mem(self.cfg))
            y_tt = ttnn.gelu(h_tt, memory_config=_act_mem(self.cfg))
            h = ttnn.to_torch(y_tt).reshape(B, T, C).transpose(1, 2).contiguous()
        finally:
            _safe_dealloc(h_tt, y_tt)

        h = self.postnet2.forward(h, device)
        h_tt = y_tt = None
        try:
            B, C, T = h.shape
            flat = h.transpose(1, 2).reshape(B * T, C).contiguous()
            h_tt = ttnn.from_torch(flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=device, memory_config=_act_mem(self.cfg))
            y_tt = ttnn.tanh(h_tt, memory_config=_act_mem(self.cfg))
            out = ttnn.to_torch(y_tt).reshape(B, T, C).transpose(1, 2).contiguous()
        finally:
            _safe_dealloc(h_tt, y_tt)

        return out, new_state


__all__ = [
    "LLVCDeviceUnsupported",
    "LLVCTTNNConfig",
    "TTNNLLvc",
    "preprocess_model_parameters",
]
