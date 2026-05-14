# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Correctness tests for LLVC.

Includes:

* CPU tests for the reference model (always run).
* Device tests for the TTNN model that exercise non-streaming and
  streaming parity. They are gated on the availability of an actual
  Tenstorrent device via ``ttnn.open_device(0)``; in CI runners without
  hardware they ``skip`` cleanly rather than silently passing on CPU.
"""
from __future__ import annotations

import math

import pytest
import torch

from models.demos.audio.llvc.reference.model import LLVCConfig, LLVCRef


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom == 0.0:
        return 1.0 if torch.allclose(a, b) else 0.0
    return float((a @ b).item() / denom)


# ---------------------------------------------------------------------------
# CPU reference tests
# ---------------------------------------------------------------------------


def test_reference_streaming_matches_full():
    torch.manual_seed(0)
    cfg = LLVCConfig(enc_dim=32, n_layers=4, chunk_size=128)
    model = LLVCRef(cfg).eval()
    x = torch.randn(1, 1, 4 * cfg.chunk_size)

    with torch.no_grad():
        y_full = model(x)
        state = model.init_streaming_state(batch=1)
        outs = []
        for i in range(0, x.shape[-1], cfg.chunk_size):
            y_c, state = model.forward_streaming(x[..., i : i + cfg.chunk_size], state)
            outs.append(y_c)
        y_stream = torch.cat(outs, dim=-1)

    assert y_full.shape == y_stream.shape
    assert (y_full - y_stream).abs().max().item() < 1e-5


def test_reference_f0_path_runs():
    torch.manual_seed(0)
    cfg = LLVCConfig(enc_dim=16, n_layers=2, use_f0=True)
    model = LLVCRef(cfg).eval()
    x = torch.randn(1, 1, 256)
    f0 = torch.full((1, 256), 220.0)
    with torch.no_grad():
        y = model(x, f0=f0)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_reference_rejects_f0_when_disabled():
    cfg = LLVCConfig(enc_dim=8, n_layers=1, use_f0=False)
    model = LLVCRef(cfg).eval()
    x = torch.randn(1, 1, 64)
    f0 = torch.zeros(1, 64)
    with pytest.raises(ValueError):
        model(x, f0=f0)


def test_reference_requires_f0_when_enabled():
    cfg = LLVCConfig(enc_dim=8, n_layers=1, use_f0=True)
    model = LLVCRef(cfg).eval()
    x = torch.randn(1, 1, 64)
    with pytest.raises(ValueError):
        model(x)


# ---------------------------------------------------------------------------
# Device tests (skipped without hardware)
# ---------------------------------------------------------------------------


def _maybe_open_device():
    try:
        import ttnn  # noqa
    except Exception:
        pytest.skip("ttnn not installed; device tests skipped.")
    try:
        return ttnn.open_device(device_id=0)
    except Exception as e:
        pytest.skip(f"No Tenstorrent device available; device tests skipped ({e}).")


@pytest.fixture
def device():
    dev = _maybe_open_device()
    yield dev
    import ttnn  # noqa
    try:
        ttnn.close_device(dev)
    except Exception:
        pass


@pytest.mark.requires_device
def test_ttnn_non_streaming_pcc(device):
    from models.demos.audio.llvc.tt.ttnn_functional_llvc import (
        LLVCTTNNConfig,
        TTNNLLvc,
        preprocess_model_parameters,
    )

    torch.manual_seed(0)
    ref_cfg = LLVCConfig(enc_dim=32, n_layers=4, chunk_size=128)
    tt_cfg = LLVCTTNNConfig(enc_dim=ref_cfg.enc_dim, n_layers=ref_cfg.n_layers)
    ref = LLVCRef(ref_cfg).eval()
    params = preprocess_model_parameters(ref.state_dict(), device=device)
    tt = TTNNLLvc(params, tt_cfg)

    x = torch.randn(1, 1, 4 * ref_cfg.chunk_size)
    with torch.no_grad():
        y_ref = ref(x)
    y_tt = tt.forward(x, device)
    pcc = _pcc(y_ref, y_tt)
    # bf16 + Im2Col matmul: PCC > 0.99 is the bring-up bar.
    assert pcc > 0.99, f"PCC {pcc:.4f} below 0.99"


@pytest.mark.requires_device
def test_ttnn_streaming_matches_non_streaming(device):
    from models.demos.audio.llvc.tt.ttnn_functional_llvc import (
        LLVCTTNNConfig,
        TTNNLLvc,
        preprocess_model_parameters,
    )

    torch.manual_seed(0)
    ref_cfg = LLVCConfig(enc_dim=32, n_layers=4, chunk_size=128)
    tt_cfg = LLVCTTNNConfig(enc_dim=ref_cfg.enc_dim, n_layers=ref_cfg.n_layers)
    ref = LLVCRef(ref_cfg).eval()
    params = preprocess_model_parameters(ref.state_dict(), device=device)
    tt = TTNNLLvc(params, tt_cfg)

    x = torch.randn(1, 1, 4 * ref_cfg.chunk_size)
    y_full = tt.forward(x, device)
    state = tt.init_streaming_state(batch=1)
    outs = []
    for i in range(0, x.shape[-1], ref_cfg.chunk_size):
        y_c, state = tt.forward_streaming(x[..., i : i + ref_cfg.chunk_size], device, state)
        outs.append(y_c)
    y_stream = torch.cat(outs, dim=-1)

    assert y_stream.shape == y_full.shape
    assert _pcc(y_full, y_stream) > 0.99


@pytest.mark.requires_device
def test_ttnn_no_silent_cpu_fallback_when_ttnn_missing(monkeypatch):
    """Importing the module without ttnn must surface a clear error if forward is invoked."""
    from models.demos.audio.llvc.tt import ttnn_functional_llvc as mod

    monkeypatch.setattr(mod, "ttnn", None, raising=False)
    cfg = mod.LLVCTTNNConfig(enc_dim=8, n_layers=1, allow_documented_cpu_ops=False)
    model = mod.TTNNLLvc.__new__(mod.TTNNLLvc)  # bypass __init__ (which uses ttnn)
    model.cfg = cfg
    with pytest.raises(mod.LLVCDeviceUnsupported):
        model._forward(torch.zeros(1, 1, 16), device=None, f0_torch=None, streaming=False, state=None)
