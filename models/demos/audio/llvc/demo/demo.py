# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end LLVC demo.

Modes
-----
``--mode cpu``      Run the PyTorch reference end-to-end.
``--mode device``   Run the TTNN port (requires a Tenstorrent device).

Streaming
---------
Add ``--streaming`` to run chunk-by-chunk. The demo prints PCC of
streaming vs non-streaming to verify state caching is correct, plus the
streaming RTF and per-chunk latency.

Inputs
------
``--input PATH`` accepts a 16 kHz mono ``.wav``. If omitted, a 4-second
synthetic test signal is used so the demo runs in any environment.
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import torch

from models.demos.audio.llvc.reference.model import LLVCConfig, LLVCRef


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    return float((a @ b).item() / denom) if denom > 0 else 1.0


def _load_audio(path: str | None, sample_rate: int) -> torch.Tensor:
    if path is None:
        # Sum-of-sinusoids placeholder
        t = torch.linspace(0, 4.0, 4 * sample_rate)
        x = 0.5 * torch.sin(2 * math.pi * 220.0 * t) + 0.3 * torch.sin(2 * math.pi * 440.0 * t)
        return x.unsqueeze(0).unsqueeze(0)
    try:
        import scipy.io.wavfile as wavfile
        sr, y = wavfile.read(path)
    except Exception as e:
        raise SystemExit(f"Failed to read {path}: {e}")
    if sr != sample_rate:
        raise SystemExit(f"Expected {sample_rate} Hz audio, got {sr}.")
    if y.ndim > 1:
        y = y.mean(axis=-1)
    y = torch.from_numpy(y).to(torch.float32)
    if y.abs().max() > 1.0:
        y = y / 32768.0
    return y.unsqueeze(0).unsqueeze(0)


def _save_wav(path: str, x: torch.Tensor, sample_rate: int):
    try:
        import numpy as np
        import scipy.io.wavfile as wavfile
        y = x.squeeze().detach().cpu().numpy()
        y = np.clip(y, -1.0, 1.0)
        wavfile.write(path, sample_rate, (y * 32767).astype("int16"))
    except Exception as e:
        print(f"[warn] could not save wav to {path}: {e}", file=sys.stderr)


def _run_cpu(x: torch.Tensor, cfg: LLVCConfig, streaming: bool):
    model = LLVCRef(cfg).eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        y_full = model(x)
        full_time = time.perf_counter() - t0
        if streaming:
            state = model.init_streaming_state(batch=1)
            outs = []
            chunk_times = []
            t0 = time.perf_counter()
            for i in range(0, x.shape[-1], cfg.chunk_size):
                cs = time.perf_counter()
                yc, state = model.forward_streaming(x[..., i : i + cfg.chunk_size], state)
                chunk_times.append(time.perf_counter() - cs)
                outs.append(yc)
            stream_time = time.perf_counter() - t0
            y_stream = torch.cat(outs, dim=-1)
            return y_full, y_stream, full_time, stream_time, sum(chunk_times) / len(chunk_times)
    return y_full, None, full_time, None, None


def _run_device(x: torch.Tensor, cfg: LLVCConfig, streaming: bool):
    import ttnn  # noqa
    from models.demos.audio.llvc.tt.ttnn_functional_llvc import (
        LLVCTTNNConfig,
        TTNNLLvc,
        preprocess_model_parameters,
    )

    device = ttnn.open_device(device_id=0)
    try:
        ref = LLVCRef(cfg).eval()
        params = preprocess_model_parameters(ref.state_dict(), device=device)
        tt_cfg = LLVCTTNNConfig(enc_dim=cfg.enc_dim, n_layers=cfg.n_layers)
        model = TTNNLLvc(params, tt_cfg)

        # Warmup
        _ = model.forward(x, device)
        t0 = time.perf_counter()
        y_full = model.forward(x, device)
        full_time = time.perf_counter() - t0

        y_stream = None
        stream_time = None
        chunk_lat = None
        if streaming:
            state = model.init_streaming_state(batch=1)
            outs = []
            chunk_times = []
            t0 = time.perf_counter()
            for i in range(0, x.shape[-1], cfg.chunk_size):
                cs = time.perf_counter()
                yc, state = model.forward_streaming(x[..., i : i + cfg.chunk_size], device, state)
                chunk_times.append(time.perf_counter() - cs)
                outs.append(yc)
            stream_time = time.perf_counter() - t0
            y_stream = torch.cat(outs, dim=-1)
            chunk_lat = sum(chunk_times) / len(chunk_times)
        return y_full, y_stream, full_time, stream_time, chunk_lat
    finally:
        try:
            ttnn.close_device(device)
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["cpu", "device"], default="cpu")
    p.add_argument("--streaming", action="store_true")
    p.add_argument("--input", default=None, help="Path to a 16 kHz mono .wav")
    p.add_argument("--output", default=None, help="Where to save the converted audio")
    p.add_argument("--enc-dim", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--chunk-size", type=int, default=512)
    args = p.parse_args()

    cfg = LLVCConfig(enc_dim=args.enc_dim, n_layers=args.n_layers, chunk_size=args.chunk_size)
    x = _load_audio(args.input, cfg.sample_rate)

    if args.mode == "cpu":
        y_full, y_stream, ft, st, cl = _run_cpu(x, cfg, args.streaming)
    else:
        y_full, y_stream, ft, st, cl = _run_device(x, cfg, args.streaming)

    audio_s = x.shape[-1] / cfg.sample_rate
    print(f"[mode={args.mode}] non-streaming: {ft:.3f}s  RTF={ft/audio_s:.3f}")
    if args.streaming and y_stream is not None:
        rtf = st / audio_s
        print(f"[mode={args.mode}] streaming    : {st:.3f}s  RTF={rtf:.3f}  chunk_latency={cl*1000:.2f} ms")
        pcc = _pcc(y_full, y_stream)
        print(f"[mode={args.mode}] streaming vs non-streaming PCC = {pcc:.4f}")
        marker_s1 = "OK" if rtf < 0.3 else "MISS"
        marker_s3 = "OK" if rtf < 0.1 else "MISS"
        print(f"[stage gates] streaming RTF<0.3 (Stage 1): {marker_s1}; <0.1 (Stage 3): {marker_s3}")

    if args.output is not None:
        _save_wav(args.output, y_stream if args.streaming else y_full, cfg.sample_rate)


if __name__ == "__main__":
    main()
