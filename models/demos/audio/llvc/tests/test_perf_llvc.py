# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Performance test for LLVC.

This test produces a perf CSV via :func:`models.perf.perf_utils.prep_perf_report`
**plus** a sidecar CSV row that surfaces the bounty's streaming-specific
metrics that ``prep_perf_report`` does not natively model:

* ``streaming_rtf``  -- real-time factor when running chunk-by-chunk.
* ``streaming_chunk_latency_ms`` -- mean wall-clock latency per chunk.
* ``streaming_meets_stage1`` -- bool, ``streaming_rtf < 0.3``.
* ``streaming_meets_stage3`` -- bool, ``streaming_rtf < 0.1``.
* ``decoder_tokens_per_sec`` -- audio samples produced per second.

The test does **not** assert pass/fail on streaming RTF on host: there is
no Tenstorrent device available in CPU-only CI, and the reference model
running on CPU is not a meaningful proxy. The values are reported so the
perf report and CSV always show the truthful number, never a misleading
✅. A device-only test method asserts the Stage 1 bound when a device is
present.
"""
from __future__ import annotations

import csv
import time

import pytest
import torch
from loguru import logger

from models.demos.audio.llvc.reference.model import LLVCConfig, LLVCRef
from models.perf.perf_utils import prep_perf_report


STAGE1_RTF_THRESHOLD = 0.3
STAGE3_RTF_THRESHOLD = 0.1


def _measure_streaming_rtf(model, x: torch.Tensor, chunk: int, sample_rate: int, warmup_passes: int = 2):
    """Run streaming inference and return (rtf, chunk_latency_s, total_s).

    A streaming RTF measurement that doesn't warm up first is dominated by
    one-time costs (PyTorch caches, MKL plan selection) on the first chunk,
    which is not representative of any real deployment. We run a small
    number of warmup passes (still through the exact same code path) before
    timing.
    """
    with torch.no_grad():
        for _ in range(warmup_passes):
            state = model.init_streaming_state(batch=x.shape[0])
            for i in range(0, x.shape[-1], chunk):
                _, state = model.forward_streaming(x[..., i : i + chunk], state)
        state = model.init_streaming_state(batch=x.shape[0])
        chunk_times = []
        t0 = time.perf_counter()
        for i in range(0, x.shape[-1], chunk):
            cs = time.perf_counter()
            _, state = model.forward_streaming(x[..., i : i + chunk], state)
            chunk_times.append(time.perf_counter() - cs)
        total = time.perf_counter() - t0
    audio_seconds = x.shape[-1] / sample_rate
    rtf = total / audio_seconds
    return rtf, sum(chunk_times) / len(chunk_times), total


def _write_streaming_sidecar(comments: str, row: dict) -> str:
    today = time.strftime("%Y_%m_%d")
    path = f"perf_llvc_streaming_{comments}_{today}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    return path


@pytest.mark.parametrize("chunk_size", [512])
def test_llvc_performance_cpu(chunk_size):
    """CPU baseline + streaming RTF reported via prep_perf_report + sidecar.

    The CPU streaming baseline is pinned to a single thread. For the small
    matmuls used in LLVC, intra-op parallelism costs more in synchronisation
    than it saves, and the goal of this test is a deterministic, comparable
    CPU number -- not a tuned production CPU benchmark.
    """
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cfg = LLVCConfig(enc_dim=64, n_layers=6, chunk_size=chunk_size)
    model = LLVCRef(cfg).eval()
    sample_rate = cfg.sample_rate
    seconds = 4.0
    x = torch.randn(1, 1, int(sample_rate * seconds))

    # Non-streaming timing
    with torch.no_grad():
        t0 = time.perf_counter()
        _ = model(x)
        first_run = time.perf_counter() - t0
        t0 = time.perf_counter()
        _ = model(x)
        second_run = time.perf_counter() - t0

    rtf_stream, chunk_lat, total_stream = _measure_streaming_rtf(model, x, chunk_size, sample_rate)

    comments = "cpu_reference"
    prep_perf_report(
        model_name="llvc",
        batch_size=1,
        inference_and_compile_time=first_run,
        inference_time=second_run,
        expected_compile_time=10.0,
        expected_inference_time=second_run,  # baseline; not a regression gate
        comments=comments,
        inference_time_cpu=second_run,
    )

    sidecar = _write_streaming_sidecar(
        comments,
        {
            "model": "llvc",
            "device": "cpu",
            "audio_seconds": f"{seconds:.3f}",
            "chunk_size": str(chunk_size),
            "chunk_latency_ms": f"{chunk_lat * 1000:.3f}",
            "streaming_rtf": f"{rtf_stream:.4f}",
            "streaming_meets_stage1": str(rtf_stream < STAGE1_RTF_THRESHOLD).lower(),
            "streaming_meets_stage3": str(rtf_stream < STAGE3_RTF_THRESHOLD).lower(),
            "decoder_tokens_per_sec": f"{x.shape[-1] / max(1e-9, total_stream):.2f}",
        },
    )
    logger.info(f"Wrote streaming perf sidecar: {sidecar}; RTF={rtf_stream:.3f}")
    torch.set_num_threads(prev_threads)

    # We do *not* assert on RTF here: CPU performance is not the bounty's
    # target. The CSV faithfully reports the number so reviewers can see
    # whether Stage 1 is met on the *device* run below.


@pytest.mark.requires_device
def test_llvc_performance_device():
    """Device perf + asserts Stage 1 streaming RTF if the device is present."""
    try:
        import ttnn  # noqa
    except Exception:
        pytest.skip("ttnn not installed; device perf skipped.")
    try:
        device = ttnn.open_device(device_id=0)
    except Exception as e:
        pytest.skip(f"No Tenstorrent device available; device perf skipped ({e}).")

    try:
        from models.demos.audio.llvc.tt.ttnn_functional_llvc import (
            LLVCTTNNConfig,
            TTNNLLvc,
            preprocess_model_parameters,
        )

        torch.manual_seed(0)
        chunk_size = 512
        ref_cfg = LLVCConfig(enc_dim=64, n_layers=6, chunk_size=chunk_size)
        tt_cfg = LLVCTTNNConfig(enc_dim=ref_cfg.enc_dim, n_layers=ref_cfg.n_layers)
        ref = LLVCRef(ref_cfg).eval()
        params = preprocess_model_parameters(ref.state_dict(), device=device)
        tt = TTNNLLvc(params, tt_cfg)

        sample_rate = ref_cfg.sample_rate
        seconds = 4.0
        x = torch.randn(1, 1, int(sample_rate * seconds))

        # Warmup
        _ = tt.forward(x, device)
        t0 = time.perf_counter()
        _ = tt.forward(x, device)
        non_streaming_time = time.perf_counter() - t0

        # Streaming
        state = tt.init_streaming_state(batch=1)
        chunk_times = []
        t0 = time.perf_counter()
        for i in range(0, x.shape[-1], chunk_size):
            cs = time.perf_counter()
            _, state = tt.forward_streaming(x[..., i : i + chunk_size], device, state)
            chunk_times.append(time.perf_counter() - cs)
        streaming_time = time.perf_counter() - t0
        rtf = streaming_time / seconds
        chunk_lat = sum(chunk_times) / len(chunk_times)

        prep_perf_report(
            model_name="llvc",
            batch_size=1,
            inference_and_compile_time=non_streaming_time,
            inference_time=non_streaming_time,
            expected_compile_time=30.0,
            expected_inference_time=non_streaming_time,
            comments="device_n300",
        )
        sidecar = _write_streaming_sidecar(
            "device_n300",
            {
                "model": "llvc",
                "device": "wormhole_b0",
                "audio_seconds": f"{seconds:.3f}",
                "chunk_size": str(chunk_size),
                "chunk_latency_ms": f"{chunk_lat * 1000:.3f}",
                "streaming_rtf": f"{rtf:.4f}",
                "streaming_meets_stage1": str(rtf < STAGE1_RTF_THRESHOLD).lower(),
                "streaming_meets_stage3": str(rtf < STAGE3_RTF_THRESHOLD).lower(),
                "decoder_tokens_per_sec": f"{x.shape[-1] / streaming_time:.2f}",
            },
        )
        logger.info(f"Wrote streaming perf sidecar: {sidecar}; RTF={rtf:.3f}")

        # The bounty makes Stage 1 a hard requirement.
        assert rtf < STAGE1_RTF_THRESHOLD, (
            f"Streaming RTF {rtf:.3f} >= {STAGE1_RTF_THRESHOLD}; Stage 1 not met."
        )
    finally:
        try:
            ttnn.close_device(device)
        except Exception:
            pass
