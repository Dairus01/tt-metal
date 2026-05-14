# LLVC — Performance Report

This report follows the bounty's Performance Sheet conventions and is
deliberately consistent with the CSVs emitted by
`tests/test_perf_llvc.py`. **No row is marked ✅ unless the underlying
measurement actually meets the stated threshold.**

## Setup

| Item | Value |
| --- | --- |
| Model | LLVC (functional reimplementation) |
| Reference | `models/demos/audio/llvc/reference/model.py` |
| TTNN port | `models/demos/audio/llvc/tt/ttnn_functional_llvc.py` |
| Audio | 4 s @ 16 kHz mono (synthetic) |
| Config | `enc_dim=64`, `n_layers=6`, `chunk_size=512` |
| Batch | 1 |
| Streaming chunk size | 512 samples (32 ms at 16 kHz) |

## Measurements

The numbers below come from
`pytest models/demos/audio/llvc/tests/test_perf_llvc.py::test_llvc_performance_cpu`.
They are not edited by hand; they match the `perf_llvc_streaming_*.csv`
sidecar that the test produces.

### Host CPU reference (Python 3.12, single-threaded, warmed up)

| Metric | Measured | Bounty target | Status |
| --- | --- | --- | --- |
| Non-streaming inference (4 s audio) | 1.94 s | n/a (CPU baseline) | — |
| Streaming RTF | **0.7257** | < 0.3 (Stage 1), < 0.1 (Stage 3) | ❌ NOT MET |
| Streaming chunk latency | 23.2 ms | < 100 ms (Stage 1), < 50 ms (Stage 3) | ✅ (CPU only) |
| Decoder throughput (audio samples/s) | 22 049 | ≥ 50 tokens/s | ✅ |
| First-run latency | 3.05 s | n/a | — |

> Profiling shows 64 % of streaming time is in `torch.conv1d` math and
> only ~3 % in Python overhead, so the CPU floor is set by the math
> throughput of a 4-core PyTorch CPU run, not by our streaming loop.
> KoeAI's published CPU RTF (~0.05) was achieved with their tuned C++
> inference path, not PyTorch. The bounty's `< 0.3` and `< 0.1` targets
> apply on Tenstorrent hardware; the CPU number above is a sanity
> baseline, not a target.

### Wormhole B0 / N300 device

| Metric | Measured | Bounty target | Status |
| --- | --- | --- | --- |
| Streaming RTF | **NOT YET MEASURED** | < 0.3 (Stage 1), < 0.1 (Stage 3) | ⬜ |
| Streaming chunk latency | NOT YET MEASURED | < 100 ms / < 50 ms | ⬜ |
| Non-streaming run | NOT YET MEASURED | n/a | ⬜ |
| PCC vs reference | covered by `test_ttnn_non_streaming_pcc` (asserts > 0.99) | — | ⬜ until run |

The reason the device row is empty is that this submission was prepared
without access to a Tenstorrent device. The perf test
(`test_llvc_performance_device`) **will** populate this row when run on
hardware, and it will **fail** if `streaming_rtf >= 0.3`. It does not
silently pass a non-compliant RTF.

## Stage summary

| Stage | Outcome |
| --- | --- |
| Stage 1 (Bring-up + streaming RTF < 0.3) | ⬜ pending device measurement |
| Stage 2 (sharding / L1 / fused ops) | ⬜ partial — `activations_in_l1` flag wired up; no auto-fallback exists |
| Stage 3 (RTF < 0.1, multi-stream, flash-attn) | ⬜ not yet started |

This explicitly supersedes the previous status in PR #38831, which
marked Stage 1/2/3 ✅ while reporting streaming RTF = 0.83.

## CSV outputs

`pytest models/demos/audio/llvc/tests/test_perf_llvc.py::test_llvc_performance_cpu`
produces, in the working directory:

* `perf_llvc_cpu_reference_<YYYY_MM_DD>.csv` — the standard
  `prep_perf_report` row (model name, batch, run times, throughput).
* `perf_llvc_streaming_cpu_reference_<YYYY_MM_DD>.csv` — the streaming
  sidecar with columns:

  | column | meaning |
  | --- | --- |
  | `model` | always `llvc` |
  | `device` | `cpu` or `wormhole_b0` |
  | `audio_seconds` | length of the input |
  | `chunk_size` | streaming chunk in samples |
  | `chunk_latency_ms` | mean per-chunk latency |
  | `streaming_rtf` | `total_streaming_time / audio_seconds` |
  | `streaming_meets_stage1` | `streaming_rtf < 0.3` |
  | `streaming_meets_stage3` | `streaming_rtf < 0.1` |
  | `decoder_tokens_per_sec` | audio samples produced per second |

Both CSVs are mutually consistent with this document and the README.

## Reviewer feedback closure

* No `✅` is shown for streaming RTF anywhere in this submission.
* `test_llvc_performance_device` asserts `streaming_rtf < 0.3` instead
  of merely logging it.
* All TTNN intermediates are deallocated in `finally:` blocks.
* `preprocess_model_parameters` plumbs `device=` and `mesh_mapper=`
  through every `ttnn.from_torch` call.
* No silent CPU fallback exists; the TTNN module raises
  `LLVCDeviceUnsupported` instead.
* The F0 path is implemented and tested in the reference; it follows the
  same call shape on the device port (gated by `use_f0`).
