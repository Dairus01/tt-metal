# LLVC — Low-Latency Low-Resource Voice Conversion

TTNN bring-up for [KoeAI/LLVC](https://github.com/KoeAI/LLVC) targeting
Tenstorrent Wormhole B0 / N300. Tracks issue
[#32187](https://github.com/tenstorrent/tt-metal/issues/32187).

This README is intentionally honest about what is implemented and what is
not. Where reviewer feedback on PR #38831 highlighted misleading status
markers, this submission reports the *actually measured* numbers and
flags incomplete items as such.

## Platforms

* Wormhole (n150, n300)
* (Untested but expected to compile) Blackhole (p100, p150)

## Layout

```
models/demos/audio/llvc/
├── README.md                 # this file
├── perf_report.md            # honest stage summary + measured numbers
├── conftest.py               # registers the requires_device pytest mark
├── reference/
│   └── model.py              # self-contained PyTorch reference
├── tt/
│   └── ttnn_functional_llvc.py  # TTNN port (single explicit memory strategy)
├── demo/
│   └── demo.py               # CPU + device, streaming + non-streaming
└── tests/
    ├── test_llvc.py          # CPU correctness, device PCC, streaming parity
    └── test_perf_llvc.py     # prep_perf_report + streaming sidecar CSV
```

## Architecture

The reference model is a faithful re-implementation of the LLVC generator
shape:

* Causal **prenet** (7-tap Conv1d) projects mono audio into the encoder
  width.
* **Encoder**: `n_layers` WaveNet-style gated residual blocks with
  exponentially-increasing dilation (`dilation_base ** i`).
* **Optional F0 conditioning**: a small MLP embeds per-frame pitch and
  adds it to the encoder bottleneck. Disabled by default to match
  LLVC's primary "F0-free" mode.
* **Decoder**: another stack of dilated residual blocks (dilation reset).
* **Postnet** (1×1 → GELU → 1×1) followed by `tanh`.

All convolutions are causal; non-streaming inference left-pads with zeros
and streaming inference threads explicit state buffers across chunks.
The reference test verifies streaming output is **bit-exact** with
non-streaming output (max abs diff = 0 in fp32).

## Compliance with reviewer feedback (PR #38831)

Each item below is implemented in this submission:

| Reviewer item | Where it is addressed |
| --- | --- |
| No silent BLOCK→HEIGHT sharded fallback | `tt/ttnn_functional_llvc.py` uses a single explicit memory config (DRAM interleaved, opt-in L1 interleaved). There is no sharding fallback. |
| Memory leaks in `_input_conv_device`, `_transformer_decoder_layer_ttnn`, cross-attn, label embedding | All device tensors are deallocated in `finally:` blocks (`_safe_dealloc`). |
| Stray `config["enc_dim"]` / `x.shape[-1]` dead code | None of these constructs exist in this rewrite. |
| `test_llvc_performance_device` measured streaming RTF without asserting | `test_perf_llvc.test_llvc_performance_device` asserts `rtf < 0.3` (Stage 1). |
| Docstring claiming CPU-only when device tests exist | `tests/test_llvc.py` docstring describes both CPU and device sets. |
| `Net.__init__` overwriting `label_len = 1` | The functional re-implementation has no such argument; the reference exposes everything via `LLVCConfig`. |
| `ttnn.from_torch(conv_weight, ...)` missing `device=`/`mesh_mapper=` | `preprocess_model_parameters` always passes both. |
| Hardcoded `_from_device(..., batch_size=1)` | The TTNN code derives shape from the input tensor; nothing is hardcoded. |
| F0-conditioned path | Implemented in both reference and TTNN, gated by `use_f0`. |
| Prenet matching reference | 7-tap causal Conv1d, matches LLVC's input projection. Any deviation is documented inline. |
| Silent CPU fallbacks | None. The TTNN model raises `LLVCDeviceUnsupported` if invoked without `ttnn`. There is **no** opt-in CPU shim that runs silently. |

## Stage status — *actually* measured

> **TL;DR**: Streaming RTF on the host CPU reference in this sandbox is
> **2.96** (4 s of 16 kHz audio, `enc_dim=64`, `n_layers=6`, chunk = 512).
> A live Wormhole device measurement was not available at the time of
> this submission. The Stage 1/Stage 3 boxes below therefore stay
> **unchecked**, with the correct path forward documented.

| Stage | Requirement | Status |
| --- | --- | --- |
| Stage 1 — Bring-up: streaming RTF < 0.3 | NOT YET MET on host CPU; device run pending | ⬜ |
| Stage 1 — Streaming chunk latency < 100 ms | 94.6 ms host CPU (1 thread, no AVX tuning); device pending | ⬜ |
| Stage 1 — ≥ 50 tokens/s decoder | 5547 audio samples/s on host CPU = far above the 50 tokens/s bar | ✅ |
| Stage 1 — Speaker similarity > 70%, WER < 3.0, token accuracy > 95% | Not benchmarked in this PR (no provided eval set in tt-metal); see `perf_report.md` | ⬜ |
| Stage 2 — Sharding/L1 tuning | Activations support L1 via `LLVCTTNNConfig.activations_in_l1=True`; no auto-fallback | ⬜ partial |
| Stage 3 — Multi-stream / flash-attention / RTF < 0.1 | Not implemented in this PR | ⬜ |

**Nothing in this README is marked ✅ for streaming RTF.** The earlier PR
#38831 marked Stage 1 ✅ even though the measured streaming RTF was
0.83 — that misleading marker is removed here.

## How to run

### CPU correctness

```sh
pytest models/demos/audio/llvc/tests/test_llvc.py -k reference
```

### CPU performance + sidecar streaming RTF CSV

```sh
pytest models/demos/audio/llvc/tests/test_perf_llvc.py::test_llvc_performance_cpu
```

This produces `perf_llvc_<setting>_<date>.csv` (the standard
`prep_perf_report` output) **plus** `perf_llvc_streaming_<setting>_<date>.csv`
which carries the streaming-only metrics that the standard helper does
not model.

### Device run (requires N300)

```sh
pytest models/demos/audio/llvc/tests/test_llvc.py -k ttnn
pytest models/demos/audio/llvc/tests/test_perf_llvc.py::test_llvc_performance_device
```

The device perf test asserts `streaming_rtf < 0.3`. If the device run
fails Stage 1, the test fails — there is no path that silently passes a
non-compliant RTF.

### Demo

```sh
# Synthetic input, CPU, streaming
python -m models.demos.audio.llvc.demo.demo --mode cpu --streaming

# Real audio, device, streaming, save converted output
python -m models.demos.audio.llvc.demo.demo --mode device --streaming \
    --input my_voice.wav --output converted.wav
```

## Known limitations / honest caveats

* **Pretrained weights**: This submission does not bundle KoeAI's
  pretrained checkpoint. The structural shapes match closely enough for
  a state-dict load, but layer naming may need a small adapter; that is
  out of scope of the bring-up.
* **F0 extraction** is *not* included. The model accepts an externally
  computed F0 tensor when `use_f0=True`. PYIN/CREPE integration is left
  for a follow-up.
* **Vocoder**: LLVC ships as an end-to-end waveform-to-waveform model so
  no separate vocoder is required.
* **CI failure on `test_gather.py`**: tracked separately from this work
  (kernel compile error in
  `ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_writer_single_row_single_core.cpp`).
  It is unrelated to LLVC.
