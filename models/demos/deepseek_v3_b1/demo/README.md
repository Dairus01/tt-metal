# DeepSeek V3 B1 Demo CLI

This folder contains a lightweight CLI for running the current DeepSeek V3 B1 host-interface demo.

The demo runs prefill + decode over `DeepSeekV3` sockets and streams decoded text to stdout.

## Files

- `cli.py`: runtime entrypoint, tokenizer/model/device wiring.
- `runner.py`: pure orchestration logic (easy to unit test).

## Requirements

1. Source environment from repo root:

   ```bash
   source source.sh
   ```

2. Enable slow dispatch mode:

   ```bash
   export TT_METAL_SLOW_DISPATCH_MODE=1
   ```

   The CLI raises a runtime error if slow dispatch mode is not enabled.

## Run

From repo root:

```bash
python -m models.demos.deepseek_v3_b1.demo.cli --prompt "Once upon a time" --max-new-tokens 128
```

### Common options

- `--prompt` (default: `""`)
- `--max-new-tokens` (default: `128`)
- `--tokenizer` (default: `deepseek-ai/DeepSeek-V3`)
- `--loopback-mode` / `--no-loopback-mode` (default: `--loopback-mode`)
- `--mesh-height` (default: `1`)
- `--mesh-width` (default: `1`)

## Current behavior

- Prompt is printed when the first decode chunk is emitted.
- Decode text is streamed token-by-token to stdout.
- Current bring-up path is loopback-focused; non-loopback mode is exposed but not covered by active B1 unit tests.

## Tests

Runner unit tests (no device required):

```bash
python -m pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_demo.py -v --tb=short --noconftest
```

Stress test (device-required, 64K decode tokens):

```bash
TT_METAL_SLOW_DISPATCH_MODE=1 python -m pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_demo.py::test_demo_decode_stress_64k_tokens -v --tb=short
```

Notes:

- Marked `slow` and `skip_post_commit` to avoid normal post-commit coverage.
- Test runs loopback-mode decode for `65536` generated tokens.
