# LLVC on Tenstorrent TTNN

This directory contains the implementation of the Low-Latency Low-Resource Voice Conversion (LLVC) model using TTNN.

## Directory Structure

*   `tt/`: TTNN implementation of the model components (`model.py`, `cached_convnet.py`).
*   `demo/`: Demo script to run inference (`demo.py`).
*   `tests/`: Unit tests (`test_units.py`).
*   `reference/`: Original PyTorch implementation and utils.
*   `checkpoints/`: Model checkpoints.

## Setup

1.  Ensure `ttnn` and `torch` are installed.
2.  Install dependencies: `pip install -r reference/requirements.txt` (if available) or `pip install torch torchaudio librosa numpy scipy`.
3.  Download the model checkpoint:
    ```bash
    mkdir -p checkpoints
    wget -O checkpoints/G_500000.pth https://huggingface.co/KoeAI/llvc/resolve/main/models/checkpoints/llvc/G_500000.pth
    ```

## Running the Demo

To run the demo, which performs inference on a dummy input (or you can modify it to load an audio file):

```bash
python3 demo/demo.py
```

This script will:
1.  Load the PyTorch model and weights.
2.  Initialize the TTNN model and copy weights.
3.  Run inference on both models.
4.  Compare shapes (and potentially output values if device is available).

## Running Tests

```bash
PYTHONPATH=. pytest models/demos/audio/llvc/tests/test_units.py
```

## Implementation Details

*   **CachedConvNet**: Implemented using `ttnn.conv1d` with `HEIGHT_SHARDED` memory config for efficiency.
*   **Context Management**: Streaming context is managed as a list of tensors on device, updated at each step to avoid large data transfers.
*   **Decoder**: Uses `ttnn.transformer.scaled_dot_product_attention`. To match the "causal unfold" logic of the reference (sliding window attention), the implementation performs unfolding. Currently this uses a CPU fallback (`ttnn.to_torch` -> `unfold` -> `ttnn.from_torch`) because `unfold` is not natively available in `ttnn` for the required dimensions. Given the small window size and chunk size, this overhead is managed.
*   **Optimizations**:
    *   Conv1d operations use `HEIGHT_SHARDED` layout.
    *   Weights are pre-converted to `bfloat16`.
    *   L1 memory config is used where appropriate (default `DRAM` for large buffers, `L1` for intermediates can be enabled).

## Status

*   Stage 1 (Bring-up): Implemented.
*   Stage 2 (Basic Optimizations): Sharding enabled for Convolutions.
