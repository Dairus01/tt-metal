
import torch
import ttnn
import pytest
import sys
from pathlib import Path

# Add reference to path
sys.path.append(str(Path(__file__).parent.parent / "reference"))
from model import Net, DilatedCausalConvEncoder, CausalTransformerDecoder, MaskNet
from cached_convnet import CachedConvNet

from models.demos.audio.llvc.tt.model import TtNet, TtDilatedCausalConvEncoder, TtCausalTransformerDecoder, TtMaskNet
from models.demos.audio.llvc.tt.cached_convnet import TtCachedConvNet

def assert_pcc(expected, actual, pcc=0.99):
    from ttnn import pearson_correlation_coefficient
    # Convert to torch
    if isinstance(actual, ttnn.Tensor):
        actual = ttnn.to_torch(actual)

    # Check shapes
    assert expected.shape == actual.shape, f"Shape mismatch: {expected.shape} vs {actual.shape}"

    # Compute PCC
    res = pearson_correlation_coefficient(expected, actual)
    assert res >= pcc, f"PCC check failed: {res} < {pcc}"

@pytest.fixture
def device():
    dev = ttnn.open_device(device_id=0)
    ttnn.SetDefaultDevice(dev)
    yield dev
    ttnn.close_device(dev)

def test_cached_convnet(device):
    # Config
    num_channels = 16
    out_channels = [16, 16]
    kernel_sizes = [3, 3]
    dilations = [1, 2]
    dropout = 0.0

    # Init PT
    pt_model = CachedConvNet(num_channels, kernel_sizes, dilations, dropout, 'add', False, out_channels, False)
    pt_model.eval()

    # Init TT
    tt_model = TtCachedConvNet(num_channels, kernel_sizes, dilations, dropout, 'add', False, out_channels, False, device)

    # Copy weights
    for i, layer in enumerate(tt_model.layers):
        pt_layer = pt_model.down_convs[i]
        # CausalConvBlock
        w = pt_layer.conv[0].weight
        b = pt_layer.conv[0].bias
        layer.set_weights(
            ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device),
            ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        )

    # Input
    B, L, C = 1, 100, num_channels
    x = torch.randn(B, C, L) # PT input

    # TT input [B, 1, L, C]
    tt_x = ttnn.from_torch(x.permute(0, 2, 1).unsqueeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Context
    ctx = pt_model.init_ctx_buf(B, torch.device('cpu'))
    tt_ctx = tt_model.init_ctx_buf(B)

    # Forward
    with torch.no_grad():
        pt_out, pt_ctx = pt_model(x, ctx)
        tt_out, tt_ctx_out = tt_model(tt_x, tt_ctx)

    # Compare Output
    # TT out: [B, 1, L, C]
    tt_out_torch = ttnn.to_torch(tt_out).squeeze(1).permute(0, 2, 1)

    assert_pcc(pt_out, tt_out_torch)

    # Compare Context
    # Need to verify context logic match.
    # pt_ctx is [B, C, TotalBufLen]
    # tt_ctx_out is List[[B, 1, BufLen_i, C_i]]
    # We can reconstruct or just check shapes.

    pass

def test_encoder(device):
    C = 32
    layers = 2
    k = 3

    pt_model = DilatedCausalConvEncoder(C, layers, k)
    pt_model.eval()

    tt_model = TtDilatedCausalConvEncoder(C, layers, k, device)

    # Weights
    for i, layer in enumerate(tt_model.layers):
        pt_layer = pt_model.dcc_layers[i] # DepthwiseSeparableConv
        layer.set_weights(
            ttnn.from_torch(pt_layer.layers[0].weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device),
            ttnn.from_torch(pt_layer.layers[0].bias, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device),
            ttnn.from_torch(pt_layer.layers[3].weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device),
            ttnn.from_torch(pt_layer.layers[3].bias, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device),
            ttnn.from_torch(pt_layer.layers[1].weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device),
            ttnn.from_torch(pt_layer.layers[1].bias, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device),
            ttnn.from_torch(pt_layer.layers[4].weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device),
            ttnn.from_torch(pt_layer.layers[4].bias, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        )

    B, L = 1, 50
    x = torch.randn(B, C, L)
    tt_x = ttnn.from_torch(x.permute(0, 2, 1).unsqueeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    ctx = pt_model.init_ctx_buf(B, torch.device('cpu'))
    tt_ctx = tt_model.init_ctx_buf(B)

    with torch.no_grad():
        pt_out, _ = pt_model(x, ctx)
        tt_out, _ = tt_model(tt_x, tt_ctx)

    tt_out_torch = ttnn.to_torch(tt_out).squeeze(1).permute(0, 2, 1)
    assert_pcc(pt_out, tt_out_torch)

# More tests...
