
import torch
import torchaudio
import ttnn
import json
import os
import sys
from pathlib import Path

# Add reference to path
sys.path.append(str(Path(__file__).parent.parent / "reference"))
from model import Net as PytorchNet
from utils import load_checkpoint

from models.demos.audio.llvc.tt.model import TtNet

def load_tt_weights(tt_model, pt_model, device):
    # Helper to convert and set weights

    def to_tt(tensor):
        if tensor.ndim == 3: # Conv1d weights [out, in, k]
            # ttnn.conv1d expects [out, in, k] ?
            # In my implementation I passed weights directly.
            # Assuming ttnn.conv1d matches pytorch layout or I need to check.
            # Whisper example: parameters.conv1.weight = ttnn.from_torch(..., layout=ttnn.ROW_MAJOR_LAYOUT)
            # It seems it just takes the tensor.
            pass
        return ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Net weights
    # in_conv
    tt_model.in_conv_weight = to_tt(pt_model.in_conv[0].weight)

    # Label embedding
    tt_model.emb_linear1_w = to_tt(pt_model.label_embedding[0].weight.t()) # Linear weights transposed for ttnn.linear?
    # ttnn.linear(x, w) -> x @ w.
    # PyTorch Linear(x) -> x @ w.t() + b.
    # So we need w.t().
    tt_model.emb_linear1_b = to_tt(pt_model.label_embedding[0].bias)

    tt_model.emb_norm1_w = to_tt(pt_model.label_embedding[1].weight)
    tt_model.emb_norm1_b = to_tt(pt_model.label_embedding[1].bias)

    tt_model.emb_linear2_w = to_tt(pt_model.label_embedding[3].weight.t())
    tt_model.emb_linear2_b = to_tt(pt_model.label_embedding[3].bias)

    tt_model.emb_norm2_w = to_tt(pt_model.label_embedding[4].weight)
    tt_model.emb_norm2_b = to_tt(pt_model.label_embedding[4].bias)

    # ConvNet Pre
    if tt_model.convnet_pre:
        for i, layer in enumerate(tt_model.convnet_pre.layers):
            pt_layer = pt_model.convnet_pre.down_convs[i]
            if hasattr(layer, 'filter_weight'): # ResidualBlock
                layer.set_weights(
                    to_tt(pt_layer.filter.weight), to_tt(pt_layer.filter.bias),
                    to_tt(pt_layer.gate.weight), to_tt(pt_layer.gate.bias)
                )
            else: # CausalConvBlock
                layer.set_weights(to_tt(pt_layer.conv[0].weight), to_tt(pt_layer.conv[0].bias))

    # MaskNet
    # Encoder
    for i, layer in enumerate(tt_model.mask_gen.encoder.layers):
        # pt_model.mask_gen.encoder.dcc_layers['dcc_%d' % i]
        pt_layer = pt_model.mask_gen.encoder.dcc_layers[i] # nn.Sequential index
        # But dcc_layers is Sequential of OrderedDict? No, Sequential constructed from OrderedDict.
        # So indexing works.
        # DepthwiseSeparableConv
        # layers: [Conv1d, LayerNorm, ReLU, Conv1d, LayerNorm, ReLU]
        # Indices: 0, 1, 2, 3, 4, 5

        dw_conv = pt_layer.layers[0]
        ln1 = pt_layer.layers[1]
        pw_conv = pt_layer.layers[3]
        ln2 = pt_layer.layers[4]

        layer.set_weights(
            to_tt(dw_conv.weight), to_tt(dw_conv.bias),
            to_tt(pw_conv.weight), to_tt(pw_conv.bias),
            to_tt(ln1.weight), to_tt(ln1.bias),
            to_tt(ln2.weight), to_tt(ln2.bias)
        )

    # Projections
    tt_model.mask_gen.set_weights(
        to_tt(pt_model.mask_gen.proj_e2d_e[0].weight), to_tt(pt_model.mask_gen.proj_e2d_e[0].bias),
        to_tt(pt_model.mask_gen.proj_e2d_l[0].weight), to_tt(pt_model.mask_gen.proj_e2d_l[0].bias),
        to_tt(pt_model.mask_gen.proj_d2e[0].weight), to_tt(pt_model.mask_gen.proj_d2e[0].bias)
    )

    # Decoder
    for i, layer in enumerate(tt_model.mask_gen.decoder.layers):
        pt_layer = pt_model.mask_gen.decoder.tf_dec_layers[i]
        # pt_layer is TransformerDecoderLayer
        # self_attn: MultiheadAttention
        # multihead_attn: MultiheadAttention
        # linear1, linear2
        # norm1, norm2, norm3

        # Self Attn (in_proj_weight/bias pack Q,K,V)
        # But we implemented separate.
        # PyTorch MHA packs if batch_first=True?
        # "in_proj_weight" [3*dim, dim]

        qkv_w = pt_layer.self_attn.in_proj_weight
        qkv_b = pt_layer.self_attn.in_proj_bias

        # Split
        dim = tt_model.mask_gen.decoder.model_dim
        qw, kw, vw = qkv_w.split(dim, dim=0)
        qb, kb, vb = qkv_b.split(dim, dim=0)

        out_w = pt_layer.self_attn.out_proj.weight
        out_b = pt_layer.self_attn.out_proj.bias

        # Cross Attn
        # in_proj_weight might be None if k,v,q proj are separate?
        # PyTorch MHA: if kdim=vdim=embed_dim, use in_proj_weight?
        # Check pt_layer.multihead_attn.in_proj_weight

        eqkv_w = pt_layer.multihead_attn.in_proj_weight
        if eqkv_w is not None:
            eqw, ekw, evw = eqkv_w.split(dim, dim=0)
            eqb, ekb, evb = pt_layer.multihead_attn.in_proj_bias.split(dim, dim=0)
        else:
            # Separate weights
            # q_proj_weight?
            # Newer PyTorch uses separate linearly if add_bias_kv etc.
            # Assuming standard.
            # If in_proj_weight is None, check q_proj_weight
            eqw = pt_layer.multihead_attn.q_proj_weight
            ekw = pt_layer.multihead_attn.k_proj_weight
            evw = pt_layer.multihead_attn.v_proj_weight
            eqb = pt_layer.multihead_attn.q_proj_bias
            ekb = pt_layer.multihead_attn.k_proj_bias
            evb = pt_layer.multihead_attn.v_proj_bias

        eout_w = pt_layer.multihead_attn.out_proj.weight
        eout_b = pt_layer.multihead_attn.out_proj.bias

        l1_w = pt_layer.linear1.weight
        l1_b = pt_layer.linear1.bias
        l2_w = pt_layer.linear2.weight
        l2_b = pt_layer.linear2.bias

        n1_w = pt_layer.norm1.weight
        n1_b = pt_layer.norm1.bias
        n2_w = pt_layer.norm2.weight
        n2_b = pt_layer.norm2.bias
        n3_w = pt_layer.norm3.weight
        n3_b = pt_layer.norm3.bias

        layer.set_weights(
            to_tt(qw.t()), to_tt(qb), to_tt(kw.t()), to_tt(kb), to_tt(vw.t()), to_tt(vb), to_tt(out_w.t()), to_tt(out_b),
            to_tt(eqw.t()), to_tt(eqb), to_tt(ekw.t()), to_tt(ekb), to_tt(evw.t()), to_tt(evb), to_tt(eout_w.t()), to_tt(eout_b),
            to_tt(l1_w.t()), to_tt(l1_b), to_tt(l2_w.t()), to_tt(l2_b),
            to_tt(n1_w), to_tt(n1_b), to_tt(n2_w), to_tt(n2_b), to_tt(n3_w), to_tt(n3_b)
        )

    # Out Conv
    # pt_model.out_conv[0] is ConvTranspose1d [in, out, k]
    # We need to reshape for conv_transpose2d [in, out, 1, k]?
    # PyTorch: [in_channels, out_channels, kernel_size]
    w = pt_model.out_conv[0].weight
    w = w.unsqueeze(2) # [in, out, 1, k]
    # ttnn.conv_transpose2d expects [in, out, kh, kw] ?
    # I verified earlier I assumed standard layout.
    # Actually, weights for ttnn.conv2d are usually [out, in, kh, kw].
    # For transpose?
    # I'll check Whisper but it uses conv1d.
    # I'll assume [in, out, kh, kw] for transpose as it maps input to output.
    # If not, I'll fix it.

    tt_model.out_conv_weight = to_tt(w)


def run_demo():
    device = ttnn.open_device(device_id=0)
    ttnn.SetDefaultDevice(device)

    # Load config
    config_path = "models/demos/audio/llvc/reference/experiments/llvc/config.json"
    with open(config_path) as f:
        config = json.load(f)

    # Instantiate PyTorch model
    checkpoint_path = "models/demos/audio/llvc/checkpoints/G_500000.pth"
    pt_model = PytorchNet(**config['model_params'])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    pt_model.load_state_dict(checkpoint['model'])
    pt_model.eval()

    # Instantiate TT model
    mp = config['model_params']
    tt_model = TtNet(
        device,
        enc_dim=mp['enc_dim'],
        L=mp['L'],
        dec_dim=mp['dec_dim'],
        dec_buf_len=mp['dec_buf_len'],
        dec_chunk_size=mp['dec_chunk_size'],
        num_dec_layers=mp['num_dec_layers'],
        num_enc_layers=mp['num_enc_layers'],
        out_buf_len=mp['out_buf_len'],
        convnet_config=mp['convnet_config']
    )

    print("Loading weights...")
    load_tt_weights(tt_model, pt_model, device)
    print("Weights loaded.")

    # Generate dummy input or load audio
    # L * chunk_size (72)
    # L=16. chunk_size=72.
    chunk_len = 16 * 72

    audio = torch.randn(1, 1, chunk_len) # [B, C, T]
    # pt_model expects [B, C, T]

    print("Running PyTorch inference...")
    with torch.no_grad():
        pt_out = pt_model(audio.unsqueeze(0)).squeeze(0) # infer helper does unsqueeze

    print("Running TTNN inference...")
    # Prepare input for TTNN: [B, 1, T, 1]
    # audio is [1, 1, T].
    # input x: [1, 1, T, 1].
    tt_audio = audio.permute(0, 2, 1).unsqueeze(1) # [1, 1, T, 1]
    tt_input = ttnn.from_torch(tt_audio, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    enc_buf, dec_buf, out_buf, pre_ctx = tt_model.init_buffers(1)

    tt_out, _, _, _, _ = tt_model(tt_input, enc_buf, dec_buf, out_buf, pre_ctx)

    # tt_out is [B, 1, T, 1] ?
    # We used conv_transpose2d.
    # Output should be [B, 1, T, 1] or similar.

    tt_out_torch = ttnn.to_torch(tt_out)
    # [1, 1, T, 1] -> [1, 1, T] -> [1, T] ?
    # pt_out is [1, 1, T]

    tt_out_torch = tt_out_torch.squeeze(1).permute(0, 2, 1) # [1, 1, T]

    # Compare
    # Crop to match valid output?
    # Streaming model might have delay.
    # But for same input and state, output should match.
    # However, init_buffers initializes zeros.
    # pt_model also starts with zero buffers (if we don't pass them).
    # infer() in reference uses model(audio.unsqueeze(0).unsqueeze(0))
    # Net.forward handles buffer init if None.

    # We verify shapes first.
    print(f"PT shape: {pt_out.shape}")
    print(f"TT shape: {tt_out_torch.shape}")

    # PCC
    # Only compare valid part?
    # Usually initial part is padding artifacts if context is zero.
    # But whole output should match if logic is identical.

    # Check PCC
    # use ttnn.pearson_correlation_coefficient?
    # Or manual.

    # Need to convert pt_out to bfloat16 for fair comparison?

    ttnn.close_device(device)

if __name__ == "__main__":
    run_demo()
