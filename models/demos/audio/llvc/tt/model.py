
import torch
import ttnn
import math
from models.demos.audio.llvc.tt.cached_convnet import TtCachedConvNet

def mod_pad(x, chunk_size, pad):
    mod = 0
    if (x.shape[-1] % chunk_size) != 0:
        mod = chunk_size - (x.shape[-1] % chunk_size)

    x = torch.nn.functional.pad(x, (0, mod))
    if pad != (0, 0):
        x = torch.nn.functional.pad(x, pad)
    return x, mod

class TtDepthwiseSeparableConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, device):
        super().__init__()
        self.device = device
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_config = ttnn.Conv1dConfig(
            dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        self.depthwise_weight = None
        self.depthwise_bias = None
        self.pointwise_weight = None
        self.pointwise_bias = None
        self.ln1_weight = None
        self.ln1_bias = None
        self.ln2_weight = None
        self.ln2_bias = None

    def set_weights(self, dw_w, dw_b, pw_w, pw_b, ln1_w, ln1_b, ln2_w, ln2_b):
        self.depthwise_weight = dw_w
        self.depthwise_bias = dw_b
        self.pointwise_weight = pw_w
        self.pointwise_bias = pw_b
        self.ln1_weight = ln1_w
        self.ln1_bias = ln1_b
        self.ln2_weight = ln2_w
        self.ln2_bias = ln2_b

    def forward(self, x):
        out = ttnn.conv1d(
            input_tensor=x,
            weight_tensor=self.depthwise_weight,
            bias_tensor=self.depthwise_bias,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            batch_size=x.shape[0],
            input_length=x.shape[2],
            kernel_size=self.kernel_size,
            stride=1,
            padding=0,
            dilation=self.dilation,
            groups=self.in_channels,
            dtype=ttnn.bfloat16,
            conv_config=self.conv_config,
            compute_config=self.compute_config
        )
        out = ttnn.layer_norm(out, weight=self.ln1_weight, bias=self.ln1_bias)
        out = ttnn.relu(out)

        out = ttnn.conv1d(
            input_tensor=out,
            weight_tensor=self.pointwise_weight,
            bias_tensor=self.pointwise_bias,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=out.shape[0],
            input_length=out.shape[2],
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            dtype=ttnn.bfloat16,
            conv_config=self.conv_config,
            compute_config=self.compute_config
        )
        out = ttnn.layer_norm(out, weight=self.ln2_weight, bias=self.ln2_bias)
        out = ttnn.relu(out)
        return out

class TtDilatedCausalConvEncoder(torch.nn.Module):
    def __init__(self, channels, num_layers, kernel_size, device):
        super().__init__()
        self.device = device
        self.channels = channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.buf_lengths = [(kernel_size - 1) * 2**i for i in range(num_layers)]
        self.layers = []
        for i in range(num_layers):
            layer = TtDepthwiseSeparableConv(channels, channels, kernel_size, stride=1, padding=0, dilation=2**i, device=device)
            self.layers.append(layer)

    def init_ctx_buf(self, batch_size):
        ctx = []
        for i in range(self.num_layers):
            buf_len = self.buf_lengths[i]
            tensor = torch.zeros(batch_size, 1, buf_len, self.channels)
            t = ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
            ctx.append(t)
        return ctx

    def forward(self, x, ctx_buf):
        new_ctx = []
        for i, layer in enumerate(self.layers):
            buf_len = self.buf_lengths[i]
            dcc_in = ttnn.concat([ctx_buf[i], x], dim=2)

            ctx_update = ttnn.slice(
                dcc_in,
                (0, 0, dcc_in.shape[2] - buf_len, 0),
                (dcc_in.shape[0], 1, dcc_in.shape[2], dcc_in.shape[3])
            )
            new_ctx.append(ctx_update)

            out = layer(dcc_in)
            x = ttnn.add(x, out)
        return x, new_ctx

class TtCausalTransformerDecoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, device):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.nhead = nhead

        self.q_weight = None; self.q_bias = None
        self.k_weight = None; self.k_bias = None
        self.v_weight = None; self.v_bias = None
        self.out_weight = None; self.out_bias = None

        self.enc_q_weight = None; self.enc_q_bias = None
        self.enc_k_weight = None; self.enc_k_bias = None
        self.enc_v_weight = None; self.enc_v_bias = None
        self.enc_out_weight = None; self.enc_out_bias = None

        self.linear1_weight = None; self.linear1_bias = None
        self.linear2_weight = None; self.linear2_bias = None

        self.norm1_weight = None; self.norm1_bias = None
        self.norm2_weight = None; self.norm2_bias = None
        self.norm3_weight = None; self.norm3_bias = None

    def set_weights(self, qw, qb, kw, kb, vw, vb, ow, ob,
                    eqw, eqb, ekw, ekb, evw, evb, eow, eob,
                    l1w, l1b, l2w, l2b,
                    n1w, n1b, n2w, n2b, n3w, n3b):
        self.q_weight = qw; self.q_bias = qb
        self.k_weight = kw; self.k_bias = kb
        self.v_weight = vw; self.v_bias = vb
        self.out_weight = ow; self.out_bias = ob

        self.enc_q_weight = eqw; self.enc_q_bias = eqb
        self.enc_k_weight = ekw; self.enc_k_bias = ekb
        self.enc_v_weight = evw; self.enc_v_bias = evb
        self.enc_out_weight = eow; self.enc_out_bias = eob

        self.linear1_weight = l1w; self.linear1_bias = l1b
        self.linear2_weight = l2w; self.linear2_bias = l2b

        self.norm1_weight = n1w; self.norm1_bias = n1b
        self.norm2_weight = n2w; self.norm2_bias = n2b
        self.norm3_weight = n3w; self.norm3_bias = n3b

    def forward(self, tgt, memory, tgt_len_chunk):
        tgt_last_tok = ttnn.slice(
            tgt,
            (0, 0, tgt.shape[2] - tgt_len_chunk, 0),
            (tgt.shape[0], 1, tgt.shape[2], tgt.shape[3])
        )

        Q = ttnn.linear(tgt_last_tok, self.q_weight, bias=self.q_bias)
        K = ttnn.linear(tgt, self.k_weight, bias=self.k_bias)
        V = ttnn.linear(tgt, self.v_weight, bias=self.v_bias)

        Q = ttnn.experimental.nlp_create_qkv_heads(Q, num_heads=self.nhead)[0]
        K = ttnn.experimental.nlp_create_qkv_heads(K, num_heads=self.nhead)[0]
        V = ttnn.experimental.nlp_create_qkv_heads(V, num_heads=self.nhead)[0]

        scale = (self.d_model // self.nhead) ** -0.5

        attn_out = ttnn.transformer.scaled_dot_product_attention(
            Q, K, V, scale=scale, is_causal=False
        )

        attn_out = ttnn.experimental.nlp_concat_heads(attn_out)
        attn_out = ttnn.linear(attn_out, self.out_weight, bias=self.out_bias)

        tgt_last_tok = ttnn.add(tgt_last_tok, attn_out)
        tgt_last_tok = ttnn.layer_norm(tgt_last_tok, weight=self.norm1_weight, bias=self.norm1_bias)

        if memory is not None:
            Q = ttnn.linear(tgt_last_tok, self.enc_q_weight, bias=self.enc_q_bias)
            K = ttnn.linear(memory, self.enc_k_weight, bias=self.enc_k_bias)
            V = ttnn.linear(memory, self.enc_v_weight, bias=self.enc_v_bias)

            Q = ttnn.experimental.nlp_create_qkv_heads(Q, num_heads=self.nhead)[0]
            K = ttnn.experimental.nlp_create_qkv_heads(K, num_heads=self.nhead)[0]
            V = ttnn.experimental.nlp_create_qkv_heads(V, num_heads=self.nhead)[0]

            attn_out = ttnn.transformer.scaled_dot_product_attention(
                Q, K, V, scale=scale, is_causal=False
            )

            attn_out = ttnn.experimental.nlp_concat_heads(attn_out)
            attn_out = ttnn.linear(attn_out, self.enc_out_weight, bias=self.enc_out_bias)

            tgt_last_tok = ttnn.add(tgt_last_tok, attn_out)
            tgt_last_tok = ttnn.layer_norm(tgt_last_tok, weight=self.norm2_weight, bias=self.norm2_bias)

        ffn_out = ttnn.linear(tgt_last_tok, self.linear1_weight, bias=self.linear1_bias)
        ffn_out = ttnn.relu(ffn_out)
        ffn_out = ttnn.linear(ffn_out, self.linear2_weight, bias=self.linear2_bias)

        tgt_last_tok = ttnn.add(tgt_last_tok, ffn_out)
        tgt_last_tok = ttnn.layer_norm(tgt_last_tok, weight=self.norm3_weight, bias=self.norm3_bias)

        return tgt_last_tok

class TtCausalTransformerDecoder(torch.nn.Module):
    def __init__(self, model_dim, ctx_len, chunk_size, num_layers, nhead, device):
        super().__init__()
        self.device = device
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.ctx_len = ctx_len
        self.chunk_size = chunk_size
        self.layers = []
        for i in range(num_layers):
            layer = TtCausalTransformerDecoderLayer(model_dim, nhead, 2*model_dim, 0.0, device)
            self.layers.append(layer)

        self.pos_enc_table = self._get_pos_enc_table(model_dim, 2000).to(torch.bfloat16)
        self.tt_pos_enc = ttnn.from_torch(self.pos_enc_table, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def _get_pos_enc_table(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0).unsqueeze(0)

    def init_ctx_buf(self, batch_size):
        ctx = []
        for i in range(self.num_layers + 1):
            tensor = torch.zeros(batch_size, 1, self.ctx_len, self.model_dim)
            t = ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
            ctx.append(t)
        return ctx

    def _causal_unfold_ttnn(self, x):
        # x: [B, 1, L_total, C]
        # output: [B*Chunk, 1, WindowSize, C]

        B = x.shape[0]
        L_total = x.shape[2]
        C = x.shape[3]
        window_size = self.ctx_len + 1
        chunk = L_total - self.ctx_len

        slices = []
        for i in range(chunk):
            s = ttnn.slice(
                x,
                (0, 0, i, 0),
                (B, 1, i + window_size, C)
            )
            slices.append(s)

        # Concat along batch dim?
        # ttnn.concat usually concats on existing dim.
        # slices[i] is [B, 1, Window, C].
        # We want [B*Chunk, 1, Window, C].
        # If we concat on dim 0, we get [B*Chunk, 1, Window, C] IF B=1.
        # If B > 1, we might need interleave logic?
        # For now assume B=1 as standard inference.

        if B == 1:
            unfolded = ttnn.concat(slices, dim=0)
        else:
            # Not supported easily without proper reshape/permute logic
            # Fallback to CPU if B > 1 (rare for streaming)
            return self._causal_unfold_fallback(x)

        return unfolded

    def _causal_unfold_fallback(self, x):
        x_torch = ttnn.to_torch(x)
        B, _, L_total, C = x_torch.shape
        window_size = self.ctx_len + 1
        unfolded = x_torch.squeeze(1).unfold(1, window_size, 1)
        unfolded = unfolded.permute(0, 1, 3, 2)
        unfolded = unfolded.reshape(-1, 1, window_size, C)
        return ttnn.from_torch(unfolded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

    def forward(self, tgt, mem, ctx_buf):
        new_ctx = []
        mem = ttnn.concat([ctx_buf[0], mem], dim=2)

        ctx_update = ttnn.slice(
            mem,
            (0, 0, mem.shape[2] - self.ctx_len, 0),
            (mem.shape[0], 1, mem.shape[2], mem.shape[3])
        )
        new_ctx.append(ctx_update)

        mem_ctx = self._causal_unfold_ttnn(mem)

        window_size = mem_ctx.shape[2]
        pe_slice = ttnn.slice(
            self.tt_pos_enc,
            (0, 0, 0, 0),
            (1, 1, window_size, self.model_dim)
        )
        mem_ctx = ttnn.add(mem_ctx, pe_slice)

        for i, layer in enumerate(self.layers):
            tgt_in = ttnn.concat([ctx_buf[i+1], tgt], dim=2)

            ctx_update = ttnn.slice(
                tgt_in,
                (0, 0, tgt_in.shape[2] - self.ctx_len, 0),
                (tgt_in.shape[0], 1, tgt_in.shape[2], tgt_in.shape[3])
            )
            new_ctx.append(ctx_update)

            tgt_ctx = self._causal_unfold_ttnn(tgt_in)

            if i == 0:
                tgt_ctx = ttnn.add(tgt_ctx, pe_slice)

            tgt_out = layer(tgt_ctx, mem_ctx, 1)

            B = tgt.shape[0]
            Chunk = tgt.shape[2]
            C = tgt.shape[3]

            tgt_out = ttnn.reshape(tgt_out, (B, 1, Chunk, C))
            tgt = tgt_out

        return tgt, new_ctx

class TtMaskNet(torch.nn.Module):
    def __init__(self, enc_dim, num_enc_layers, dec_dim, dec_buf_len,
                 dec_chunk_size, num_dec_layers, device):
        super().__init__()
        self.encoder = TtDilatedCausalConvEncoder(enc_dim, num_enc_layers, 3, device)
        self.decoder = TtCausalTransformerDecoder(dec_dim, dec_buf_len, dec_chunk_size, num_dec_layers, 8, device)

        self.proj_e2d_e_w = None; self.proj_e2d_e_b = None
        self.proj_e2d_l_w = None; self.proj_e2d_l_b = None
        self.proj_d2e_w = None; self.proj_d2e_b = None

        self.conv_config = ttnn.Conv1dConfig(
            dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        self.device = device
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim

    def set_weights(self, pe_w, pe_b, pl_w, pl_b, pd_w, pd_b):
        self.proj_e2d_e_w = pe_w; self.proj_e2d_e_b = pe_b
        self.proj_e2d_l_w = pl_w; self.proj_e2d_l_b = pl_b
        self.proj_d2e_w = pd_w; self.proj_d2e_b = pd_b

    def forward(self, x, l, enc_buf, dec_buf):
        e, enc_buf = self.encoder(x, enc_buf)
        l = ttnn.mul(l, e)

        e_proj = ttnn.conv1d(
            input_tensor=e,
            weight_tensor=self.proj_e2d_e_w,
            bias_tensor=self.proj_e2d_e_b,
            device=self.device,
            in_channels=self.enc_dim,
            out_channels=self.dec_dim,
            batch_size=e.shape[0],
            input_length=e.shape[2],
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=self.dec_dim,
            dtype=ttnn.bfloat16,
            conv_config=self.conv_config,
            compute_config=self.compute_config
        )

        m_proj = ttnn.conv1d(
            input_tensor=l,
            weight_tensor=self.proj_e2d_l_w,
            bias_tensor=self.proj_e2d_l_b,
            device=self.device,
            in_channels=self.enc_dim,
            out_channels=self.dec_dim,
            batch_size=l.shape[0],
            input_length=l.shape[2],
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=self.dec_dim,
            dtype=ttnn.bfloat16,
            conv_config=self.conv_config,
            compute_config=self.compute_config
        )

        m, dec_buf = self.decoder(m_proj, e_proj, dec_buf)

        m = ttnn.conv1d(
            input_tensor=m,
            weight_tensor=self.proj_d2e_w,
            bias_tensor=self.proj_d2e_b,
            device=self.device,
            in_channels=self.dec_dim,
            out_channels=self.enc_dim,
            batch_size=m.shape[0],
            input_length=m.shape[2],
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=self.dec_dim,
            dtype=ttnn.bfloat16,
            conv_config=self.conv_config,
            compute_config=self.compute_config
        )

        m = ttnn.add(l, m)
        return m, enc_buf, dec_buf

class TtNet(torch.nn.Module):
    def __init__(self, device, enc_dim=512, L=16, dec_dim=256,
                 dec_buf_len=13, dec_chunk_size=13, num_dec_layers=1, num_enc_layers=8,
                 out_buf_len=4, convnet_config=None):
        super().__init__()
        self.device = device
        self.enc_dim = enc_dim
        self.L = L
        self.out_buf_len = out_buf_len

        self.in_conv_weight = None
        self.out_conv_weight = None

        self.emb_linear1_w = None; self.emb_linear1_b = None
        self.emb_norm1_w = None; self.emb_norm1_b = None
        self.emb_linear2_w = None; self.emb_linear2_b = None
        self.emb_norm2_w = None; self.emb_norm2_b = None

        self.mask_gen = TtMaskNet(enc_dim, num_enc_layers, dec_dim, dec_buf_len, dec_chunk_size, num_dec_layers, device)

        self.convnet_pre = None
        if convnet_config and convnet_config['convnet_prenet']:
            self.convnet_pre = TtCachedConvNet(
                1, convnet_config['kernel_sizes'], convnet_config['dilations'],
                convnet_config['dropout'], convnet_config['combine_residuals'],
                convnet_config['use_residual_blocks'], convnet_config['out_channels'],
                use_2d=False, device=device)

        # Derived
        self.in_conv_kernel_size = 3 * L
        self.out_conv_kernel_size = (out_buf_len + 1) * L
        self.out_padding = out_buf_len * L

    def init_buffers(self, batch_size):
        enc_buf = self.mask_gen.encoder.init_ctx_buf(batch_size)
        dec_buf = self.mask_gen.decoder.init_ctx_buf(batch_size)
        out_buf_torch = torch.zeros(batch_size, 1, self.out_buf_len, self.enc_dim)
        out_buf = ttnn.from_torch(out_buf_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        pre_ctx = None
        if self.convnet_pre:
            pre_ctx = self.convnet_pre.init_ctx_buf(batch_size)

        return enc_buf, dec_buf, out_buf, pre_ctx

    def forward(self, x, enc_buf, dec_buf, out_buf, pre_ctx=None):
        if self.convnet_pre:
            conv_out, pre_ctx = self.convnet_pre(x, pre_ctx)
            x = ttnn.add(x, conv_out)

        x = ttnn.conv1d(
            input_tensor=x,
            weight_tensor=self.in_conv_weight,
            device=self.device,
            in_channels=1,
            out_channels=self.enc_dim,
            batch_size=x.shape[0],
            input_length=x.shape[2],
            kernel_size=self.in_conv_kernel_size,
            stride=self.L,
            padding=0,
            dilation=1,
            groups=1,
            dtype=ttnn.bfloat16,
            conv_config=ttnn.Conv1dConfig(dtype=ttnn.bfloat16, shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
            compute_config=ttnn.init_device_compute_kernel_config(self.device.arch(), math_fidelity=ttnn.MathFidelity.LoFi)
        )
        x = ttnn.relu(x)

        label = torch.zeros(x.shape[0], 1, 1, 1)
        l = ttnn.from_torch(label, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        l = ttnn.linear(l, self.emb_linear1_w, bias=self.emb_linear1_b)
        l = ttnn.layer_norm(l, weight=self.emb_norm1_w, bias=self.emb_norm1_b)
        l = ttnn.relu(l)
        l = ttnn.linear(l, self.emb_linear2_w, bias=self.emb_linear2_b)
        l = ttnn.layer_norm(l, weight=self.emb_norm2_w, bias=self.emb_norm2_b)
        l = ttnn.relu(l)

        m, enc_buf, dec_buf = self.mask_gen(x, l, enc_buf, dec_buf)

        x = ttnn.mul(x, m)

        x = ttnn.concat([out_buf, x], dim=2)

        out_buf = ttnn.slice(
            x,
            (0, 0, x.shape[2] - self.out_buf_len, 0),
            (x.shape[0], 1, x.shape[2], x.shape[3])
        )

        x = ttnn.conv_transpose2d(
            input_tensor=x,
            weight_tensor=self.out_conv_weight,
            device=self.device,
            in_channels=self.enc_dim,
            out_channels=1,
            batch_size=x.shape[0],
            input_height=1,
            input_width=x.shape[2],
            kernel_size=(1, self.out_conv_kernel_size),
            stride=(1, self.L),
            padding=(0, self.out_padding),
            output_padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            dtype=ttnn.bfloat16
        )

        x = ttnn.tanh(x)

        return x, enc_buf, dec_buf, out_buf, pre_ctx
