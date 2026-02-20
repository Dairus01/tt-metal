
import torch
import ttnn

class TtResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout, device):
        super().__init__()
        self.device = device
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.output_crop = dilation * (kernel_size - 1)
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

        self.filter_weight = None
        self.filter_bias = None
        self.gate_weight = None
        self.gate_bias = None

    def set_weights(self, filter_w, filter_b, gate_w, gate_b):
        self.filter_weight = filter_w
        self.filter_bias = filter_b
        self.gate_weight = gate_w
        self.gate_bias = gate_b

    def forward(self, x):
        # x: [B, 1, L_total, C_in]
        # output_crop reduces length by dilation * (k-1)

        filtered = ttnn.conv1d(
            input_tensor=x,
            weight_tensor=self.filter_weight,
            bias_tensor=self.filter_bias,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=x.shape[0],
            input_length=x.shape[2],
            kernel_size=self.kernel_size,
            stride=1,
            padding=0,
            dilation=self.dilation,
            groups=1,
            dtype=ttnn.bfloat16,
            conv_config=self.conv_config,
            compute_config=self.compute_config
        )
        filtered = ttnn.tanh(filtered)

        gated = ttnn.conv1d(
            input_tensor=x,
            weight_tensor=self.gate_weight,
            bias_tensor=self.gate_bias,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=x.shape[0],
            input_length=x.shape[2],
            kernel_size=self.kernel_size,
            stride=1,
            padding=0,
            dilation=self.dilation,
            groups=1,
            dtype=ttnn.bfloat16,
            conv_config=self.conv_config,
            compute_config=self.compute_config
        )
        gated = ttnn.sigmoid(gated)

        residual = ttnn.mul(filtered, gated)

        # Crop input to match residual length
        # x: [B, 1, L_total, C_in]
        # output_crop length at start is removed by conv
        # So we slice x from output_crop to end.

        x_cropped = ttnn.slice(
            x,
            (0, 0, self.output_crop, 0),
            (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        )

        # Handle channel mismatch
        if self.in_channels != self.out_channels:
            # Pad channels.
            # ttnn.pad arguments: input_tensor, padding, value
            # padding is ((front, back), ...) for each dim?
            # Or [front, back, front, back...]?
            # Documentation says: padding (List[Tuple[int, int]])

            # [B, 1, L, C]
            # We want to pad dim 3 (C).
            # ((0,0), (0,0), (0,0), (0, diff))

            diff = self.out_channels - self.in_channels
            padding = ((0,0), (0,0), (0,0), (0, diff))
            x_cropped = ttnn.pad(x_cropped, padding, 0)

        output = ttnn.add(x_cropped, residual)
        return output

class TtCausalConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout, device):
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
        self.weight = None
        self.bias = None

    def set_weights(self, w, b):
        self.weight = w
        self.bias = b

    def forward(self, x):
        out = ttnn.conv1d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=x.shape[0],
            input_length=x.shape[2],
            kernel_size=self.kernel_size,
            stride=1,
            padding=0,
            dilation=self.dilation,
            groups=1,
            dtype=ttnn.bfloat16,
            conv_config=self.conv_config,
            compute_config=self.compute_config
        )
        out = ttnn.leaky_relu(out)
        return out


class TtCachedConvNet(torch.nn.Module):
    def __init__(self, num_channels, kernel_sizes, dilations,
                 dropout, combine_residuals, use_residual_blocks,
                 out_channels, use_2d, device):
        super().__init__()
        self.device = device
        self.num_layers = len(kernel_sizes)
        self.layers = []
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        self.buf_lengths = [(k - 1) * d for k, d in zip(kernel_sizes, dilations)]

        self.combine_residuals = combine_residuals
        self.use_2d = use_2d

        for i in range(self.num_layers):
            in_ch = num_channels if i == 0 else out_channels[i - 1]
            out_ch = out_channels[i]
            if use_residual_blocks:
                layer = TtResidualBlock(in_ch, out_ch, kernel_sizes[i], dilations[i], dropout, device)
            else:
                layer = TtCausalConvBlock(in_ch, out_ch, kernel_sizes[i], dilations[i], dropout, device)
            self.layers.append(layer)

    def init_ctx_buf(self, batch_size):
        # Return a list of tensors
        ctx = []
        for i in range(self.num_layers):
            buf_len = self.buf_lengths[i]
            if i == 0:
                channels = self.layers[0].in_channels
            else:
                channels = self.layers[i-1].out_channels # Input to layer i

            # [B, 1, buf_len, C]
            tensor = torch.zeros(batch_size, 1, buf_len, channels)
            t = ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
            ctx.append(t)
        return ctx

    def forward(self, x, ctx):
        # x: [B, 1, L, C]
        # ctx: List[ttnn.Tensor]

        new_ctx = []

        for i, layer in enumerate(self.layers):
            buf_len = self.buf_lengths[i]

            # Concat ctx[i] and x
            # ctx[i] is [B, 1, buf_len, C]
            # x is [B, 1, L, C]

            conv_in = ttnn.concat([ctx[i], x], dim=2)

            # Update ctx for next time
            # tail of conv_in
            # [B, 1, L+buf_len, C] -> slice last buf_len

            ctx_update = ttnn.slice(
                conv_in,
                (0, 0, conv_in.shape[2] - buf_len, 0),
                (conv_in.shape[0], 1, conv_in.shape[2], conv_in.shape[3])
            )
            new_ctx.append(ctx_update)

            # Run layer
            layer_out = layer(conv_in)

            if self.combine_residuals == 'add':
                x = ttnn.add(x, layer_out)
            elif self.combine_residuals == 'multiply':
                x = ttnn.mul(x, layer_out)
            else:
                x = layer_out

        return x, new_ctx
