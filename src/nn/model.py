# File: src/nn/model.py
# File: src/nn/model.py
import math

import torch
import torch.nn as nn

# Use relative imports
from ..config import EnvConfig, ModelConfig


def conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    padding: int | tuple[int, int] | str,
    use_batch_norm: bool,
    activation: type[nn.Module],
) -> nn.Sequential:
    """Creates a standard convolutional block."""
    layers: list[nn.Module] = [  # Explicitly type the list
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=not use_batch_norm,
        )
    ]
    if use_batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(activation())
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    """Standard Residual Block."""

    def __init__(
        self, channels: int, use_batch_norm: bool, activation: type[nn.Module]
    ):
        super().__init__()
        self.conv1 = conv_block(channels, channels, 3, 1, 1, use_batch_norm, activation)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=not use_batch_norm)
        self.bn2 = nn.BatchNorm2d(channels) if use_batch_norm else nn.Identity()
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # Added return type hint
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.activation(out)
        return out


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        # Ensure the second slice doesn't go out of bounds if d_model is odd
        # Corrected slicing for odd d_model
        if d_model % 2 != 0:
            pe[:, 0, 1::2] = torch.cos(
                position * div_term[:-1]
            )  # Use div_term[:-1] for odd
        else:
            pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return x


class AlphaTriangleNet(nn.Module):
    """
    Neural Network architecture for AlphaTriangle.
    Includes optional Transformer Encoder block after CNN body.
    """

    def __init__(self, model_config: ModelConfig, env_config: EnvConfig):
        super().__init__()
        self.model_config = model_config
        self.env_config = env_config
        # Cast ACTION_DIM to int
        self.action_dim = int(env_config.ACTION_DIM)

        activation_cls: type[nn.Module] = getattr(nn, model_config.ACTIVATION_FUNCTION)

        # --- CNN Body ---
        conv_layers: list[nn.Module] = []  # Explicitly type the list
        in_channels = model_config.GRID_INPUT_CHANNELS
        for i, out_channels in enumerate(model_config.CONV_FILTERS):
            conv_layers.append(
                conv_block(
                    in_channels,
                    out_channels,
                    model_config.CONV_KERNEL_SIZES[i],
                    model_config.CONV_STRIDES[i],
                    model_config.CONV_PADDING[i],
                    model_config.USE_BATCH_NORM,
                    activation_cls,
                )
            )
            in_channels = out_channels
        self.conv_body = nn.Sequential(*conv_layers)

        # --- Residual Body ---
        res_layers: list[nn.Module] = []  # Explicitly type the list
        if model_config.NUM_RESIDUAL_BLOCKS > 0:
            res_channels = model_config.RESIDUAL_BLOCK_FILTERS
            if in_channels != res_channels:
                # Add projection layer if channels don't match
                res_layers.append(
                    conv_block(
                        in_channels,
                        res_channels,
                        1,
                        1,
                        0,
                        model_config.USE_BATCH_NORM,
                        activation_cls,
                    )
                )
                in_channels = res_channels
            for _ in range(model_config.NUM_RESIDUAL_BLOCKS):
                res_layers.append(
                    ResidualBlock(
                        in_channels, model_config.USE_BATCH_NORM, activation_cls
                    )
                )
        self.res_body = nn.Sequential(*res_layers)
        self.cnn_output_channels = in_channels  # Channels after CNN/Res blocks

        # --- Transformer Body (Optional) ---
        self.transformer_body = None
        self.pos_encoder = None
        self.input_proj: nn.Module = nn.Identity()
        self.transformer_output_size = 0

        if model_config.USE_TRANSFORMER and model_config.TRANSFORMER_LAYERS > 0:
            transformer_input_dim = model_config.TRANSFORMER_DIM
            if self.cnn_output_channels != transformer_input_dim:
                self.input_proj = nn.Conv2d(
                    self.cnn_output_channels, transformer_input_dim, kernel_size=1
                )
            else:
                self.input_proj = nn.Identity()

            self.pos_encoder = PositionalEncoding(transformer_input_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=transformer_input_dim,
                nhead=model_config.TRANSFORMER_HEADS,
                dim_feedforward=model_config.TRANSFORMER_FC_DIM,
                activation=model_config.ACTIVATION_FUNCTION.lower(),
                batch_first=False,  # Expects (Seq, Batch, Dim)
                norm_first=True,
            )
            transformer_norm = nn.LayerNorm(transformer_input_dim)
            self.transformer_body = nn.TransformerEncoder(
                encoder_layer,
                num_layers=model_config.TRANSFORMER_LAYERS,
                norm=transformer_norm,
            )

            # Calculate transformer output size using a dummy forward pass
            dummy_input_grid = torch.zeros(
                1, model_config.GRID_INPUT_CHANNELS, env_config.ROWS, env_config.COLS
            )
            with torch.no_grad():
                cnn_out = self.conv_body(dummy_input_grid)
                res_out = self.res_body(cnn_out)
                proj_out = self.input_proj(res_out)
                b, d, h, w = proj_out.shape
                # Reshape for transformer: (Seq, Batch, Dim) -> (H*W, B, D)
                # transformer_input_dummy = proj_out.flatten(2).permute(2, 0, 1) # Removed unused variable
                # Positional encoding added in forward pass
                # Transformer output shape is (Seq, Batch, Dim)
                self.transformer_output_size = h * w * d  # Dim is d here
        else:
            # Calculate flattened size after conv/res blocks if no transformer
            dummy_input_grid = torch.zeros(
                1, model_config.GRID_INPUT_CHANNELS, env_config.ROWS, env_config.COLS
            )
            with torch.no_grad():
                conv_output = self.conv_body(dummy_input_grid)
                res_output = self.res_body(conv_output)
                self.flattened_cnn_size = res_output.numel()

        # --- Shared Fully Connected Layers ---
        if model_config.USE_TRANSFORMER and model_config.TRANSFORMER_LAYERS > 0:
            combined_input_size = (
                self.transformer_output_size + model_config.OTHER_NN_INPUT_FEATURES_DIM
            )
        else:
            combined_input_size = (
                self.flattened_cnn_size + model_config.OTHER_NN_INPUT_FEATURES_DIM
            )

        shared_fc_layers: list[nn.Module] = []  # Explicitly type the list
        in_features = combined_input_size
        for hidden_dim in model_config.FC_DIMS_SHARED:
            shared_fc_layers.append(nn.Linear(in_features, hidden_dim))
            if model_config.USE_BATCH_NORM:
                # Use BatchNorm1d for FC layers
                shared_fc_layers.append(nn.BatchNorm1d(hidden_dim))
            shared_fc_layers.append(activation_cls())
            in_features = hidden_dim
        self.shared_fc = nn.Sequential(*shared_fc_layers)

        # --- Policy Head ---
        policy_head_layers: list[nn.Module] = []  # Explicitly type the list
        policy_in_features = in_features
        # Iterate through hidden dims if any
        for hidden_dim in model_config.POLICY_HEAD_DIMS:
            policy_head_layers.append(nn.Linear(policy_in_features, hidden_dim))
            if model_config.USE_BATCH_NORM:
                policy_head_layers.append(nn.BatchNorm1d(hidden_dim))
            policy_head_layers.append(activation_cls())
            policy_in_features = hidden_dim
        # Final layer to output action dimension logits
        # Use self.action_dim which is already cast to int
        policy_head_layers.append(nn.Linear(policy_in_features, self.action_dim))
        self.policy_head = nn.Sequential(*policy_head_layers)

        # --- Value Head ---
        value_head_layers: list[nn.Module] = []  # Explicitly type the list
        value_in_features = in_features
        # Iterate through hidden dims, excluding the final output dim (1)
        for hidden_dim in model_config.VALUE_HEAD_DIMS[:-1]:
            value_head_layers.append(nn.Linear(value_in_features, hidden_dim))
            if model_config.USE_BATCH_NORM:
                value_head_layers.append(nn.BatchNorm1d(hidden_dim))
            value_head_layers.append(activation_cls())
            value_in_features = hidden_dim
        # Final layer to output the single value
        value_head_layers.append(
            nn.Linear(value_in_features, model_config.VALUE_HEAD_DIMS[-1])
        )
        value_head_layers.append(nn.Tanh())  # Apply Tanh activation
        self.value_head = nn.Sequential(*value_head_layers)

    def forward(
        self, grid_state: torch.Tensor, other_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        Returns: (policy_logits, value)
        """
        conv_out = self.conv_body(grid_state)
        res_out = self.res_body(conv_out)

        # Optional Transformer Body
        if (
            self.model_config.USE_TRANSFORMER
            and self.transformer_body is not None  # Check if transformer exists
            and self.pos_encoder is not None
        ):
            proj_out = self.input_proj(res_out)  # Shape: (B, D, H, W)
            b, d, h, w = proj_out.shape
            # Reshape for transformer: (Seq, Batch, Dim) -> (H*W, B, D)
            transformer_input = proj_out.flatten(2).permute(2, 0, 1)
            # Add positional encoding
            transformer_input = self.pos_encoder(transformer_input)
            # Pass through transformer encoder
            transformer_output = self.transformer_body(
                transformer_input
            )  # Shape: (Seq, Batch, Dim)
            # Flatten transformer output: (Seq, Batch, Dim) -> (Batch, Seq*Dim)
            flattened_features = transformer_output.permute(1, 0, 2).flatten(1)
        else:
            # Flatten CNN output if no transformer
            flattened_features = res_out.view(res_out.size(0), -1)

        # Combine with other features
        combined_features = torch.cat([flattened_features, other_features], dim=1)

        # Shared FC Layers and Heads
        shared_out = self.shared_fc(combined_features)
        policy_logits = self.policy_head(shared_out)
        value = self.value_head(shared_out)

        return policy_logits, value
