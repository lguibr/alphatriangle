import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from src.config import ModelConfig, EnvConfig
import math


def conv_block(
    in_channels, out_channels, kernel_size, stride, padding, use_batch_norm, activation
):
    layers = [
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
    def __init__(self, channels, use_batch_norm, activation):
        super().__init__()
        self.conv1 = conv_block(channels, channels, 3, 1, 1, use_batch_norm, activation)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=not use_batch_norm)
        self.bn2 = nn.BatchNorm2d(channels) if use_batch_norm else nn.Identity()
        self.activation = activation()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.activation(out)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
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
        self.action_dim = env_config.ACTION_DIM

        activation = getattr(nn, model_config.ACTIVATION_FUNCTION)

        conv_layers = []
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
                    activation,
                )
            )
            in_channels = out_channels
        self.conv_body = nn.Sequential(*conv_layers)

        res_layers = []
        if model_config.NUM_RESIDUAL_BLOCKS > 0:
            res_channels = model_config.RESIDUAL_BLOCK_FILTERS
            if in_channels != res_channels:
                res_layers.append(
                    conv_block(
                        in_channels,
                        res_channels,
                        1,
                        1,
                        0,
                        model_config.USE_BATCH_NORM,
                        activation,
                    )
                )
                in_channels = res_channels
            for _ in range(model_config.NUM_RESIDUAL_BLOCKS):
                res_layers.append(
                    ResidualBlock(in_channels, model_config.USE_BATCH_NORM, activation)
                )
        self.res_body = nn.Sequential(*res_layers)

        self.transformer_body = None
        self.pos_encoder = None
        self.transformer_output_size = 0
        self.cnn_output_channels = in_channels

        if model_config.USE_TRANSFORMER:
            if self.cnn_output_channels != model_config.TRANSFORMER_DIM:
                self.input_proj = nn.Conv2d(
                    self.cnn_output_channels,
                    model_config.TRANSFORMER_DIM,
                    kernel_size=1,
                )
                self.transformer_input_dim = model_config.TRANSFORMER_DIM
            else:
                self.input_proj = nn.Identity()
                self.transformer_input_dim = self.cnn_output_channels

            self.pos_encoder = PositionalEncoding(self.transformer_input_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.transformer_input_dim,
                nhead=model_config.TRANSFORMER_HEADS,
                dim_feedforward=model_config.TRANSFORMER_FC_DIM,
                activation=model_config.ACTIVATION_FUNCTION.lower(),
                batch_first=False,
                norm_first=True,
            )
            transformer_norm = nn.LayerNorm(self.transformer_input_dim)
            self.transformer_body = nn.TransformerEncoder(
                encoder_layer,
                num_layers=model_config.TRANSFORMER_LAYERS,
                norm=transformer_norm,
            )
            dummy_input_grid = torch.zeros(
                1, model_config.GRID_INPUT_CHANNELS, env_config.ROWS, env_config.COLS
            )
            with torch.no_grad():
                cnn_out = self.conv_body(dummy_input_grid)
                res_out = self.res_body(cnn_out)  # B, C, H, W
                proj_out = self.input_proj(res_out)  # B, D, H, W
                b, d, h, w = proj_out.shape
                # Reshape for transformer: (Seq, Batch, Dim) -> (H*W, B, D)
                transformer_input = proj_out.flatten(2).permute(
                    2, 0, 1
                )  # Seq=H*W, Batch=B, Dim=D
                # Positional encoding is added inside forward pass
                # Transformer output shape is (Seq, Batch, Dim)
                self.transformer_output_size = h * w * self.transformer_input_dim
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
        if model_config.USE_TRANSFORMER:
            combined_input_size = (
                self.transformer_output_size + model_config.OTHER_NN_INPUT_FEATURES_DIM
            )
        else:
            combined_input_size = (
                self.flattened_cnn_size + model_config.OTHER_NN_INPUT_FEATURES_DIM
            )

        shared_fc_layers = []
        in_features = combined_input_size
        for hidden_dim in model_config.FC_DIMS_SHARED:
            shared_fc_layers.append(nn.Linear(in_features, hidden_dim))
            if model_config.USE_BATCH_NORM:
                # Use LayerNorm if coming from Transformer, BatchNorm otherwise?
                # Let's stick to BatchNorm for consistency with original design for now.
                shared_fc_layers.append(nn.BatchNorm1d(hidden_dim))
            shared_fc_layers.append(activation())
            in_features = hidden_dim
        self.shared_fc = nn.Sequential(*shared_fc_layers)

        # --- Policy Head ---
        policy_head_layers = []
        policy_in_features = in_features
        for hidden_dim in model_config.POLICY_HEAD_DIMS:
            policy_head_layers.append(nn.Linear(policy_in_features, hidden_dim))
            if model_config.USE_BATCH_NORM:
                policy_head_layers.append(nn.BatchNorm1d(hidden_dim))
            policy_head_layers.append(activation())
            policy_in_features = hidden_dim
        policy_head_layers.append(nn.Linear(policy_in_features, self.action_dim))
        self.policy_head = nn.Sequential(*policy_head_layers)

        # --- Value Head ---
        value_head_layers = []
        value_in_features = in_features
        for hidden_dim in model_config.VALUE_HEAD_DIMS[:-1]:
            value_head_layers.append(nn.Linear(value_in_features, hidden_dim))
            if model_config.USE_BATCH_NORM:
                value_head_layers.append(nn.BatchNorm1d(hidden_dim))
            value_head_layers.append(activation())
            value_in_features = hidden_dim
        value_head_layers.append(
            nn.Linear(value_in_features, model_config.VALUE_HEAD_DIMS[-1])
        )
        value_head_layers.append(nn.Tanh())
        self.value_head = nn.Sequential(*value_head_layers)

    def forward(
        self, grid_state: torch.Tensor, other_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        Returns: (policy_logits, value)
        """
        conv_out = self.conv_body(grid_state)
        res_out = self.res_body(conv_out)

        # Optional Transformer Body
        if (
            self.model_config.USE_TRANSFORMER
            and self.transformer_body
            and self.pos_encoder
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
