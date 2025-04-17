# File: alphatriangle/nn/model.py
import math
from typing import cast

import torch
import torch.nn as nn

from ..config import EnvConfig, ModelConfig

# --- REMOVED: Incorrect self-import ---
# from .model import AlphaTriangleNet
# --- END REMOVED ---


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
    layers: list[nn.Module] = [
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out: torch.Tensor = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.activation(out)
        return out


class PositionalEncoding(nn.Module):
    """Injects sinusoidal positional encoding. (Adapted from PyTorch tutorial)"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive for PositionalEncoding")
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  # Shape: [max_len, 1]
        # --- CHANGE: Simplified calculation based on tutorial ---
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )  # Shape: [d_model / 2]
        pe = torch.zeros(max_len, d_model)  # Shape: [max_len, d_model]

        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cos to odd indices (if they exist)
        # Note: div_term is already the correct size for broadcasting with pe[:, 1::2]
        # because its length is ceil(d_model / 2). If d_model is odd,
        # the last element of div_term won't be used for the cos calculation anyway.
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term)

        # Add the batch dimension (1) expected by register_buffer and forward pass
        # Shape becomes [max_len, 1, d_model]
        pe = pe.unsqueeze(1)
        # --- END CHANGE ---

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
                (Note: AlphaTriangleNet might pass [batch_size, embedding_dim, seq_len (H*W)])
                It needs to be permuted before applying positional encoding if that's the case.
                Here, we assume the input is already [seq_len, batch_size, embedding_dim].

        Returns:
            Tensor with added positional encoding.
        """
        pe_buffer = self.pe
        if not isinstance(pe_buffer, torch.Tensor):
            raise TypeError("PositionalEncoding buffer 'pe' is not a Tensor.")

        if x.shape[0] > pe_buffer.shape[0]:
            raise ValueError(
                f"Input sequence length {x.shape[0]} exceeds max_len {pe_buffer.shape[0]} of PositionalEncoding"
            )
        if x.shape[2] != pe_buffer.shape[2]:
            raise ValueError(
                f"Input embedding dimension {x.shape[2]} does not match PositionalEncoding dimension {pe_buffer.shape[2]}"
            )

        # Add positional encoding
        # Slicing pe_buffer[:x.size(0)] handles variable sequence lengths
        x = x + pe_buffer[: x.size(0)]
        return cast("torch.Tensor", self.dropout(x))


class AlphaTriangleNet(nn.Module):
    """
    Neural Network architecture for AlphaTriangle.
    Includes optional Transformer Encoder block after CNN body.
    Supports Distributional Value Head (C51).
    """

    def __init__(self, model_config: ModelConfig, env_config: EnvConfig):
        super().__init__()
        self.model_config = model_config
        self.env_config = env_config
        # Cast ACTION_DIM to int
        self.action_dim = int(env_config.ACTION_DIM)  # type: ignore[call-overload]

        activation_cls: type[nn.Module] = getattr(nn, model_config.ACTIVATION_FUNCTION)

        # --- CNN Body ---
        conv_layers: list[nn.Module] = []
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
        res_layers: list[nn.Module] = []
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

            self.pos_encoder = PositionalEncoding(transformer_input_dim, dropout=0.1)
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
                # Size after flattening H*W dimensions
                self.transformer_output_size = h * w * d
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
        policy_head_layers: list[nn.Module] = []
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

        # --- Value Head (Distributional) --- CHANGED
        value_head_layers: list[nn.Module] = []
        value_in_features = in_features
        # Iterate through hidden dims if any
        for hidden_dim in model_config.VALUE_HEAD_DIMS:
            value_head_layers.append(nn.Linear(value_in_features, hidden_dim))
            if model_config.USE_BATCH_NORM:
                value_head_layers.append(nn.BatchNorm1d(hidden_dim))
            value_head_layers.append(activation_cls())
            value_in_features = hidden_dim
        # Final layer to output logits for each value atom
        value_head_layers.append(
            nn.Linear(value_in_features, model_config.NUM_VALUE_ATOMS)
        )
        # REMOVED: Tanh activation - we need logits for cross-entropy loss
        # value_head_layers.append(nn.Tanh())
        self.value_head = nn.Sequential(*value_head_layers)
        # --- END CHANGED ---

    def forward(
        self, grid_state: torch.Tensor, other_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        Returns: (policy_logits, value_distribution_logits)
        """
        conv_out = self.conv_body(grid_state)
        res_out = self.res_body(conv_out)

        # Optional Transformer Body
        if (
            self.model_config.USE_TRANSFORMER
            and self.transformer_body is not None
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
        # --- CHANGED: Return value logits ---
        value_logits = self.value_head(shared_out)
        return policy_logits, value_logits
        # --- END CHANGED ---
