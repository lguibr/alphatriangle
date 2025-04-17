# File: alphatriangle/config/model_config.py
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    """
    Configuration for the Neural Network model (Pydantic model).
    --- Further Enhanced Capacity Configuration ---
    --- Includes Distributional RL (C51) Parameters ---
    """

    GRID_INPUT_CHANNELS: int = Field(default=1, gt=0)
    # --- CNN Architecture Parameters ---
    # Increased depth and width again
    CONV_FILTERS: list[int] = Field(default=[128, 256, 512])  # CHANGED
    CONV_KERNEL_SIZES: list[int | tuple[int, int]] = Field(default=[3, 3, 3])
    CONV_STRIDES: list[int | tuple[int, int]] = Field(default=[1, 1, 1])
    CONV_PADDING: list[int | tuple[int, int] | str] = Field(default=[1, 1, 1])

    # Increased residual blocks and match filter size
    NUM_RESIDUAL_BLOCKS: int = Field(default=8, ge=0)  # CHANGED
    RESIDUAL_BLOCK_FILTERS: int = Field(
        default=512, gt=0
    )  # CHANGED (Match last conv filter)

    # --- Transformer Parameters (Further Enhanced) ---
    USE_TRANSFORMER: bool = Field(default=True)
    # Increased dimensions for Transformer capacity, match ResNet output
    TRANSFORMER_DIM: int = Field(default=512, gt=0)  # CHANGED
    TRANSFORMER_HEADS: int = Field(
        default=16, gt=0
    )  # CHANGED (Needs to divide TRANSFORMER_DIM: 512/16=32)
    TRANSFORMER_LAYERS: int = Field(default=6, ge=0)  # CHANGED
    TRANSFORMER_FC_DIM: int = Field(
        default=1024, gt=0
    )  # CHANGED (Increased feedforward dim)

    # --- Fully Connected Layers (Further Enhanced) ---
    # Increased dimensions for shared and head layers
    FC_DIMS_SHARED: list[int] = Field(default=[1024])  # CHANGED
    POLICY_HEAD_DIMS: list[int] = Field(
        default=[1024]
    )  # CHANGED (Output dim added automatically)

    # --- Distributional Value Head Parameters --- ADDED
    NUM_VALUE_ATOMS: int = Field(default=51, gt=1)  # Number of atoms for C51
    VALUE_MIN: float = Field(default=-10.0)  # Minimum value support
    VALUE_MAX: float = Field(default=10.0)  # Maximum value support

    # --- Value Head Dims (Now outputs NUM_VALUE_ATOMS) --- CHANGED
    VALUE_HEAD_DIMS: list[int] = Field(
        default=[1024]
    )  # Final output dim (NUM_VALUE_ATOMS) added automatically

    # --- Other Hyperparameters ---
    ACTIVATION_FUNCTION: Literal["ReLU", "GELU", "SiLU", "Tanh", "Sigmoid"] = Field(
        default="ReLU"  # GELU or SiLU might also work well with Transformers
    )
    USE_BATCH_NORM: bool = Field(default=True)

    # --- Input Feature Dimension ---
    # This depends on alphatriangle/features/extractor.py and should match its output.
    # Default calculation: 3 slots * 7 shape feats + 3 avail feats + 6 explicit feats = 30
    # This calculation has NOT changed with the model architecture changes.
    OTHER_NN_INPUT_FEATURES_DIM: int = Field(default=30, gt=0)  # UNCHANGED

    @model_validator(mode="after")
    def check_conv_layers_consistency(self) -> "ModelConfig":
        # Ensure attributes exist before checking lengths
        if (
            hasattr(self, "CONV_FILTERS")
            and hasattr(self, "CONV_KERNEL_SIZES")
            and hasattr(self, "CONV_STRIDES")
            and hasattr(self, "CONV_PADDING")
        ):
            n_filters = len(self.CONV_FILTERS)
            if not (
                len(self.CONV_KERNEL_SIZES) == n_filters
                and len(self.CONV_STRIDES) == n_filters
                and len(self.CONV_PADDING) == n_filters
            ):
                raise ValueError(
                    "Lengths of CONV_FILTERS, CONV_KERNEL_SIZES, CONV_STRIDES, and CONV_PADDING must match."
                )
        return self

    @model_validator(mode="after")
    def check_residual_filter_match(self) -> "ModelConfig":
        # Ensure attributes exist before checking
        if (
            hasattr(self, "NUM_RESIDUAL_BLOCKS")
            and self.NUM_RESIDUAL_BLOCKS > 0
            and hasattr(self, "CONV_FILTERS")
            and self.CONV_FILTERS
            and hasattr(self, "RESIDUAL_BLOCK_FILTERS")
            and self.CONV_FILTERS[-1] != self.RESIDUAL_BLOCK_FILTERS
        ):
            # This warning is now handled by the projection layer in the model if needed
            pass  # Model handles projection if needed
        return self

    @model_validator(mode="after")
    def check_transformer_config(self) -> "ModelConfig":
        # Ensure attributes exist before checking
        if hasattr(self, "USE_TRANSFORMER") and self.USE_TRANSFORMER:
            if not hasattr(self, "TRANSFORMER_LAYERS") or self.TRANSFORMER_LAYERS < 0:
                raise ValueError("TRANSFORMER_LAYERS cannot be negative.")
            if self.TRANSFORMER_LAYERS > 0:
                if not hasattr(self, "TRANSFORMER_DIM") or self.TRANSFORMER_DIM <= 0:
                    raise ValueError(
                        "TRANSFORMER_DIM must be positive if TRANSFORMER_LAYERS > 0."
                    )
                if (
                    not hasattr(self, "TRANSFORMER_HEADS")
                    or self.TRANSFORMER_HEADS <= 0
                ):
                    raise ValueError(
                        "TRANSFORMER_HEADS must be positive if TRANSFORMER_LAYERS > 0."
                    )
                if self.TRANSFORMER_DIM % self.TRANSFORMER_HEADS != 0:
                    raise ValueError(
                        f"TRANSFORMER_DIM ({self.TRANSFORMER_DIM}) must be divisible by TRANSFORMER_HEADS ({self.TRANSFORMER_HEADS})."
                    )
                if (
                    not hasattr(self, "TRANSFORMER_FC_DIM")
                    or self.TRANSFORMER_FC_DIM <= 0
                ):
                    raise ValueError(
                        "TRANSFORMER_FC_DIM must be positive if TRANSFORMER_LAYERS > 0."
                    )
        return self

    @model_validator(mode="after")
    def check_transformer_dim_consistency(self) -> "ModelConfig":
        # Ensure attributes exist before checking
        if (
            hasattr(self, "USE_TRANSFORMER")
            and self.USE_TRANSFORMER
            and hasattr(self, "TRANSFORMER_LAYERS")
            and self.TRANSFORMER_LAYERS > 0
            and hasattr(self, "CONV_FILTERS")
            and self.CONV_FILTERS
            and hasattr(self, "TRANSFORMER_DIM")
        ):
            cnn_output_channels = (
                self.RESIDUAL_BLOCK_FILTERS
                if hasattr(self, "NUM_RESIDUAL_BLOCKS") and self.NUM_RESIDUAL_BLOCKS > 0
                else self.CONV_FILTERS[-1]
            )
            if cnn_output_channels != self.TRANSFORMER_DIM:
                # This is handled by an input projection layer in the model now
                pass  # Model handles projection
        return self

    # --- ADDED: Validation for distributional parameters ---
    @model_validator(mode="after")
    def check_value_distribution_params(self) -> "ModelConfig":
        # --- CHANGED: Combined nested if ---
        if (
            hasattr(self, "VALUE_MIN")
            and hasattr(self, "VALUE_MAX")
            and self.VALUE_MIN >= self.VALUE_MAX
        ):
            raise ValueError("VALUE_MIN must be strictly less than VALUE_MAX.")
        # --- END CHANGED ---
        return self


ModelConfig.model_rebuild(force=True)
