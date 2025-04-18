# File: alphatriangle/config/model_config.py
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    """
    Configuration for the Neural Network model (Pydantic model).
    --- TUNED FOR SMALLER CAPACITY (~3M Params Target, Laptop Feasible) ---
    """

    # Input channels for the grid (e.g., 1 for occupancy, more for history/colors)
    GRID_INPUT_CHANNELS: int = Field(default=1, gt=0)

    # --- CNN Architecture Parameters ---
    CONV_FILTERS: list[int] = Field(default=[32, 64, 128])  # Smaller CNN
    CONV_KERNEL_SIZES: list[int | tuple[int, int]] = Field(default=[3, 3, 3])
    CONV_STRIDES: list[int | tuple[int, int]] = Field(default=[1, 1, 1])
    CONV_PADDING: list[int | tuple[int, int] | str] = Field(default=[1, 1, 1])

    # --- Residual Block Parameters ---
    NUM_RESIDUAL_BLOCKS: int = Field(default=2, ge=0)  # Fewer blocks
    RESIDUAL_BLOCK_FILTERS: int = Field(default=128, gt=0)  # Match last conv filter

    # --- Transformer Parameters (Optional) ---
    USE_TRANSFORMER: bool = Field(default=True)  # Keep Transformer enabled
    TRANSFORMER_DIM: int = Field(default=128, gt=0)  # Match Res block filters
    TRANSFORMER_HEADS: int = Field(default=4, gt=0)  # Moderate heads
    TRANSFORMER_LAYERS: int = Field(default=2, ge=0)  # Fewer layers
    TRANSFORMER_FC_DIM: int = Field(default=256, gt=0)  # Moderate feedforward dim

    # --- Fully Connected Layers ---
    FC_DIMS_SHARED: list[int] = Field(default=[128])  # Single shared layer

    # --- Policy Head ---
    POLICY_HEAD_DIMS: list[int] = Field(default=[128])  # Single policy layer

    # --- Distributional Value Head Parameters ---
    NUM_VALUE_ATOMS: int = Field(default=51, gt=1)  # Standard C51 atoms
    VALUE_MIN: float = Field(default=-10.0)  # Reasonable expected value range
    VALUE_MAX: float = Field(default=10.0)  # Reasonable expected value range

    # --- Value Head Dims ---
    VALUE_HEAD_DIMS: list[int] = Field(default=[128])  # Single value layer

    # --- Other Hyperparameters ---
    ACTIVATION_FUNCTION: Literal["ReLU", "GELU", "SiLU", "Tanh", "Sigmoid"] = Field(
        default="ReLU"
    )
    USE_BATCH_NORM: bool = Field(default=True)

    # --- Input Feature Dimension ---
    # Dimension of the non-grid feature vector concatenated after CNN/Transformer.
    # Must match the output of features.extractor.get_combined_other_features.
    OTHER_NN_INPUT_FEATURES_DIM: int = Field(default=30, gt=0)  # Keep default

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

    @model_validator(mode="after")
    def check_value_distribution_params(self) -> "ModelConfig":
        if (
            hasattr(self, "VALUE_MIN")
            and hasattr(self, "VALUE_MAX")
            and self.VALUE_MIN >= self.VALUE_MAX
        ):
            raise ValueError("VALUE_MIN must be strictly less than VALUE_MAX.")
        return self


ModelConfig.model_rebuild(force=True)
