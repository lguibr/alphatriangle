# File: src/config/env_config.py
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator


class EnvConfig(BaseModel):
    """Configuration for the game environment (Pydantic model)."""

    ROWS: int = Field(default=8, gt=0)
    # Provide a default that matches the default ROWS
    COLS_PER_ROW: list[int] = Field(default=[9, 11, 13, 15, 15, 13, 11, 9])
    COLS: int = Field(default=15, gt=0)
    NUM_SHAPE_SLOTS: int = Field(default=3, gt=0)
    MIN_LINE_LENGTH: int = Field(default=3, gt=0)

    @field_validator("COLS_PER_ROW")
    @classmethod
    def check_cols_per_row_length(cls, v: list[int], info) -> list[int]:
        # info.data might not be available during initial validation in Pydantic v2
        # Check 'values' if info.data is empty
        data = info.data if info.data else info.values
        rows = data.get("ROWS")
        if rows is None:
            # If ROWS isn't available yet, we can't validate length robustly.
            # Pydantic v2's model_validator will catch this later if needed.
            # However, since ROWS has a default, this case is less likely.
            return v
        if len(v) != rows:
            raise ValueError(f"COLS_PER_ROW length ({len(v)}) must equal ROWS ({rows})")
        if any(width <= 0 for width in v):
            raise ValueError("All values in COLS_PER_ROW must be positive.")
        return v

    @model_validator(mode="after")
    def check_cols_match_max_cols_per_row(self) -> "EnvConfig":
        """Ensure COLS is at least the maximum width required by any row."""
        # Check if COLS_PER_ROW exists before accessing it
        if hasattr(self, "COLS_PER_ROW") and self.COLS_PER_ROW:
            max_row_width = max(self.COLS_PER_ROW, default=0)
            if max_row_width > self.COLS:
                raise ValueError(
                    f"COLS ({self.COLS}) must be >= the maximum value in COLS_PER_ROW ({max_row_width})"
                )
        elif not hasattr(self, "COLS_PER_ROW"):
            # Handle case where validation runs before COLS_PER_ROW is set (shouldn't happen with defaults)
            pass
        return self

    @computed_field  # type: ignore[misc] # Decorator requires Pydantic v2
    @property
    def ACTION_DIM(self) -> int:
        """Total number of possible actions (shape_slot * row * col)."""
        # Ensure attributes exist before calculating
        if (
            hasattr(self, "NUM_SHAPE_SLOTS")
            and hasattr(self, "ROWS")
            and hasattr(self, "COLS")
        ):
            return self.NUM_SHAPE_SLOTS * self.ROWS * self.COLS
        # Provide a default or raise an error if attributes aren't ready
        # Returning 0 might be safer than raising during Pydantic initialization.
        return 0


# Ensure model is rebuilt after changes
EnvConfig.model_rebuild(force=True)
