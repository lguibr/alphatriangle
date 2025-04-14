from pydantic import BaseModel, Field, computed_field, field_validator, model_validator


class EnvConfig(BaseModel):
    """Configuration for the game environment (Pydantic model)."""

    ROWS: int = Field(8, gt=0)
    COLS_PER_ROW: list[int] = Field(default=[9, 11, 13, 15, 15, 13, 11, 9])
    COLS: int = Field(default=15, gt=0)
    NUM_SHAPE_SLOTS: int = Field(3, gt=0)
    MIN_LINE_LENGTH: int = Field(3, gt=0)

    @field_validator("COLS_PER_ROW")
    @classmethod
    def check_cols_per_row_length(cls, v: list[int], info) -> list[int]:
        rows = info.data.get("ROWS")
        if rows is None:
            # This might happen during initial validation before ROWS is processed
            # Pydantic v2 should handle dependencies better, but let's be safe
            return v
        if len(v) != rows:
            raise ValueError(f"COLS_PER_ROW length ({len(v)}) must equal ROWS ({rows})")
        if any(width <= 0 for width in v):
            raise ValueError("All values in COLS_PER_ROW must be positive.")
        return v

    @model_validator(mode="after")
    def check_cols_match_max_cols_per_row(self) -> "EnvConfig":
        """Ensure COLS is at least the maximum width required by any row."""
        max_row_width = max(self.COLS_PER_ROW, default=0)
        if max_row_width > self.COLS:
            raise ValueError(
                f"COLS ({self.COLS}) must be >= the maximum value in COLS_PER_ROW ({max_row_width})"
            )
        return self

    @computed_field
    def ACTION_DIM(self) -> int:
        """Total number of possible actions (shape_slot * row * col)."""
        return self.NUM_SHAPE_SLOTS * self.ROWS * self.COLS
