# File: alphatriangle/config/env_config.py
# No changes needed for this refactoring step. Keep the existing content.
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator


class EnvConfig(BaseModel):
    """Configuration for the game environment (Pydantic model)."""

    ROWS: int = Field(default=8, gt=0)
    # Provide a default that matches the default ROWS
    COLS_PER_ROW: list[int] = Field(default=[9, 11, 13, 15, 15, 13, 11, 9])
    COLS: int = Field(default=15, gt=0)
    NUM_SHAPE_SLOTS: int = Field(default=3, gt=0)
    MIN_LINE_LENGTH: int = Field(default=3, gt=0)

    # --- Reward System Constants (v3) ---
    REWARD_PER_PLACED_TRIANGLE: float = Field(default=0.01)
    REWARD_PER_CLEARED_TRIANGLE: float = Field(default=0.5)
    REWARD_PER_STEP_ALIVE: float = Field(default=0.005)
    PENALTY_GAME_OVER: float = Field(default=-10.0)
    # --- End Reward System Constants ---

    @field_validator("COLS_PER_ROW")
    @classmethod
    def check_cols_per_row_length(cls, v: list[int], info) -> list[int]:
        data = info.data if info.data else info.values
        rows = data.get("ROWS")
        if rows is None:
            return v
        if len(v) != rows:
            raise ValueError(f"COLS_PER_ROW length ({len(v)}) must equal ROWS ({rows})")
        if any(width <= 0 for width in v):
            raise ValueError("All values in COLS_PER_ROW must be positive.")
        return v

    @model_validator(mode="after")
    def check_cols_match_max_cols_per_row(self) -> "EnvConfig":
        """Ensure COLS is at least the maximum width required by any row."""
        if hasattr(self, "COLS_PER_ROW") and self.COLS_PER_ROW:
            max_row_width = max(self.COLS_PER_ROW, default=0)
            if max_row_width > self.COLS:
                raise ValueError(
                    f"COLS ({self.COLS}) must be >= the maximum value in COLS_PER_ROW ({max_row_width})"
                )
        elif not hasattr(self, "COLS_PER_ROW"):
            pass
        return self

    @computed_field  # type: ignore[misc] # Decorator requires Pydantic v2
    @property
    def ACTION_DIM(self) -> int:
        """Total number of possible actions (shape_slot * row * col)."""
        if (
            hasattr(self, "NUM_SHAPE_SLOTS")
            and hasattr(self, "ROWS")
            and hasattr(self, "COLS")
        ):
            return self.NUM_SHAPE_SLOTS * self.ROWS * self.COLS
        return 0


EnvConfig.model_rebuild(force=True)
