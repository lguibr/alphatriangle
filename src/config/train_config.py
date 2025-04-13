# File: src/config/train_config.py
import time
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class TrainConfig(BaseModel):
    """
    Configuration for the training process (Pydantic model).
    --- SERIOUS CONFIGURATION ---
    """

    RUN_NAME: str = Field(
        # More descriptive default run name
        default_factory=lambda: f"train_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    LOAD_CHECKPOINT_PATH: str | None = Field(
        None
    )  # Explicit path overrides auto-resume
    LOAD_BUFFER_PATH: str | None = Field(None)  # Explicit path overrides auto-resume
    AUTO_RESUME_LATEST: bool = Field(
        True
    )  # Resume from latest previous run if no explicit path
    DEVICE: Literal["auto", "cuda", "cpu", "mps"] = Field(
        "auto"
    )  # 'auto' is recommended
    RANDOM_SEED: int = Field(42)

    # --- Training Loop ---
    # Increased steps for longer training (e.g., overnight)
    MAX_TRAINING_STEPS: int | None = Field(default=200_000, ge=1)

    # --- Workers & Batching ---
    # More workers for faster data generation (adjust based on CPU cores)
    NUM_SELF_PLAY_WORKERS: int = Field(12, ge=1)
    WORKER_DEVICE: Literal["auto", "cuda", "cpu", "mps"] = Field(
        "cpu"
    )  # Workers usually on CPU
    # Larger batch size for more stable gradients
    BATCH_SIZE: int = Field(128, ge=1)
    # Significantly larger buffer
    BUFFER_CAPACITY: int = Field(100_000, ge=1)
    # Start training only after a decent amount of data is collected
    MIN_BUFFER_SIZE_TO_TRAIN: int = Field(10_000, ge=1)
    # Update worker networks less frequently to reduce overhead
    WORKER_UPDATE_FREQ_STEPS: int = Field(100, ge=1)

    # --- Optimizer ---
    OPTIMIZER_TYPE: Literal["Adam", "AdamW", "SGD"] = Field("AdamW")
    LEARNING_RATE: float = Field(1e-4, gt=0)  # Common starting point
    WEIGHT_DECAY: float = Field(1e-4, ge=0)  # Slightly higher weight decay
    GRADIENT_CLIP_VALUE: float | None = Field(default=1.0)  # Keep gradient clipping

    # --- LR Scheduler ---
    LR_SCHEDULER_TYPE: Literal["StepLR", "CosineAnnealingLR"] | None = Field(
        default="CosineAnnealingLR"
    )
    LR_SCHEDULER_T_MAX: int | None = Field(
        default=None  # Set automatically based on MAX_TRAINING_STEPS
    )
    LR_SCHEDULER_ETA_MIN: float = Field(1e-6, ge=0)  # End LR

    # --- Loss Weights ---
    POLICY_LOSS_WEIGHT: float = Field(1.0, ge=0)
    VALUE_LOSS_WEIGHT: float = Field(1.0, ge=0)
    ENTROPY_BONUS_WEIGHT: float = Field(
        0.01, ge=0
    )  # Small entropy bonus can help exploration

    # --- Checkpointing ---
    # Save checkpoints less frequently
    CHECKPOINT_SAVE_FREQ_STEPS: int = Field(1000, ge=1)

    # --- Prioritized Experience Replay (PER) ---
    USE_PER: bool = Field(True)  # Keep PER enabled
    PER_ALPHA: float = Field(0.6, ge=0)  # Standard value
    PER_BETA_INITIAL: float = Field(0.4, ge=0, le=1.0)  # Standard value
    PER_BETA_FINAL: float = Field(1.0, ge=0, le=1.0)  # Anneal to 1.0
    PER_BETA_ANNEAL_STEPS: int | None = Field(
        None  # Set automatically based on MAX_TRAINING_STEPS
    )
    PER_EPSILON: float = Field(1e-5, gt=0)  # Small value to avoid zero priority

    @model_validator(mode="after")
    def check_buffer_sizes(self) -> "TrainConfig":
        if self.MIN_BUFFER_SIZE_TO_TRAIN > self.BUFFER_CAPACITY:
            raise ValueError(
                "MIN_BUFFER_SIZE_TO_TRAIN cannot be greater than BUFFER_CAPACITY."
            )
        if self.BATCH_SIZE > self.BUFFER_CAPACITY:
            raise ValueError("BATCH_SIZE cannot be greater than BUFFER_CAPACITY.")
        if self.BATCH_SIZE > self.MIN_BUFFER_SIZE_TO_TRAIN:
            # This is acceptable, just means training starts later
            pass
        return self

    @model_validator(mode="after")
    def set_scheduler_t_max(self) -> "TrainConfig":
        if (
            self.LR_SCHEDULER_TYPE == "CosineAnnealingLR"
            and self.LR_SCHEDULER_T_MAX is None
        ):
            if self.MAX_TRAINING_STEPS is not None:
                self.LR_SCHEDULER_T_MAX = self.MAX_TRAINING_STEPS
                print(
                    f"Set LR_SCHEDULER_T_MAX to MAX_TRAINING_STEPS ({self.MAX_TRAINING_STEPS})"
                )
            else:
                # Set a very large default if MAX_TRAINING_STEPS is infinite
                self.LR_SCHEDULER_T_MAX = 1_000_000
                print(
                    f"Warning: MAX_TRAINING_STEPS is None, setting LR_SCHEDULER_T_MAX to default {self.LR_SCHEDULER_T_MAX}"
                )
        if self.LR_SCHEDULER_T_MAX is not None and self.LR_SCHEDULER_T_MAX <= 0:
            raise ValueError("LR_SCHEDULER_T_MAX must be positive if set.")
        return self

    @model_validator(mode="after")
    def set_per_beta_anneal_steps(self) -> "TrainConfig":
        if self.USE_PER and self.PER_BETA_ANNEAL_STEPS is None:
            if self.MAX_TRAINING_STEPS is not None:
                self.PER_BETA_ANNEAL_STEPS = self.MAX_TRAINING_STEPS
                print(
                    f"Set PER_BETA_ANNEAL_STEPS to MAX_TRAINING_STEPS ({self.MAX_TRAINING_STEPS})"
                )
            else:
                # Set a very large default if MAX_TRAINING_STEPS is infinite
                self.PER_BETA_ANNEAL_STEPS = 1_000_000
                print(
                    f"Warning: MAX_TRAINING_STEPS is None, setting PER_BETA_ANNEAL_STEPS to default {self.PER_BETA_ANNEAL_STEPS}"
                )
        if self.PER_BETA_ANNEAL_STEPS is not None and self.PER_BETA_ANNEAL_STEPS <= 0:
            raise ValueError("PER_BETA_ANNEAL_STEPS must be positive if set.")
        return self

    @field_validator("GRADIENT_CLIP_VALUE")
    @classmethod
    def check_gradient_clip(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError("GRADIENT_CLIP_VALUE must be positive if set.")
        return v

    @field_validator("PER_BETA_FINAL")
    @classmethod
    def check_per_beta_final(cls, v: float, info) -> float:
        initial_beta = info.data.get("PER_BETA_INITIAL")
        if initial_beta is not None and v < initial_beta:
            raise ValueError("PER_BETA_FINAL cannot be less than PER_BETA_INITIAL")
        return v
