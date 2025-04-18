# File: alphatriangle/config/train_config.py
import logging
import time
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# Get logger instance
logger = logging.getLogger(__name__)


class TrainConfig(BaseModel):
    """
    Configuration for the training process (Pydantic model).
    --- TUNED FOR MORE SUBSTANTIAL LEARNING RUNS ---
    """

    RUN_NAME: str = Field(
        # More descriptive default run name
        default_factory=lambda: f"train_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    LOAD_CHECKPOINT_PATH: str | None = Field(default=None)
    LOAD_BUFFER_PATH: str | None = Field(default=None)
    AUTO_RESUME_LATEST: bool = Field(default=True)  # Resume if possible
    # --- DEVICE: Defaults to 'auto' for automatic detection (CUDA > MPS > CPU) ---
    # This controls the device for the main Trainer process.
    DEVICE: Literal["auto", "cuda", "cpu", "mps"] = Field(default="auto")
    RANDOM_SEED: int = Field(default=42)

    # --- Training Loop ---
    MAX_TRAINING_STEPS: int | None = Field(default=100_000, ge=1)  # Target steps

    # --- Workers & Batching ---
    NUM_SELF_PLAY_WORKERS: int = Field(
        default=8,  # Default workers, capped by cores
        ge=1,
        description="Suggested number of workers. Actual number may be adjusted based on detected CPU cores.",
    )
    # --- WORKER_DEVICE: Defaults to 'cpu' for self-play workers ---
    # Workers run MCTS and NN eval; CPU is often sufficient and avoids GPU contention.
    WORKER_DEVICE: Literal["auto", "cuda", "cpu", "mps"] = Field(default="cpu")
    BATCH_SIZE: int = Field(default=128, ge=1)  # Moderate batch size
    BUFFER_CAPACITY: int = Field(default=200_000, ge=1)  # Larger buffer
    MIN_BUFFER_SIZE_TO_TRAIN: int = Field(
        default=20_000,
        ge=1,  # Start training after 10% fill
    )
    WORKER_UPDATE_FREQ_STEPS: int = Field(default=500, ge=1)

    # --- N-Step Returns ---
    N_STEP_RETURNS: int = Field(default=5, ge=1)  # 5-step returns
    GAMMA: float = Field(default=0.99, gt=0, le=1.0)  # Discount factor

    # --- Optimizer ---
    OPTIMIZER_TYPE: Literal["Adam", "AdamW", "SGD"] = Field(default="AdamW")
    LEARNING_RATE: float = Field(default=2e-4, gt=0)
    WEIGHT_DECAY: float = Field(default=1e-4, ge=0)
    GRADIENT_CLIP_VALUE: float | None = Field(default=1.0)

    # --- LR Scheduler ---
    LR_SCHEDULER_TYPE: Literal["StepLR", "CosineAnnealingLR"] | None = Field(
        default="CosineAnnealingLR"
    )
    # T_MAX will be set automatically based on new MAX_TRAINING_STEPS
    LR_SCHEDULER_T_MAX: int | None = Field(default=None)
    LR_SCHEDULER_ETA_MIN: float = Field(default=1e-6, ge=0)

    # --- Loss Weights ---
    POLICY_LOSS_WEIGHT: float = Field(default=1.0, ge=0)
    VALUE_LOSS_WEIGHT: float = Field(default=1.0, ge=0)
    ENTROPY_BONUS_WEIGHT: float = Field(default=0.001, ge=0)  # Small entropy bonus

    # --- Checkpointing ---
    CHECKPOINT_SAVE_FREQ_STEPS: int = Field(default=2500, ge=1)  # Save every 2500 steps

    # --- Prioritized Experience Replay (PER) ---
    USE_PER: bool = Field(default=True)  # Enable PER
    PER_ALPHA: float = Field(default=0.6, ge=0)
    PER_BETA_INITIAL: float = Field(default=0.4, ge=0, le=1.0)
    PER_BETA_FINAL: float = Field(default=1.0, ge=0, le=1.0)
    # Anneal steps will be set automatically based on MAX_TRAINING_STEPS
    PER_BETA_ANNEAL_STEPS: int | None = Field(default=None)
    PER_EPSILON: float = Field(default=1e-5, gt=0)

    # --- Model Compilation ---
    COMPILE_MODEL: bool = Field(
        default=True,
        description=(
            "Enable torch.compile() for potential speedup (Trainer only). Requires PyTorch 2.0+. "
            "May have initial overhead or compatibility issues on some setups/GPUs "
            "(especially non-CUDA backends like MPS). Set to False if encountering problems. "
            "The application will attempt compilation and fall back gracefully if it fails."
        ),
    )

    @model_validator(mode="after")
    def check_buffer_sizes(self) -> "TrainConfig":
        # Ensure attributes exist before comparing
        if (
            hasattr(self, "MIN_BUFFER_SIZE_TO_TRAIN")
            and hasattr(self, "BUFFER_CAPACITY")
            and self.MIN_BUFFER_SIZE_TO_TRAIN > self.BUFFER_CAPACITY
        ):
            raise ValueError(
                "MIN_BUFFER_SIZE_TO_TRAIN cannot be greater than BUFFER_CAPACITY."
            )
        if (
            hasattr(self, "BATCH_SIZE")
            and hasattr(self, "BUFFER_CAPACITY")
            and self.BATCH_SIZE > self.BUFFER_CAPACITY
        ):
            raise ValueError("BATCH_SIZE cannot be greater than BUFFER_CAPACITY.")
        if (
            hasattr(self, "BATCH_SIZE")
            and hasattr(self, "MIN_BUFFER_SIZE_TO_TRAIN")
            and self.BATCH_SIZE > self.MIN_BUFFER_SIZE_TO_TRAIN
        ):
            # Allow batch size to be larger than min buffer size (will just wait longer)
            pass
        return self

    @model_validator(mode="after")
    def set_scheduler_t_max(self) -> "TrainConfig":
        # Ensure attributes exist before checking
        if (
            hasattr(self, "LR_SCHEDULER_TYPE")
            and self.LR_SCHEDULER_TYPE == "CosineAnnealingLR"
            and hasattr(self, "LR_SCHEDULER_T_MAX")
            and self.LR_SCHEDULER_T_MAX is None  # Only set if not manually specified
        ):
            if (
                hasattr(self, "MAX_TRAINING_STEPS")
                and self.MAX_TRAINING_STEPS is not None
            ):
                # Assign to self.LR_SCHEDULER_T_MAX only if MAX_TRAINING_STEPS is valid
                if self.MAX_TRAINING_STEPS >= 1:
                    self.LR_SCHEDULER_T_MAX = self.MAX_TRAINING_STEPS
                    logger.info(
                        f"Set LR_SCHEDULER_T_MAX to MAX_TRAINING_STEPS ({self.MAX_TRAINING_STEPS})"
                    )
                else:
                    # Handle invalid MAX_TRAINING_STEPS case if necessary
                    self.LR_SCHEDULER_T_MAX = 100_000  # Fallback (matches new default)
                    logger.warning(
                        f"Warning: MAX_TRAINING_STEPS is invalid ({self.MAX_TRAINING_STEPS}), setting LR_SCHEDULER_T_MAX to default {self.LR_SCHEDULER_T_MAX}"
                    )
            else:
                self.LR_SCHEDULER_T_MAX = 100_000  # Fallback (matches new default)
                logger.warning(
                    f"Warning: MAX_TRAINING_STEPS is None, setting LR_SCHEDULER_T_MAX to default {self.LR_SCHEDULER_T_MAX}"
                )

        if (
            hasattr(self, "LR_SCHEDULER_T_MAX")
            and self.LR_SCHEDULER_T_MAX is not None
            and self.LR_SCHEDULER_T_MAX <= 0
        ):
            raise ValueError("LR_SCHEDULER_T_MAX must be positive if set.")
        return self

    @model_validator(mode="after")
    def set_per_beta_anneal_steps(self) -> "TrainConfig":
        # Ensure attributes exist before checking
        if (
            hasattr(self, "USE_PER")
            and self.USE_PER
            and hasattr(self, "PER_BETA_ANNEAL_STEPS")
            and self.PER_BETA_ANNEAL_STEPS is None  # Only set if not manually specified
        ):
            if (
                hasattr(self, "MAX_TRAINING_STEPS")
                and self.MAX_TRAINING_STEPS is not None
            ):
                # Assign to self.PER_BETA_ANNEAL_STEPS only if MAX_TRAINING_STEPS is valid
                if self.MAX_TRAINING_STEPS >= 1:
                    self.PER_BETA_ANNEAL_STEPS = self.MAX_TRAINING_STEPS
                    logger.info(
                        f"Set PER_BETA_ANNEAL_STEPS to MAX_TRAINING_STEPS ({self.MAX_TRAINING_STEPS})"
                    )
                else:
                    # Handle invalid MAX_TRAINING_STEPS case if necessary
                    self.PER_BETA_ANNEAL_STEPS = (
                        100_000  # Fallback (matches new default)
                    )
                    logger.warning(
                        f"Warning: MAX_TRAINING_STEPS is invalid ({self.MAX_TRAINING_STEPS}), setting PER_BETA_ANNEAL_STEPS to default {self.PER_BETA_ANNEAL_STEPS}"
                    )
            else:
                self.PER_BETA_ANNEAL_STEPS = 100_000  # Fallback (matches new default)
                logger.warning(
                    f"Warning: MAX_TRAINING_STEPS is None, setting PER_BETA_ANNEAL_STEPS to default {self.PER_BETA_ANNEAL_STEPS}"
                )

        if (
            hasattr(self, "PER_BETA_ANNEAL_STEPS")
            and self.PER_BETA_ANNEAL_STEPS is not None
            and self.PER_BETA_ANNEAL_STEPS <= 0
        ):
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
        # info.data might not be available during initial validation in Pydantic v2
        # Check 'values' if info.data is empty
        data = info.data if info.data else info.values
        initial_beta = data.get("PER_BETA_INITIAL")
        if initial_beta is not None and v < initial_beta:
            raise ValueError("PER_BETA_FINAL cannot be less than PER_BETA_INITIAL")
        return v


# Ensure model is rebuilt after changes
TrainConfig.model_rebuild(force=True)
