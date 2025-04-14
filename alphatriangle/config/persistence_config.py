from pathlib import Path

from pydantic import BaseModel, Field, computed_field


class PersistenceConfig(BaseModel):
    """Configuration for saving/loading artifacts (Pydantic model)."""

    ROOT_DATA_DIR: str = Field(default=".alphatriangle_data")
    RUNS_DIR_NAME: str = Field(default="runs")
    MLFLOW_DIR_NAME: str = Field(default="mlruns")

    CHECKPOINT_SAVE_DIR_NAME: str = Field(default="checkpoints")
    BUFFER_SAVE_DIR_NAME: str = Field(default="buffers")
    GAME_STATE_SAVE_DIR_NAME: str = Field(default="game_states")
    LOG_DIR_NAME: str = Field(default="logs")

    LATEST_CHECKPOINT_FILENAME: str = Field(default="latest.pkl")
    BEST_CHECKPOINT_FILENAME: str = Field(default="best.pkl")
    BUFFER_FILENAME: str = Field(default="buffer.pkl")
    CONFIG_FILENAME: str = Field(default="configs.json")

    RUN_NAME: str = Field(default="default_run")

    SAVE_GAME_STATES: bool = Field(default=False)
    GAME_STATE_SAVE_FREQ_EPISODES: int = Field(default=5, ge=1)

    SAVE_BUFFER: bool = Field(default=True)
    BUFFER_SAVE_FREQ_STEPS: int = Field(default=10, ge=1)

    @computed_field  # type: ignore[misc] # Decorator requires Pydantic v2
    @property
    def MLFLOW_TRACKING_URI(self) -> str:
        """Constructs the file URI for MLflow tracking using pathlib."""
        # Ensure attributes exist before calculating
        if hasattr(self, "ROOT_DATA_DIR") and hasattr(self, "MLFLOW_DIR_NAME"):
            abs_path = Path(self.ROOT_DATA_DIR).joinpath(self.MLFLOW_DIR_NAME).resolve()
            return abs_path.as_uri()
        return ""

    def get_run_base_dir(self, run_name: str | None = None) -> str:
        """Gets the base directory for a specific run."""
        # Ensure attributes exist before calculating
        if not hasattr(self, "ROOT_DATA_DIR") or not hasattr(self, "RUNS_DIR_NAME"):
            return ""  # Fallback
        name = run_name if run_name else self.RUN_NAME
        return str(Path(self.ROOT_DATA_DIR).joinpath(self.RUNS_DIR_NAME, name))

    def get_mlflow_abs_path(self) -> str:
        """Gets the absolute OS path to the MLflow directory as a string."""
        # Ensure attributes exist before calculating
        if not hasattr(self, "ROOT_DATA_DIR") or not hasattr(self, "MLFLOW_DIR_NAME"):
            return ""  # Fallback
        abs_path = Path(self.ROOT_DATA_DIR).joinpath(self.MLFLOW_DIR_NAME).resolve()
        return str(abs_path)


# Ensure model is rebuilt after changes
PersistenceConfig.model_rebuild(force=True)
