# File: alphatriangle/config/persistence_config.py
from pathlib import Path

from pydantic import BaseModel, Field, computed_field


class PersistenceConfig(BaseModel):
    """Configuration for saving/loading artifacts (Pydantic model)."""

    ROOT_DATA_DIR: str = Field(default=".alphatriangle_data")
    RUNS_DIR_NAME: str = Field(default="runs")
    MLFLOW_DIR_NAME: str = Field(default="mlruns")

    CHECKPOINT_SAVE_DIR_NAME: str = Field(default="checkpoints")
    BUFFER_SAVE_DIR_NAME: str = Field(default="buffers")
    LOG_DIR_NAME: str = Field(default="logs")
    TENSORBOARD_DIR_NAME: str = Field(default="tensorboard")

    LATEST_CHECKPOINT_FILENAME: str = Field(default="latest.pkl")
    BEST_CHECKPOINT_FILENAME: str = Field(default="best.pkl")
    BUFFER_FILENAME: str = Field(default="buffer.pkl")
    CONFIG_FILENAME: str = Field(default="configs.json")

    RUN_NAME: str = Field(default="default_run")

    SAVE_BUFFER: bool = Field(default=True)
    BUFFER_SAVE_FREQ_STEPS: int = Field(default=1000, ge=1)

    def _get_absolute_root(self) -> Path:
        """Resolves ROOT_DATA_DIR to an absolute path."""
        # Use Path.cwd() as the base if ROOT_DATA_DIR is relative
        if not self.ROOT_DATA_DIR:  # Handle empty string case
            return Path.cwd() / ".alphatriangle_data"  # Default if empty
        return Path(self.ROOT_DATA_DIR).resolve()

    @computed_field  # type: ignore[misc] # Decorator requires Pydantic v2
    @property
    def MLFLOW_TRACKING_URI(self) -> str:
        """Constructs the absolute file URI for MLflow tracking using pathlib."""
        if hasattr(self, "ROOT_DATA_DIR") and hasattr(self, "MLFLOW_DIR_NAME"):
            root_path = self._get_absolute_root()
            abs_path = root_path.joinpath(self.MLFLOW_DIR_NAME).resolve()
            return abs_path.as_uri()
        return ""

    def get_run_base_dir(self, run_name: str | None = None) -> str:
        """Gets the absolute base directory path for a specific run."""
        if not hasattr(self, "ROOT_DATA_DIR") or not hasattr(self, "RUNS_DIR_NAME"):
            return ""
        name = run_name if run_name else self.RUN_NAME
        root_path = self._get_absolute_root()
        return str(root_path.joinpath(self.RUNS_DIR_NAME, name))

    def get_mlflow_abs_path(self) -> str:
        """Gets the absolute OS path to the MLflow directory as a string."""
        if not hasattr(self, "ROOT_DATA_DIR") or not hasattr(self, "MLFLOW_DIR_NAME"):
            return ""
        root_path = self._get_absolute_root()
        abs_path = root_path.joinpath(self.MLFLOW_DIR_NAME).resolve()
        return str(abs_path)

    def get_tensorboard_log_dir(self, run_name: str | None = None) -> str:
        """Gets the absolute directory path for TensorBoard logs for a specific run."""
        run_base = self.get_run_base_dir(run_name)  # This now returns absolute path
        if not run_base or not hasattr(self, "TENSORBOARD_DIR_NAME"):
            return ""
        # Path() constructor handles string paths correctly
        return str(Path(run_base) / self.TENSORBOARD_DIR_NAME)


# Ensure model is rebuilt after changes
PersistenceConfig.model_rebuild(force=True)
