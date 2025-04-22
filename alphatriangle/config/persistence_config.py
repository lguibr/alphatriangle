from pathlib import Path

from pydantic import BaseModel, Field, computed_field


class PersistenceConfig(BaseModel):
    """Configuration for saving/loading artifacts (Pydantic model)."""

    # Root directory for all persistent data, relative to project root.
    ROOT_DATA_DIR: str = Field(default=".alphatriangle_data")
    RUNS_DIR_NAME: str = Field(default="runs")
    MLFLOW_DIR_NAME: str = Field(default="mlruns")

    CHECKPOINT_SAVE_DIR_NAME: str = Field(default="checkpoints")
    BUFFER_SAVE_DIR_NAME: str = Field(default="buffers")
    LOG_DIR_NAME: str = Field(default="logs")
    TENSORBOARD_DIR_NAME: str = Field(default="tensorboard")
    PROFILE_DIR_NAME: str = Field(default="profile_data")  # Added for consistency

    LATEST_CHECKPOINT_FILENAME: str = Field(default="latest.pkl")
    BEST_CHECKPOINT_FILENAME: str = Field(default="best.pkl")
    BUFFER_FILENAME: str = Field(default="buffer.pkl")
    CONFIG_FILENAME: str = Field(default="configs.json")

    RUN_NAME: str = Field(default="default_run")

    SAVE_BUFFER: bool = Field(default=True)
    BUFFER_SAVE_FREQ_STEPS: int = Field(default=1000, ge=1)

    def _get_absolute_root(self) -> Path:
        """Resolves ROOT_DATA_DIR to an absolute path relative to the project root."""
        # Assume the config file is loaded relative to the project root
        # or the script is run from the project root.
        project_root = Path.cwd()  # Or determine project root differently if needed
        root_path = project_root / self.ROOT_DATA_DIR
        return root_path.resolve()

    @computed_field  # type: ignore[misc] # Decorator requires Pydantic v2
    @property
    def MLFLOW_TRACKING_URI(self) -> str:
        """Constructs the absolute file URI for MLflow tracking using pathlib."""
        if hasattr(self, "MLFLOW_DIR_NAME"):
            abs_path = self.get_mlflow_abs_path()
            # Ensure the path exists for the URI to be valid for mlflow ui
            abs_path.mkdir(parents=True, exist_ok=True)
            return abs_path.as_uri()
        return ""

    def get_runs_root_dir(self) -> Path:
        """Gets the absolute path to the directory containing all runs."""
        if not hasattr(self, "RUNS_DIR_NAME"):
            return Path()  # Return empty path if config is incomplete
        root_path = self._get_absolute_root()
        return root_path / self.RUNS_DIR_NAME

    def get_run_base_dir(self, run_name: str | None = None) -> Path:
        """Gets the absolute base directory path for a specific run."""
        runs_root = self.get_runs_root_dir()
        name = run_name if run_name else self.RUN_NAME
        return runs_root / name

    def get_mlflow_abs_path(self) -> Path:
        """Gets the absolute OS path to the MLflow directory."""
        if not hasattr(self, "MLFLOW_DIR_NAME"):
            return Path()
        root_path = self._get_absolute_root()
        return root_path / self.MLFLOW_DIR_NAME

    def get_tensorboard_log_dir(self, run_name: str | None = None) -> Path:
        """Gets the absolute directory path for TensorBoard logs for a specific run."""
        run_base = self.get_run_base_dir(run_name)
        if not run_base or not hasattr(self, "TENSORBOARD_DIR_NAME"):
            return Path()
        return run_base / self.TENSORBOARD_DIR_NAME

    def get_profile_data_dir(self, run_name: str | None = None) -> Path:
        """Gets the absolute directory path for profile data for a specific run."""
        run_base = self.get_run_base_dir(run_name)
        if not run_base or not hasattr(self, "PROFILE_DIR_NAME"):
            return Path()
        return run_base / self.PROFILE_DIR_NAME


# Ensure model is rebuilt after changes
PersistenceConfig.model_rebuild(force=True)
