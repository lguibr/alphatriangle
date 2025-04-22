# File: alphatriangle/training/logging_utils.py
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow

if TYPE_CHECKING:
    import io

    from .components import TrainingComponents

logger = logging.getLogger(__name__)


class Tee:
    """Helper class to duplicate stdout/stderr to a file."""

    def __init__(self, filename: str, mode: str = "a"):
        self.filename = filename
        self.mode = mode
        self.file: io.TextIOWrapper | None = None
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def __enter__(self):
        # Use Path().open() within context manager
        self.file = Path(self.filename).open(self.mode, encoding="utf-8")
        sys.stdout = self  # type: ignore
        sys.stderr = self  # type: ignore
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        if self.file:
            self.file.close()
            self.file = None  # Reset file handle

    def write(self, data):
        if self.file:
            self.file.write(data)
        self.stdout.write(data)  # Write to original stdout

    def flush(self):
        if self.file:
            self.file.flush()
        self.stdout.flush()


def get_root_logger() -> logging.Logger:
    """Gets the root logger instance."""
    return logging.getLogger()


def setup_file_logging(
    persist_config: Any, run_name: str, log_prefix: str = "run"
) -> str:
    """Sets up file logging for the current run."""
    log_dir = (
        Path(persist_config.get_run_base_dir(run_name)) / persist_config.LOG_DIR_NAME
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = f"{log_prefix}_{run_name}.log"
    log_file_path = log_dir / log_filename

    root_logger = get_root_logger()

    # Check if a file handler for this path already exists
    handler_exists = any(
        isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file_path
        for h in root_logger.handlers
    )

    if not handler_exists:
        file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        logger.info(f"Added file handler for: {log_file_path}")
    else:
        logger.warning("File handler already exists for root logger.")

    return str(log_file_path)


def log_configs_to_mlflow(components: "TrainingComponents"):
    """Logs all configuration parameters to MLflow."""
    logger.info("Logging configurations to MLflow...")
    configs_to_log = {
        "TrainConfig": components.train_config.model_dump(),
        "ModelConfig": components.model_config.model_dump(),
        "EnvConfig": components.env_config.model_dump(),
        "MCTSConfig": components.mcts_config.model_dump(),
        "PersistenceConfig": components.persist_config.model_dump(),
    }

    # Log individual parameters
    for config_name, config_dict in configs_to_log.items():
        for key, value in config_dict.items():
            param_name = f"{config_name}/{key}"
            # Truncate long lists/dicts for MLflow param logging
            if isinstance(value, list) and len(value) > 10:
                value_str = f"[List len={len(value)}]"
            elif isinstance(value, dict) and len(value) > 10:
                value_str = f"{{Dict keys={len(value)}}}"
            else:
                value_str = str(value)

            # MLflow param values have a limit (e.g., 250 chars)
            if len(value_str) > 250:
                value_str = value_str[:247] + "..."
            try:
                mlflow.log_param(param_name, value_str)
            except Exception as e:
                logger.warning(f"Could not log param '{param_name}' to MLflow: {e}")

    # Save combined config as artifact
    try:
        components.data_manager.save_run_config(configs_to_log)
        logger.info("Logged combined config JSON as MLflow artifact.")
    except Exception as e:
        logger.error(f"Failed to save/log run config JSON: {e}", exc_info=True)
