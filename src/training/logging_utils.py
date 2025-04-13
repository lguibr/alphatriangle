# File: src/training/logging_utils.py
import logging
import os
import sys
import io  # Import io module
from typing import TextIO, Optional, List

from src.config import PersistenceConfig
import mlflow  # Import mlflow here
import numpy as np  # Import numpy here
import torch  # Import torch here
from collections import deque  # Import deque here

# Import TrainingComponents for type hinting
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .components import TrainingComponents


# --- Tee Class ---
class Tee:
    """
    Helper class to duplicate stream output to multiple targets.
    Includes fileno() method for compatibility with modules like faulthandler.
    """

    def __init__(
        self, *streams: TextIO, main_stream_for_fileno: Optional[TextIO] = None
    ):
        self.streams = streams
        # Store the stream whose fileno should be reported (usually original stderr/stdout)
        self.main_stream_for_fileno = main_stream_for_fileno

    def write(self, message: str):
        for stream in self.streams:
            try:
                stream.write(message)
            except Exception as e:
                # Write error to the original stderr to avoid recursion if logging fails
                print(f"Tee Error writing: {e}", file=sys.__stderr__)
        self.flush()

    def flush(self):
        for stream in self.streams:
            try:
                if hasattr(stream, "flush"):
                    stream.flush()
            except Exception as e:
                print(f"Tee Error flushing: {e}", file=sys.__stderr__)

    def isatty(self) -> bool:
        # Return True if any underlying stream is a TTY
        # Prioritize the main_stream if it exists and has isatty
        if self.main_stream_for_fileno and hasattr(
            self.main_stream_for_fileno, "isatty"
        ):
            return self.main_stream_for_fileno.isatty()
        # Otherwise, check other streams
        return any(getattr(s, "isatty", lambda: False)() for s in self.streams)

    def fileno(self) -> int:
        """
        Return the file descriptor of the main_stream_for_fileno.
        Required by modules like faulthandler.
        """
        if self.main_stream_for_fileno and hasattr(
            self.main_stream_for_fileno, "fileno"
        ):
            try:
                return self.main_stream_for_fileno.fileno()
            except io.UnsupportedOperation:
                # If the main stream doesn't support fileno, raise the error
                raise
            except Exception as e:
                print(
                    f"Tee Error getting fileno from main stream: {e}",
                    file=sys.__stderr__,
                )
                raise io.UnsupportedOperation("Main stream fileno failed") from e
        # If no main stream specified or it lacks fileno, raise error
        raise io.UnsupportedOperation(
            "Tee object does not have a valid file descriptor."
        )


# --- End Tee Class ---

# Get logger instance for this module
logger = logging.getLogger(__name__)


def get_root_logger() -> logging.Logger:
    """Gets the root logger instance."""
    return logging.getLogger()


def setup_file_logging(
    persist_config: PersistenceConfig, run_name: str, mode_suffix: str
) -> str:
    """Sets up file logging for the current run."""
    run_base_dir = persist_config.get_run_base_dir(run_name)
    log_dir = os.path.join(run_base_dir, persist_config.LOG_DIR_NAME)
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"{run_name}_{mode_suffix}.log")

    file_handler = logging.FileHandler(log_file_path, mode="w")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(formatter)

    root_logger = get_root_logger()
    # Remove existing file handlers to avoid duplicates if called multiple times
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            try:
                handler.close()  # Close before removing
            except Exception as e:
                logger.error(f"Error closing existing file handler: {e}")
            root_logger.removeHandler(handler)

    root_logger.addHandler(file_handler)
    return log_file_path


def log_configs_to_mlflow(components: "TrainingComponents"):
    """Logs configuration parameters to MLflow."""
    try:
        from src.config import APP_NAME

        mlflow.log_param("APP_NAME", APP_NAME)
        # Use model_dump for Pydantic v2
        mlflow.log_params(components.train_config.model_dump())
        mlflow.log_params(components.env_config.model_dump())
        mlflow.log_params(components.model_config.model_dump())
        mlflow.log_params(components.mcts_config.model_dump())
        persist_params = components.persist_config.model_dump(
            exclude={"MLFLOW_TRACKING_URI"}
        )
        mlflow.log_params(persist_params)
        logger.info("Logged configuration parameters to MLflow.")

        # Save config JSON artifact
        all_configs = {
            "train_config": components.train_config.model_dump(),
            "env_config": components.env_config.model_dump(),
            "model_config": components.model_config.model_dump(),
            "mcts_config": components.mcts_config.model_dump(),
            "persist_config": components.persist_config.model_dump(),
        }
        components.data_manager.save_run_config(all_configs)

    except Exception as e:
        logger.error(f"Failed to log parameters/configs to MLflow: {e}", exc_info=True)


def log_metrics_to_mlflow(metrics: dict, step: int):
    """Logs a dictionary of metrics to MLflow."""
    try:
        numeric_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.number)) and np.isfinite(v):
                numeric_metrics[k] = float(v)
            else:
                logger.debug(
                    f"Skipping non-finite/non-numeric metric for MLflow: {k}={v} (type: {type(v)})"
                )
        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics, step=step)
    except Exception as e:
        logger.error(f"Failed to log metrics to MLflow: {e}")
