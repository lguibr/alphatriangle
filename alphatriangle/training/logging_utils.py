import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
import numpy as np

if TYPE_CHECKING:
    from alphatriangle.config import PersistenceConfig

    from .components import TrainingComponents

logger = logging.getLogger(__name__)


class Tee:
    """Helper class to redirect stdout/stderr to both console and a file."""

    def __init__(self, stream1, stream2, main_stream_for_fileno):
        self.stream1 = stream1
        self.stream2 = stream2
        self._main_stream_for_fileno = main_stream_for_fileno

    def write(self, data):
        self.stream1.write(data)
        self.stream2.write(data)
        self.flush()

    def flush(self):
        self.stream1.flush()
        self.stream2.flush()

    def fileno(self):
        return self._main_stream_for_fileno.fileno()

    def isatty(self):
        return self._main_stream_for_fileno.isatty()


def get_root_logger() -> logging.Logger:
    """Gets the root logger instance."""
    return logging.getLogger()


def setup_file_logging(
    persist_config: "PersistenceConfig", run_name: str, mode_suffix: str
) -> str:
    """Sets up file logging for the current run."""
    run_base_dir = Path(persist_config.get_run_base_dir(run_name))
    log_dir = run_base_dir / persist_config.LOG_DIR_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / f"{run_name}_{mode_suffix}.log"

    file_handler = logging.FileHandler(log_file_path, mode="w")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(formatter)

    root_logger = get_root_logger()
    if not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
        root_logger.addHandler(file_handler)
        logger.info(f"Added file handler logging to: {log_file_path}")
    else:
        logger.warning("File handler already exists for root logger.")

    return str(log_file_path)


def log_configs_to_mlflow(components: "TrainingComponents"):
    """Logs configuration parameters to MLflow."""
    if not mlflow.active_run():
        logger.warning("No active MLflow run found. Cannot log configs.")
        return

    logger.info("Logging configuration parameters to MLflow...")
    try:
        mlflow.log_params(components.env_config.model_dump())
        mlflow.log_params(components.model_config.model_dump())
        mlflow.log_params(components.train_config.model_dump())
        mlflow.log_params(components.mcts_config.model_dump())
        mlflow.log_params(components.persist_config.model_dump())
        logger.info("Configuration parameters logged to MLflow.")
    except Exception as e:
        logger.error(f"Failed to log parameters to MLflow: {e}", exc_info=True)


def log_metrics_to_mlflow(metrics: dict[str, Any], step: int):
    """Logs metrics to MLflow."""
    if not mlflow.active_run():
        logger.warning("No active MLflow run found. Cannot log metrics.")
        return

    try:
        # Filter only numeric, finite metrics
        numeric_metrics = {}
        for k, v in metrics.items():
            # Use isinstance with | for multiple types
            if isinstance(v, int | float | np.number) and np.isfinite(v):
                numeric_metrics[k] = float(v)
            else:
                logger.debug(
                    f"Skipping non-numeric or non-finite metric for MLflow: {k}={v} (type: {type(v)})"
                )
        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics, step=step)
            logger.debug(
                f"Logged {len(numeric_metrics)} metrics to MLflow at step {step}."
            )
        else:
            logger.debug(f"No valid numeric metrics to log at step {step}.")
    except Exception as e:
        logger.error(f"Failed to log metrics to MLflow: {e}", exc_info=True)
