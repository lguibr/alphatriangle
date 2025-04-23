# File: alphatriangle/training/logging_utils.py
import logging
from typing import TYPE_CHECKING

import mlflow

if TYPE_CHECKING:
    from .components import TrainingComponents

logger = logging.getLogger(__name__)


def log_configs_to_mlflow(components: "TrainingComponents"):
    """Logs AlphaTriangle-specific configurations to MLflow."""
    if not components.trieye_actor:
        logger.warning("Cannot log configs to MLflow: TrieyeActor handle is missing.")
        return

    try:
        # Log AlphaTriangle specific configs
        mlflow.log_params(components.train_config.model_dump())
        mlflow.log_params(components.model_config.model_dump())
        mlflow.log_params(components.env_config.model_dump())
        # MCTS config is SearchConfiguration, convert to dict
        mcts_dict = {
            f"mcts_{k}": v for k, v in components.mcts_config.model_dump().items()
        }
        mlflow.log_params(mcts_dict)
        logger.info("Logged AlphaTriangle configurations to MLflow.")

        # Note: TrieyeConfig is logged automatically by the TrieyeActor itself
        # when save_initial_config is called internally.

    except Exception as e:
        logger.error(f"Failed to log configurations to MLflow: {e}", exc_info=True)
