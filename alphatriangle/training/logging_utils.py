import logging
from typing import TYPE_CHECKING

import mlflow
from pydantic import BaseModel

if TYPE_CHECKING:
    from .components import TrainingComponents

logger = logging.getLogger(__name__)


class Tee:
    """Helper class to redirect stdout/stderr to logger."""

    def __init__(self, name: str, stream, log_level: int = logging.INFO):
        self.name = name
        self.stream = stream
        self.logger = logging.getLogger(name)
        self.log_level = log_level
        self.linebuf = ""

    def write(self, buf):
        self.stream.write(buf)
        self.stream.flush()
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        self.stream.flush()


def log_configs_to_mlflow(components: "TrainingComponents"):
    """Logs all configuration parameters to MLflow."""
    logger.info("Logging configurations to MLflow...")
    configs_to_log = {
        "TrainConfig": components.train_config,
        "ModelConfig": components.model_config,
        "EnvConfig": components.env_config,
        "PersistenceConfig": components.persist_config,
        "MCTSConfig": components.mcts_config,  # This is trimcts.SearchConfiguration
        "StatsConfig": components.stats_config,
    }

    all_params = {}
    config_dict_for_json = {}

    for name, config_obj in configs_to_log.items():
        if isinstance(config_obj, BaseModel):
            # Use model_dump for Pydantic models
            params = config_obj.model_dump()
            config_dict_for_json[name] = params
        elif hasattr(config_obj, "__dict__"):
            # Fallback for non-Pydantic objects (like trimcts.SearchConfiguration)
            params = config_obj.__dict__
            config_dict_for_json[name] = params
        else:
            logger.warning(
                f"Cannot serialize config '{name}' of type {type(config_obj)}"
            )
            params = {"value": str(config_obj)}  # Log string representation
            config_dict_for_json[name] = {"value": str(config_obj)}

        # Flatten nested dicts for MLflow params (optional)
        flat_params = {}
        for key, value in params.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_params[f"{name}.{key}.{sub_key}"] = sub_value
            else:
                flat_params[f"{name}.{key}"] = value
        all_params.update(flat_params)

    # Log flattened parameters to MLflow
    try:
        # Log parameters individually, handling potential length limits
        MAX_PARAM_VAL_LENGTH = 250  # MLflow limit
        for key, value in all_params.items():
            str_value = str(value)
            if len(str_value) > MAX_PARAM_VAL_LENGTH:
                str_value = str_value[:MAX_PARAM_VAL_LENGTH] + "..."
            mlflow.log_param(key, str_value)
        logger.info(f"Logged {len(all_params)} parameters to MLflow.")
    except Exception as e:
        logger.error(f"Failed to log parameters to MLflow: {e}", exc_info=True)

    # Save the structured config dictionary as a JSON artifact
    try:
        components.data_manager.save_run_config(config_dict_for_json)
    except Exception as e:
        logger.error(f"Failed to save config JSON artifact: {e}", exc_info=True)
