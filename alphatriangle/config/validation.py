import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from .env_config import EnvConfig
from .mcts_config import MCTSConfig
from .model_config import ModelConfig
from .persistence_config import PersistenceConfig
from .train_config import TrainConfig
from .vis_config import VisConfig

logger = logging.getLogger(__name__)


def print_config_info_and_validate(mcts_config_instance: MCTSConfig | None):
    """Prints configuration summary and performs validation using Pydantic."""
    print("-" * 40)
    print("Configuration Validation & Summary")
    print("-" * 40)
    all_valid = True
    configs_validated: dict[str, Any] = {}

    config_classes: dict[str, type[BaseModel]] = {
        "Environment": EnvConfig,
        "Model": ModelConfig,
        "Training": TrainConfig,
        "Visualization": VisConfig,
        "Persistence": PersistenceConfig,
        "MCTS": MCTSConfig,
    }

    for name, ConfigClass in config_classes.items():
        instance: BaseModel | None = None
        try:
            if name == "MCTS":
                if mcts_config_instance is not None:
                    instance = MCTSConfig.model_validate(
                        mcts_config_instance.model_dump()
                    )
                    print(f"[{name}] - Instance provided & validated OK")
                else:
                    instance = ConfigClass()
                    print(f"[{name}] - Validated OK (Instantiated Default)")
            else:
                instance = ConfigClass()
                print(f"[{name}] - Validated OK")
            configs_validated[name] = instance
        except ValidationError as e:
            logger.error(f"Validation failed for {name} Config:")
            logger.error(e)
            all_valid = False
            configs_validated[name] = None
        except Exception as e:
            logger.error(
                f"Unexpected error instantiating/validating {name} Config: {e}"
            )
            all_valid = False
            configs_validated[name] = None

    print("-" * 40)
    print("Configuration Values:")
    print("-" * 40)

    for name, instance in configs_validated.items():
        print(f"--- {name} Config ---")
        if instance:
            dump_data = instance.model_dump()
            for field_name, value in dump_data.items():
                if isinstance(value, list) and len(value) > 5:
                    print(f"  {field_name}: [List with {len(value)} items]")
                elif isinstance(value, dict) and len(value) > 5:
                    print(f"  {field_name}: {{Dict with {len(value)} keys}}")
                else:
                    print(f"  {field_name}: {value}")
        else:
            print("  <Validation Failed>")
        print("-" * 20)

    print("-" * 40)
    if not all_valid:
        logger.critical("Configuration validation failed. Please check errors above.")
        raise ValueError("Invalid configuration settings.")
    else:
        logger.info("All configurations validated successfully.")
    print("-" * 40)
