import logging
from typing import Any

from pydantic import BaseModel, ValidationError

# Import EnvConfig from trianglengin
# REMOVE DisplayConfig import
from trianglengin.config import EnvConfig

from .mcts_config import MCTSConfig
from .model_config import ModelConfig
from .persistence_config import PersistenceConfig
from .train_config import TrainConfig

logger = logging.getLogger(__name__)


def print_config_info_and_validate(mcts_config_instance: MCTSConfig | None):
    """Prints configuration summary and performs validation using Pydantic."""
    print("-" * 40)
    print("Configuration Validation & Summary")
    print("-" * 40)
    all_valid = True
    configs_validated: dict[str, Any] = {}

    config_classes: dict[str, type[BaseModel]] = {
        "Environment": EnvConfig,  # Uses trianglengin.EnvConfig
        "Model": ModelConfig,
        "Training": TrainConfig,
        # REMOVE DisplayConfig
        # "Display": DisplayConfig,
        "Persistence": PersistenceConfig,
        "MCTS": MCTSConfig,
    }

    for name, ConfigClass in config_classes.items():
        instance: BaseModel | None = None
        try:
            if name == "MCTS":
                if mcts_config_instance is not None:
                    # Validate the provided instance against the class definition
                    instance = MCTSConfig.model_validate(
                        mcts_config_instance.model_dump()
                    )
                    print(f"[{name}] - Instance provided & validated OK")
                else:
                    # Instantiate default if no instance provided
                    instance = ConfigClass()
                    print(f"[{name}] - Validated OK (Instantiated Default)")
            else:
                # Instantiate default for other configs
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
            # Use model_dump for Pydantic v2
            dump_data = instance.model_dump()
            for field_name, value in dump_data.items():
                # Simple representation for long lists/dicts
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
