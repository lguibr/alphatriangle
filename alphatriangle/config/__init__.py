# File: alphatriangle/config/__init__.py
# Import EnvConfig from trianglengin
from trianglengin.config import EnvConfig

from .app_config import APP_NAME
from .mcts_config import MCTSConfig
from .model_config import ModelConfig
from .persistence_config import PersistenceConfig
from .train_config import TrainConfig
from .validation import print_config_info_and_validate

# REMOVE VisConfig import

__all__ = [
    "APP_NAME",
    "EnvConfig",  # Now imported from trianglengin
    "ModelConfig",
    "PersistenceConfig",
    "TrainConfig",
    # "VisConfig", # REMOVED
    "MCTSConfig",
    "print_config_info_and_validate",
]
