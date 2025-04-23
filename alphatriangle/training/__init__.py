# File: alphatriangle/training/__init__.py
from .components import TrainingComponents

# Utilities
from .logging_utils import log_configs_to_mlflow
from .loop import TrainingLoop
from .loop_helpers import LoopHelpers

# Re-export runner functions
from .runner import run_training

# REMOVE visual runner import
from .setup import setup_training_components
from .worker_manager import WorkerManager

# Explicitly define __all__
__all__ = [
    # Core Components
    "TrainingComponents",
    "TrainingLoop",
    # Helpers & Managers
    "WorkerManager",
    "LoopHelpers",
    "setup_training_components",
    # Runners (re-exported)
    "run_training",
    # Logging Utilities
    "log_configs_to_mlflow",
]
