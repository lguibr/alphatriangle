# File: alphatriangle/training/__init__.py
"""
Training module containing the pipeline, loop, components, and utilities
for orchestrating the reinforcement learning training process.
"""

# Core components & classes
from .components import TrainingComponents

# Utilities
from .logging_utils import Tee, get_root_logger, setup_file_logging
from .loop import TrainingLoop
from .loop_helpers import LoopHelpers

# Re-export runner functions
from .runners import (
    run_training_headless_mode,
    run_training_visual_mode,
)
from .setup import setup_training_components

# from .pipeline import TrainingPipeline # REMOVED
from .worker_manager import WorkerManager

# Explicitly define __all__
__all__ = [
    # Core Components
    "TrainingComponents",
    "TrainingLoop",
    # "TrainingPipeline", # REMOVED
    # Helpers & Managers
    "WorkerManager",
    "LoopHelpers",
    "setup_training_components",
    # Runners (re-exported)
    "run_training_headless_mode",
    "run_training_visual_mode",
    # Logging Utilities
    "setup_file_logging",
    "get_root_logger",
    "Tee",
]
