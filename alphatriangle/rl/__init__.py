# File: src/rl/__init__.py
"""
Reinforcement Learning (RL) module.
Contains the core components for training an agent using self-play and MCTS.
"""

# Core RL classes (Trainer, Buffer are still relevant)
from .core.buffer import ExperienceBuffer
from .core.trainer import Trainer

# Self-play functionality (Ray actor)
from .self_play.worker import SelfPlayWorker
from .types import SelfPlayResult  # Import Pydantic model

# Note: TrainingOrchestrator is removed, its functionality is now in src.training
# from .core.orchestrator import TrainingOrchestrator

__all__ = [
    # Core components used by the training pipeline
    "Trainer",
    "ExperienceBuffer",
    # Self-play components
    "SelfPlayWorker",
    "SelfPlayResult",
]
