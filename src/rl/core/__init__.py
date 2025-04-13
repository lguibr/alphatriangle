# File: src/rl/core/__init__.py
"""
Core RL components: Trainer, Buffer.
The Orchestrator logic has been moved to the src.training module.
"""

# Import the final classes intended for export from their respective modules.
from .buffer import ExperienceBuffer
from .trainer import Trainer

# Removed: from .orchestrator import TrainingOrchestrator
# Removed: from .visual_state_actor import VisualStateActor

__all__ = [
    "Trainer",
    "ExperienceBuffer",
]
