"""
Core RL components: Trainer, Buffer.
The Orchestrator logic has been moved to the alphatriangle.training module.
"""

from .buffer import ExperienceBuffer
from .trainer import Trainer

__all__ = [
    "Trainer",
    "ExperienceBuffer",
]
