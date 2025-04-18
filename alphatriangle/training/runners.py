# File: alphatriangle/training/runners.py
"""
Entry points for running training modes.
Imports functions from specific runner modules.
"""

from .headless_runner import run_training_headless_mode
from .visual_runner import run_training_visual_mode

__all__ = [
    "run_training_headless_mode",
    "run_training_visual_mode",
]
