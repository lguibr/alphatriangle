# File: alphatriangle/training/runners.py
"""
Entry points for running training modes.
Imports functions from specific runner modules.
"""

# Import from the renamed runner
from .runner import run_training  # Rename export

__all__ = [
    "run_training",  # Export the single runner function
]
