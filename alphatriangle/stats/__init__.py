# File: alphatriangle/stats/__init__.py
"""
Statistics collection and plotting module.
"""

from alphatriangle.utils.types import StatsCollectorData

from . import plot_utils
from .collector import StatsCollectorActor
from .plot_definitions import PlotDefinitions, PlotType  # Import new definitions
from .plot_rendering import render_subplot  # Import new rendering function
from .plotter import Plotter

__all__ = [
    # Core Collector
    "StatsCollectorActor",
    "StatsCollectorData",
    # Plotting Orchestrator
    "Plotter",
    # Plotting Definitions & Rendering Logic
    "PlotDefinitions",
    "PlotType",
    "render_subplot",
    # Plotting Utilities
    "plot_utils",
]
