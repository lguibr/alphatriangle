"""
Statistics collection module.
"""

# Import StatsCollectorData from utils where it should reside
from alphatriangle.utils.types import StatsCollectorData

from .collector import StatsCollectorActor

# REMOVE Plotter, PlotDefinitions, PlotType, render_subplot, plot_utils

__all__ = [
    # Core Collector
    "StatsCollectorActor",
    "StatsCollectorData",  # Export type alias
]
