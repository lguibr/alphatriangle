# File: alphatriangle/stats/__init__.py
"""
Statistics collection module. Handles asynchronous collection of raw metrics
and processes/logs them according to configuration.
"""

# Use full path import for mypy compatibility with Ray actors
from alphatriangle.stats.collector import StatsCollectorActor

from .processor import StatsProcessor

# Update import to use the new filename
from .stats_types import LogContext, MetricConfig, RawMetricEvent, StatsConfig

__all__ = [
    # Core Collector Actor
    "StatsCollectorActor",
    # Processor Logic (might be internal to actor)
    "StatsProcessor",
    # Configuration & Types
    "StatsConfig",
    "MetricConfig",
    "RawMetricEvent",
    "LogContext",
]
