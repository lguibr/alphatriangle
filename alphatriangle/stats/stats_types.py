# File: alphatriangle/stats/types.py
from typing import Any  # Import cast

import numpy as np
from pydantic import BaseModel, Field

# Re-export relevant config types for convenience
from ..config.stats_config import (
    AggregationMethod,
    DataSource,
    LogTarget,
    MetricConfig,
    StatsConfig,
    XAxis,
)


class RawMetricEvent(BaseModel):
    """Structure for raw metric data points sent to the collector."""

    name: str = Field(
        ...,
        description="Identifier for the raw event (e.g., 'loss/policy', 'episode_end')",
    )
    value: float | int = Field(..., description="The numerical value of the event.")
    global_step: int = Field(
        ..., description="The training step associated with this event."
    )
    timestamp: float | None = Field(
        default=None, description="Optional timestamp of the event occurrence."
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional additional context (e.g., worker_id, score, length).",
    )

    def is_valid(self) -> bool:
        """Checks if the value is finite."""
        # Cast numpy bool_ to standard bool
        return bool(np.isfinite(self.value))


class LogContext(BaseModel):
    """Context information passed to the StatsProcessor during logging."""

    latest_step: int
    last_log_time: float  # Timestamp of the last time processor ran
    current_time: float  # Timestamp of the current processor run
    event_timestamps: dict[
        str, list[tuple[float, int]]
    ]  # event_name -> list of (timestamp, global_step)
    latest_values: dict[str, tuple[int, float]]  # event_name -> (global_step, value)


__all__ = [
    "StatsConfig",
    "MetricConfig",
    "AggregationMethod",
    "LogTarget",
    "DataSource",
    "XAxis",
    "RawMetricEvent",
    "LogContext",
]
