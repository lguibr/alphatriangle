# File: alphatriangle/config/stats_config.py
import logging
from typing import Literal

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# --- Enums and Literals ---
AggregationMethod = Literal[
    "latest", "mean", "sum", "rate", "min", "max", "std", "count"
]
LogTarget = Literal["mlflow", "tensorboard", "console", "stats_actor"]
DataSource = Literal["trainer", "worker", "loop", "buffer", "system"]
XAxis = Literal["global_step", "wall_time", "episode"]


# --- Metric Definition ---
class MetricConfig(BaseModel):
    """Configuration for a single metric to be tracked and logged."""

    name: str = Field(
        ..., description="Unique name for the metric (e.g., 'Loss/Total')"
    )
    source: DataSource = Field(
        ..., description="Origin of the raw metric data (e.g., 'trainer', 'worker')"
    )
    raw_event_name: str | None = Field(
        default=None,
        description="Specific raw event name if different from metric name (e.g., 'episode_end')",
    )
    aggregation: AggregationMethod = Field(
        default="latest",
        description="How to aggregate raw values over the logging interval ('rate' calculates per second)",
    )
    log_frequency_steps: int = Field(
        default=1,
        description="Log metric every N global steps. Set to 0 to disable step-based logging.",
        ge=0,
    )
    log_frequency_seconds: float = Field(
        default=0.0,
        description="Log metric every N seconds. Set to 0 to disable time-based logging.",
        ge=0.0,
    )
    log_to: list[LogTarget] = Field(
        default=["mlflow", "tensorboard"],
        description="Where to log the processed metric.",
    )
    x_axis: XAxis = Field(
        default="global_step", description="The primary x-axis for logging."
    )
    # Optional: For rate calculation, specify the numerator event
    rate_numerator_event: str | None = Field(
        default=None,
        description="Raw event name for the numerator in rate calculation (e.g., 'step_completed')",
    )

    @field_validator("log_frequency_steps", "log_frequency_seconds")
    @classmethod
    def check_log_frequency(cls, v):  # Removed info argument
        """Ensure at least one frequency is set if logging is enabled."""
        # This validation is tricky as it depends on other fields.
        # We'll rely on the processor logic to handle cases where both are 0.
        return v

    @field_validator("rate_numerator_event")
    @classmethod
    def check_rate_config(cls, v, info):
        """Ensure numerator is specified if aggregation is 'rate'."""
        # info.data might not be available in Pydantic v2 validators
        # Access values via info.values if needed, but direct access is preferred
        if info.data.get("aggregation") == "rate" and v is None:
            metric_name = info.data.get("name", "Unknown Metric")
            raise ValueError(
                f"Metric '{metric_name}' has aggregation 'rate' but 'rate_numerator_event' is not set."
            )
        return v

    @property
    def event_key(self) -> str:
        """The key used to store/retrieve raw events for this metric."""
        return self.raw_event_name or self.name


# --- Main Stats Configuration ---
class StatsConfig(BaseModel):
    """Overall configuration for statistics collection and logging."""

    # Interval (in seconds) for the StatsProcessor to process collected data
    processing_interval_seconds: float = Field(
        default=2.0,
        description="How often the StatsProcessor aggregates and logs metrics.",
        gt=0,
    )

    # Default metrics - can be overridden or extended
    metrics: list[MetricConfig] = Field(
        default_factory=lambda: [
            # --- Trainer Metrics ---
            MetricConfig(
                name="Loss/Total",
                source="trainer",
                aggregation="mean",
                log_frequency_steps=10,
                log_to=["mlflow", "tensorboard"],
            ),
            MetricConfig(
                name="Loss/Policy",
                source="trainer",
                aggregation="mean",
                log_frequency_steps=10,
                log_to=["mlflow", "tensorboard"],
            ),
            MetricConfig(
                name="Loss/Value",
                source="trainer",
                aggregation="mean",
                log_frequency_steps=10,
                log_to=["mlflow", "tensorboard"],
            ),
            MetricConfig(
                name="Loss/Entropy",
                source="trainer",
                aggregation="mean",
                log_frequency_steps=10,
                log_to=["mlflow", "tensorboard"],
            ),
            MetricConfig(
                name="Loss/Mean_TD_Error",
                source="trainer",
                aggregation="mean",
                log_frequency_steps=10,
                log_to=["mlflow", "tensorboard"],
            ),
            MetricConfig(
                name="LearningRate",
                source="trainer",
                aggregation="latest",
                log_frequency_steps=10,
                log_to=["mlflow", "tensorboard"],
            ),
            # --- Worker Metrics (Aggregated) ---
            MetricConfig(
                name="Episode/Final_Score",
                source="worker",
                raw_event_name="episode_end",  # Use context['score']
                aggregation="mean",
                log_frequency_steps=1,  # Log every step where an episode ended
                log_to=["mlflow", "tensorboard"],
            ),
            MetricConfig(
                name="Episode/Length",
                source="worker",
                raw_event_name="episode_end",  # Use context['length']
                aggregation="mean",
                log_frequency_steps=1,
                log_to=["mlflow", "tensorboard"],
            ),
            MetricConfig(
                name="MCTS/Avg_Simulations",
                source="worker",
                raw_event_name="mcts_step",  # Use value from mcts_step event
                aggregation="mean",
                log_frequency_steps=50,
                log_to=["mlflow", "tensorboard"],
            ),
            MetricConfig(
                name="RL/Step_Reward_Mean",
                source="worker",
                raw_event_name="step_reward",  # Use value from step_reward event
                aggregation="mean",
                log_frequency_steps=50,
                log_to=["mlflow", "tensorboard"],
            ),
            # --- Loop/System Metrics ---
            MetricConfig(
                name="Buffer/Size",
                source="loop",
                aggregation="latest",
                log_frequency_steps=10,
                log_to=["mlflow", "tensorboard"],
            ),
            MetricConfig(
                name="Progress/Total_Simulations",
                source="loop",
                aggregation="latest",
                log_frequency_steps=10,
                log_to=["mlflow", "tensorboard"],
            ),
            MetricConfig(
                name="Progress/Episodes_Played",
                source="loop",
                aggregation="latest",
                log_frequency_steps=10,
                log_to=["mlflow", "tensorboard"],
            ),
            MetricConfig(
                name="System/Num_Active_Workers",
                source="loop",
                aggregation="latest",
                log_frequency_steps=50,
                log_to=["mlflow", "tensorboard"],
            ),
            MetricConfig(
                name="System/Num_Pending_Tasks",
                source="loop",
                aggregation="latest",
                log_frequency_steps=50,
                log_to=["mlflow", "tensorboard"],
            ),
            # --- Rate Metrics ---
            MetricConfig(
                name="Rate/Steps_Per_Sec",
                source="loop",
                aggregation="rate",
                rate_numerator_event="step_completed",  # Assumes loop logs this event
                log_frequency_seconds=5.0,
                log_frequency_steps=0,  # Primarily time-based
                log_to=["mlflow", "tensorboard", "console"],
            ),
            MetricConfig(
                name="Rate/Episodes_Per_Sec",
                source="worker",
                aggregation="rate",
                rate_numerator_event="episode_end",  # Count episode_end events
                log_frequency_seconds=5.0,
                log_frequency_steps=0,
                log_to=["mlflow", "tensorboard", "console"],
            ),
            MetricConfig(
                name="Rate/Simulations_Per_Sec",
                source="worker",  # Sims happen in worker MCTS steps
                aggregation="rate",
                rate_numerator_event="mcts_step",  # Sum the 'value' of mcts_step events
                log_frequency_seconds=5.0,
                log_frequency_steps=0,
                log_to=["mlflow", "tensorboard", "console"],
            ),
            # --- PER Beta ---
            MetricConfig(
                name="PER/Beta",
                source="buffer",  # Or loop, depending on where it's calculated
                aggregation="latest",
                log_frequency_steps=10,
                log_to=["mlflow", "tensorboard"],
            ),
            # --- Event Markers ---
            MetricConfig(
                name="Event/Weight_Update",  # Matches old key for compatibility
                source="loop",
                aggregation="count",  # Count occurrences per interval
                log_frequency_steps=10,  # Log counts every 10 steps
                log_to=["mlflow", "tensorboard"],  # Log count, not just marker
            ),
        ]
    )

    @field_validator("metrics")
    @classmethod
    def check_metric_names_unique(cls, metrics: list[MetricConfig]):
        """Ensure all configured metric names are unique."""
        names = [m.name for m in metrics]
        if len(names) != len(set(names)):
            from collections import Counter

            duplicates = [name for name, count in Counter(names).items() if count > 1]
            raise ValueError(f"Duplicate metric names found in config: {duplicates}")
        return metrics


# Instantiate default config
default_stats_config = StatsConfig()

# Rebuild model
StatsConfig.model_rebuild(force=True)
MetricConfig.model_rebuild(force=True)
