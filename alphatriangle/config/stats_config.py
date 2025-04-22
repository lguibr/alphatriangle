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
    # Optional: For metrics derived from context dicts (like episode_end)
    context_key: str | None = Field(
        default=None,
        description="Key within the RawMetricEvent context dictionary to extract the value from.",
    )

    @field_validator("log_frequency_steps", "log_frequency_seconds")
    @classmethod
    def check_log_frequency(cls, v):
        """Ensure at least one frequency is set if logging is enabled."""
        # This validation is tricky as it depends on other fields.
        # We'll rely on the processor logic to handle cases where both are 0.
        return v

    @field_validator("rate_numerator_event")
    @classmethod
    def check_rate_config(cls, v, info):
        """Ensure numerator is specified if aggregation is 'rate'."""
        if info.data.get("aggregation") == "rate" and v is None:
            metric_name = info.data.get("name", "Unknown Metric")
            raise ValueError(
                f"Metric '{metric_name}' has aggregation 'rate' but 'rate_numerator_event' is not set."
            )
        return v

    @field_validator("context_key")
    @classmethod
    def check_context_key_config(cls, v, info):
        """Ensure context_key is set only when appropriate (e.g., for episode_end derived metrics)."""
        raw_event = info.data.get("raw_event_name")
        if v is not None and raw_event is None:
            metric_name = info.data.get("name", "Unknown Metric")
            logger.warning(
                f"Metric '{metric_name}' has 'context_key' set ('{v}') but no 'raw_event_name'. "
                "This might not work as expected unless the event name itself is used as context source."
            )
        # Add more specific checks if needed, e.g., ensure context_key is set for specific raw_event_names
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
        default=1.0,
        description="How often the StatsProcessor aggregates and logs metrics.",
        gt=0,
    )

    # Default metrics - can be overridden or extended
    metrics: list[MetricConfig] = Field(
        default_factory=lambda: [
            # --- Trainer Metrics (Log based on steps) ---
            MetricConfig(
                name="Loss/Total",
                source="trainer",
                aggregation="mean",
                log_frequency_steps=10,  # Log every 10 training steps
                log_frequency_seconds=0,
                log_to=["mlflow", "tensorboard"],
            ),
            MetricConfig(
                name="Loss/Policy",
                source="trainer",
                aggregation="mean",
                log_frequency_steps=10,
                log_frequency_seconds=0,
                log_to=["mlflow", "tensorboard"],
            ),
            MetricConfig(
                name="Loss/Value",
                source="trainer",
                aggregation="mean",
                log_frequency_steps=10,
                log_frequency_seconds=0,
                log_to=["mlflow", "tensorboard"],
            ),
            MetricConfig(
                name="Loss/Entropy",
                source="trainer",
                aggregation="mean",
                log_frequency_steps=10,
                log_frequency_seconds=0,
                log_to=["mlflow", "tensorboard"],
            ),
            MetricConfig(
                name="Loss/Mean_Abs_TD_Error",
                source="trainer",
                aggregation="mean",
                log_frequency_steps=10,
                log_frequency_seconds=0,
                log_to=["mlflow", "tensorboard"],
            ),
            MetricConfig(
                name="LearningRate",
                source="trainer",
                aggregation="latest",
                log_frequency_steps=10,
                log_frequency_seconds=0,
                log_to=["mlflow", "tensorboard"],
            ),
            # --- Worker Metrics (Log based on time) ---
            MetricConfig(
                name="Episode/Final_Score",
                source="worker",
                raw_event_name="episode_end",
                context_key="score",
                aggregation="mean",
                log_frequency_steps=0,  # Log based on time
                log_frequency_seconds=5.0,
                log_to=["mlflow", "tensorboard"],
            ),
            MetricConfig(
                name="Episode/Length",
                source="worker",
                raw_event_name="episode_end",
                context_key="length",
                aggregation="mean",
                log_frequency_steps=0,  # Log based on time
                log_frequency_seconds=5.0,
                log_to=["mlflow", "tensorboard"],
            ),
            MetricConfig(
                name="Episode/Triangles_Cleared_Total",
                source="worker",
                raw_event_name="episode_end",
                context_key="triangles_cleared",
                aggregation="mean",
                log_frequency_steps=0,  # Log based on time
                log_frequency_seconds=5.0,
                log_to=["mlflow", "tensorboard"],
            ),
            MetricConfig(
                name="MCTS/Avg_Simulations_Per_Step",
                source="worker",
                raw_event_name="mcts_step",
                aggregation="mean",
                log_frequency_steps=0,  # Log based on time
                log_frequency_seconds=10.0,
                log_to=["mlflow", "tensorboard"],
            ),
            MetricConfig(
                name="RL/Step_Reward_Mean",
                source="worker",
                raw_event_name="step_reward",
                aggregation="mean",
                log_frequency_steps=0,  # Log based on time
                log_frequency_seconds=10.0,
                log_to=["mlflow", "tensorboard"],
            ),
            # --- Loop/System Metrics (Log based on time) ---
            MetricConfig(
                name="Buffer/Size",
                source="loop",
                aggregation="latest",
                log_frequency_steps=0,  # Log based on time
                log_frequency_seconds=5.0,
                log_to=["mlflow", "tensorboard"],
            ),
            MetricConfig(
                name="Progress/Total_Simulations",
                source="loop",
                aggregation="latest",
                log_frequency_steps=0,  # Log based on time
                log_frequency_seconds=5.0,
                log_to=["mlflow", "tensorboard"],
            ),
            MetricConfig(
                name="Progress/Episodes_Played",
                source="loop",
                aggregation="latest",
                log_frequency_steps=0,  # Log based on time
                log_frequency_seconds=5.0,
                log_to=["mlflow", "tensorboard"],
            ),
            MetricConfig(
                name="Progress/Weight_Updates_Total",
                source="loop",
                aggregation="latest",
                log_frequency_steps=0,  # Log based on time
                log_frequency_seconds=5.0,
                log_to=["mlflow", "tensorboard"],
            ),
            MetricConfig(
                name="System/Num_Active_Workers",
                source="loop",
                aggregation="latest",
                log_frequency_steps=0,  # Log based on time
                log_frequency_seconds=10.0,
                log_to=["mlflow", "tensorboard"],
            ),
            MetricConfig(
                name="System/Num_Pending_Tasks",
                source="loop",
                aggregation="latest",
                log_frequency_steps=0,  # Log based on time
                log_frequency_seconds=10.0,
                log_to=["mlflow", "tensorboard"],
            ),
            # --- Rate Metrics (Log based on time) ---
            MetricConfig(
                name="Rate/Steps_Per_Sec",
                source="loop",
                aggregation="rate",
                rate_numerator_event="step_completed",
                log_frequency_seconds=5.0,
                log_frequency_steps=0,
                log_to=["mlflow", "tensorboard", "console"],
            ),
            MetricConfig(
                name="Rate/Episodes_Per_Sec",
                source="worker",
                aggregation="rate",
                rate_numerator_event="episode_end",
                log_frequency_seconds=5.0,
                log_frequency_steps=0,
                log_to=["mlflow", "tensorboard", "console"],
            ),
            MetricConfig(
                name="Rate/Simulations_Per_Sec",
                source="worker",
                aggregation="rate",
                rate_numerator_event="mcts_step",
                log_frequency_seconds=5.0,
                log_frequency_steps=0,
                log_to=["mlflow", "tensorboard", "console"],
            ),
            # --- PER Beta (Log based on steps, as it changes with training step) ---
            MetricConfig(
                name="PER/Beta",
                source="buffer",
                aggregation="latest",
                log_frequency_steps=10,
                log_frequency_seconds=0,
                log_to=["mlflow", "tensorboard"],
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
