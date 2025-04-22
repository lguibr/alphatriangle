# File: alphatriangle/stats/processor.py
import logging
from collections import defaultdict
from typing import TYPE_CHECKING  # Import cast

import mlflow
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from .types import LogContext, MetricConfig

if TYPE_CHECKING:
    from ..config import StatsConfig

logger = logging.getLogger(__name__)


class StatsProcessor:
    """
    Processes raw metric data collected by the StatsCollectorActor and logs
    aggregated results according to the StatsConfig.
    """

    def __init__(
        self,
        config: "StatsConfig",
        run_name: str,
        tb_writer: SummaryWriter | None,
    ):
        self.config = config
        self.run_name = run_name
        self.tb_writer = tb_writer
        self._metric_configs: dict[str, MetricConfig] = {
            mc.name: mc for mc in config.metrics
        }
        # State for rate calculation
        self._last_rate_log_time: dict[str, float] = {}
        self._last_rate_log_step: dict[str, int] = {}
        self._last_rate_numerator_count: dict[str, int] = {}

        logger.info("StatsProcessor initialized.")

    def _aggregate_values(self, values: list[float], method: str) -> float | int | None:
        """Aggregates a list of values based on the specified method."""
        if not values:
            return None
        try:
            if method == "latest":
                return values[-1]
            elif method == "mean":
                # Cast numpy float to standard float
                return float(np.mean(values))
            elif method == "sum":
                # Cast numpy result to standard float/int
                result = np.sum(values)
                # Use np.issubdtype to check for integer types
                return (
                    int(result)
                    if np.issubdtype(result.dtype, np.integer)
                    else float(result)
                )
            elif method == "min":
                # Cast numpy result to standard float/int
                result = np.min(values)
                return (
                    int(result)
                    if np.issubdtype(result.dtype, np.integer)
                    else float(result)
                )
            elif method == "max":
                # Cast numpy result to standard float/int
                result = np.max(values)
                return (
                    int(result)
                    if np.issubdtype(result.dtype, np.integer)
                    else float(result)
                )
            elif method == "std":
                # Cast numpy float to standard float
                return float(np.std(values))
            elif method == "count":
                return len(values)
            else:
                logger.warning(f"Unsupported aggregation method: {method}")
                return None
        except Exception as e:
            logger.error(f"Error during aggregation '{method}': {e}")
            return None

    def _calculate_rate(
        self,
        metric_config: MetricConfig,
        context: LogContext,
        rate_numerators_in_batch: dict[str, float],  # Pass pre-calculated numerators
    ) -> float | None:
        """Calculates rate per second for a given metric using pre-calculated numerators."""
        if (
            metric_config.aggregation != "rate"
            or not metric_config.rate_numerator_event
        ):
            return None

        event_key: str | None = (
            metric_config.rate_numerator_event
        )  # Assign to typed var
        # Check event_key is not None before using it as dict key
        if event_key is None:
            logger.warning(
                f"Rate metric '{metric_config.name}' is missing rate_numerator_event."
            )
            return None

        current_time = context.current_time
        # Use last_log_time from context as the start of the interval
        last_log_time = context.last_log_time
        time_delta = current_time - last_log_time

        if time_delta <= 1e-3:  # Avoid division by zero or tiny intervals
            return None  # Not enough time passed

        # Use the pre-calculated numerator sum/count for this batch
        numerator_in_batch = rate_numerators_in_batch.get(event_key, 0.0)
        rate = numerator_in_batch / time_delta

        # No need to update internal state here, as it's based on the processing interval

        return rate

    def _should_log(
        self,
        metric_config: MetricConfig,
        global_step: int,
        context: LogContext,
    ) -> bool:
        """Determines if a metric should be logged at the current step/time."""
        log_by_step = (
            metric_config.log_frequency_steps > 0
            and global_step % metric_config.log_frequency_steps == 0
        )

        # Check time-based logging using the last log time specific to this metric if available
        last_metric_log_time = self._last_rate_log_time.get(
            metric_config.name, 0.0
        )  # Use rate log time dict
        log_by_time = (
            metric_config.log_frequency_seconds > 0
            and (context.current_time - last_metric_log_time)
            >= metric_config.log_frequency_seconds
        )

        # Special case for rates: only log if time frequency is met
        if metric_config.aggregation == "rate":
            # Rates are calculated over the interval, so log if the interval is met
            return log_by_time

        # For non-rate metrics, log if *either* step or time frequency is met
        return log_by_step or log_by_time

    def _log_to_targets(
        self,
        metric_config: MetricConfig,
        value: float | int,
        log_step: int,
    ):
        """Logs the processed value to configured targets."""
        if not np.isfinite(value):
            logger.warning(
                f"Attempted to log non-finite value for {metric_config.name}: {value}. Skipping."
            )
            return

        log_value = float(value)  # Ensure float for logging consistency

        if "mlflow" in metric_config.log_to:
            try:
                mlflow.log_metric(metric_config.name, log_value, step=log_step)
            except Exception as e:
                logger.error(f"Failed to log {metric_config.name} to MLflow: {e}")

        if "tensorboard" in metric_config.log_to and self.tb_writer:
            try:
                self.tb_writer.add_scalar(metric_config.name, log_value, log_step)
            except Exception as e:
                logger.error(f"Failed to log {metric_config.name} to TensorBoard: {e}")

        if "console" in metric_config.log_to:
            # Basic console logging, could be enhanced
            logger.info(f"STATS [{log_step}]: {metric_config.name} = {log_value:.4f}")

    def _process_and_log_internal(
        self,
        raw_data: dict[int, dict[str, list[float]]],
        context: LogContext,
    ):
        """Internal logic for processing and logging, shared by public methods."""
        if not raw_data:
            return

        aggregated_values_for_logging: dict[str, list[tuple[int, float | int]]] = (
            defaultdict(list)
        )

        # Calculate total counts/sums for rate numerators across the processed steps *before* aggregation loop
        rate_numerators_in_batch: dict[str, float] = defaultdict(float)
        # Rename unused loop variable step to _step (Ruff B007 fix)
        for _step, step_data in raw_data.items():
            for metric_config_inner_loop in self.config.metrics:  # Use different name
                if (
                    metric_config_inner_loop.aggregation == "rate"
                    and metric_config_inner_loop.rate_numerator_event
                ):
                    event_key: str | None = (
                        metric_config_inner_loop.rate_numerator_event
                    )  # Assign to typed var
                    # Add check: event_key should not be None here due to validator
                    if event_key and event_key in step_data:
                        if event_key == "mcts_step":  # Sum values for simulations
                            rate_numerators_in_batch[event_key] += sum(
                                step_data[event_key]
                            )
                        else:  # Count events otherwise
                            rate_numerators_in_batch[event_key] += len(
                                step_data[event_key]
                            )

        # --- First loop: Aggregate step-based metrics ---
        for step, step_data in sorted(raw_data.items()):
            for metric_config in self.config.metrics:  # Outer loop variable
                event_key = metric_config.event_key
                if event_key in step_data:
                    raw_values = step_data[event_key]
                    # Handle episode score/length extraction from context if needed
                    # This part needs refinement based on how context is passed in episode_end event
                    if (
                        event_key == "episode_end"
                        and metric_config.name == "Episode/Final_Score"
                    ):
                        # Example: Assuming score is the *first* value if multiple episodes end at same step
                        # This needs a better way, maybe log score as separate event or process context
                        agg_value = self._aggregate_values(
                            [raw_values[0]], "latest"
                        )  # Placeholder
                    elif (
                        event_key == "episode_end"
                        and metric_config.name == "Episode/Length"
                    ):
                        agg_value = self._aggregate_values(
                            [raw_values[0]], "latest"
                        )  # Placeholder
                    elif (
                        metric_config.aggregation != "rate"
                    ):  # Rates handled separately below
                        agg_value = self._aggregate_values(
                            raw_values, metric_config.aggregation
                        )
                    else:
                        agg_value = None  # Skip aggregation for rates here

                    if agg_value is not None:
                        aggregated_values_for_logging[metric_config.name].append(
                            (step, agg_value)
                        )

        # Log aggregated step-based metrics
        for metric_name, step_value_list in aggregated_values_for_logging.items():
            # Add check for metric_config existence (MyPy fix)
            metric_config_maybe: MetricConfig | None = self._metric_configs.get(
                metric_name
            )
            if not metric_config_maybe:
                logger.warning(
                    f"Metric config not found for '{metric_name}' during logging."
                )
                continue
            # Use a different name for the variable holding the specific config for this metric
            current_metric_config: MetricConfig = (
                metric_config_maybe  # Assign to non-optional type
            )

            # Log each step's aggregated value if frequency matches
            for step, agg_value in step_value_list:
                # Use _should_log which checks step frequency for non-rate metrics
                if self._should_log(current_metric_config, step, context):
                    log_step = step  # Use the actual step the data corresponds to
                    self._log_to_targets(current_metric_config, agg_value, log_step)
                    # Update last log time for time-based frequency check ONLY if logged by time
                    if (
                        current_metric_config.log_frequency_seconds > 0
                        and (
                            context.current_time
                            - self._last_rate_log_time.get(
                                current_metric_config.name, 0.0
                            )
                        )
                        >= current_metric_config.log_frequency_seconds
                    ):
                        self._last_rate_log_time[current_metric_config.name] = (
                            context.current_time
                        )

        # --- Second loop: Calculate and log rate metrics ---
        # Use a different loop variable name here to fix MyPy error
        for rate_metric_config in self.config.metrics:
            # Use _should_log which checks time frequency for rate metrics
            if self._should_log(rate_metric_config, context.latest_step, context):
                rate_value = self._calculate_rate(
                    rate_metric_config, context, rate_numerators_in_batch
                )
                if rate_value is not None:
                    # Log rate against the latest global step
                    self._log_to_targets(
                        rate_metric_config, rate_value, context.latest_step
                    )
                    # Update last log time for this metric
                    self._last_rate_log_time[rate_metric_config.name] = (
                        context.current_time
                    )

    def process_and_log(
        self,
        raw_data: dict[int, dict[str, list[float]]],
        context: LogContext,
    ):
        """
        Public method to process and log, called by the actor.
        Delegates to the internal processing logic.
        """
        self._process_and_log_internal(raw_data, context)
