# File: alphatriangle/stats/processor.py
import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from mlflow.tracking import MlflowClient
from torch.utils.tensorboard import SummaryWriter

from .stats_types import LogContext, MetricConfig, RawMetricEvent

if TYPE_CHECKING:
    from ..config import StatsConfig

logger = logging.getLogger(__name__)


class StatsProcessor:
    """
    Processes raw metric data collected by the StatsCollectorActor and logs
    aggregated results according to the StatsConfig. Uses MlflowClient for robustness.
    Prioritizes correct logging over MLflow UI rendering quirks.
    """

    def __init__(
        self,
        config: "StatsConfig",
        run_name: str,
        tb_writer: SummaryWriter | None,
        mlflow_run_id: str | None,
    ):
        self.config = config
        self.run_name = run_name
        self.mlflow_run_id = mlflow_run_id
        self._metric_configs: dict[str, MetricConfig] = {
            mc.name: mc for mc in config.metrics
        }
        # Track last logged time *per metric* for time-based frequency checks
        self._last_log_time: dict[str, float] = {}

        self.tb_writer = tb_writer
        self.mlflow_client: MlflowClient | None = None

        if self.mlflow_run_id:
            try:
                self.mlflow_client = MlflowClient()
                logger.info("StatsProcessor: MlflowClient initialized.")
            except Exception as e:
                logger.error(f"StatsProcessor: Failed to initialize MlflowClient: {e}")
                self.mlflow_client = None
        else:
            logger.warning(
                "StatsProcessor: No MLflow run ID provided, MLflow metric logging will be disabled."
            )

        logger.info("StatsProcessor initialized.")

    def _aggregate_values(self, values: list[float], method: str) -> float | int | None:
        """Aggregates a list of values based on the specified method."""
        if not values:
            return None
        try:
            finite_values = [v for v in values if np.isfinite(v)]
            if not finite_values:
                # Log less verbosely if no finite values
                # logger.warning(f"No finite values found for aggregation method {method}.")
                return None

            if method == "latest":
                return finite_values[-1]
            elif method == "mean":
                return float(np.mean(finite_values))
            elif method == "sum":
                result = np.sum(finite_values)
                return (
                    int(result)
                    if np.issubdtype(result.dtype, np.integer)
                    else float(result)
                )
            elif method == "min":
                result = np.min(finite_values)
                return (
                    int(result)
                    if np.issubdtype(result.dtype, np.integer)
                    else float(result)
                )
            elif method == "max":
                result = np.max(finite_values)
                return (
                    int(result)
                    if np.issubdtype(result.dtype, np.integer)
                    else float(result)
                )
            elif method == "std":
                return float(np.std(finite_values))
            elif method == "count":
                return len(finite_values)
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
        raw_data_for_interval: dict[int, dict[str, list[RawMetricEvent]]],
    ) -> float | None:
        """Calculates rate per second for a given metric over the interval."""
        if (
            metric_config.aggregation != "rate"
            or not metric_config.rate_numerator_event
        ):
            return None

        event_key: str | None = metric_config.rate_numerator_event
        if event_key is None:
            logger.warning(
                f"Rate metric '{metric_config.name}' is missing rate_numerator_event."
            )
            return None

        current_time = context.current_time
        last_log_time = context.last_log_time  # Use the overall interval time
        time_delta = current_time - last_log_time

        if time_delta <= 1e-3:  # Avoid division by zero or tiny intervals
            return None

        # Sum the numerator counts *within this processing interval*
        numerator_count = 0.0
        for _step, step_data in raw_data_for_interval.items():
            if event_key in step_data:
                event_list = step_data[event_key]
                values = [float(e.value) for e in event_list if e.is_valid()]
                if event_key == "mcts_step":  # Sum simulations
                    numerator_count += sum(values)
                else:  # Count events for other rates
                    numerator_count += len(values)

        rate = numerator_count / time_delta
        return rate

    def _should_log(
        self,
        metric_config: MetricConfig,
        current_step: int,  # The step associated with the *aggregated value*
        context: LogContext,
    ) -> bool:
        """
        Determines if a metric should be logged based on simple step OR time frequency.
        """
        metric_name = metric_config.name
        # Check step frequency: Log if current step is a multiple of frequency
        # Handles step 0 correctly if frequency is > 0
        log_by_step = (
            metric_config.log_frequency_steps > 0
            and current_step % metric_config.log_frequency_steps == 0
        )
        if log_by_step:
            return True

        # Check time frequency: Log if enough time has passed since *last log for this metric*
        last_logged_time = self._last_log_time.get(metric_name, 0.0)
        time_delta = context.current_time - last_logged_time
        log_by_time = (
            metric_config.log_frequency_seconds > 0
            and time_delta >= metric_config.log_frequency_seconds
        )
        return bool(log_by_time)

    def _log_to_targets(
        self,
        metric_config: MetricConfig,
        value: float | int,
        log_step: int,
        log_time: float,
    ):
        """Logs the processed value to configured targets and updates tracking."""
        if not np.isfinite(value):
            logger.warning(
                f"Attempted to log non-finite value for {metric_config.name}: {value}. Skipping."
            )
            return

        log_value = float(value)
        metric_name = metric_config.name

        if (
            "mlflow" in metric_config.log_to
            and self.mlflow_client
            and self.mlflow_run_id
        ):
            try:
                # Add debug log here
                logger.debug(
                    f"Logging to MLflow: key='{metric_name}', value={log_value:.4f}, step={log_step}, timestamp={int(log_time * 1000)}"
                )
                self.mlflow_client.log_metric(
                    run_id=self.mlflow_run_id,
                    key=metric_name,
                    value=log_value,
                    step=log_step,
                    timestamp=int(log_time * 1000),
                )
            except Exception as e:
                logger.error(f"Failed to log {metric_name} to MLflow using client: {e}")
        elif "mlflow" in metric_config.log_to:
            logger.warning(
                f"MLflow logging requested for {metric_name} but client/run_id is unavailable."
            )

        if "tensorboard" in metric_config.log_to and self.tb_writer:
            try:
                self.tb_writer.add_scalar(metric_name, log_value, log_step)
            except Exception as e:
                logger.error(f"Failed to log {metric_name} to TensorBoard: {e}")

        if "console" in metric_config.log_to:
            logger.info(f"STATS [{log_step}]: {metric_name} = {log_value:.4f}")

        # Update last logged time for this metric (used for time frequency check)
        self._last_log_time[metric_name] = log_time

    def _process_and_log_internal(
        self,
        raw_data: dict[int, dict[str, list[RawMetricEvent]]],
        context: LogContext,
    ):
        """Internal logic for processing and logging, shared by public methods."""
        if not raw_data:
            return

        # --- Step 1: Aggregate non-rate metrics per step ---
        aggregated_step_values: dict[str, dict[int, float | int]] = defaultdict(dict)

        for step, step_event_dict in sorted(raw_data.items()):
            for metric_config in self.config.metrics:
                # Skip rate metrics in this aggregation phase
                if metric_config.aggregation == "rate":
                    continue

                event_key = metric_config.event_key
                if event_key in step_event_dict:
                    event_list = step_event_dict[event_key]
                    values_to_aggregate: list[float] = []

                    # Extract values
                    if metric_config.context_key:
                        for event in event_list:
                            if metric_config.context_key in event.context:
                                try:
                                    val = float(
                                        event.context[metric_config.context_key]
                                    )
                                    if np.isfinite(val):
                                        values_to_aggregate.append(val)
                                except (ValueError, TypeError):
                                    pass
                    else:
                        values_to_aggregate = [
                            float(e.value) for e in event_list if e.is_valid()
                        ]

                    # Aggregate if values exist
                    if values_to_aggregate:
                        agg_value = self._aggregate_values(
                            values_to_aggregate, metric_config.aggregation
                        )
                        if agg_value is not None:
                            aggregated_step_values[metric_config.name][step] = agg_value

        # --- Step 2: Log aggregated non-rate metrics based on frequency ---
        for metric_name, step_values in aggregated_step_values.items():
            if not step_values:  # Skip if no values were aggregated for this metric
                continue

            maybe_metric_config = self._metric_configs.get(metric_name)
            # Get the config *once* per metric name
            if maybe_metric_config is None:
                logger.warning(
                    f"Metric config not found for '{metric_name}' during logging."
                )
                continue

            metric_config = maybe_metric_config
            # Check if metric_config is None before proceeding

            # Log the value associated with the *latest step* in the batch if conditions met
            latest_step_in_batch = max(step_values.keys())
            latest_value = step_values[latest_step_in_batch]

            # Check frequency *after* confirming config exists
            # Now metric_config is guaranteed to be MetricConfig here
            if self._should_log(metric_config, latest_step_in_batch, context):
                self._log_to_targets(
                    metric_config,
                    latest_value,
                    latest_step_in_batch,
                    context.current_time,
                )

        # --- Step 3: Calculate and log rate metrics ---
        for rate_metric_config in self.config.metrics:
            if rate_metric_config.aggregation == "rate":
                # Use the overall context step for logging rates
                log_step = context.latest_step
                # Check only time frequency for rate logging
                last_rate_log_time = self._last_log_time.get(
                    rate_metric_config.name, 0.0
                )
                should_log_rate_time = (
                    rate_metric_config.log_frequency_seconds > 0
                    and context.current_time - last_rate_log_time
                    >= rate_metric_config.log_frequency_seconds
                )

                if should_log_rate_time:
                    rate_value = self._calculate_rate(
                        rate_metric_config, context, raw_data
                    )
                    if rate_value is not None:
                        self._log_to_targets(
                            rate_metric_config,
                            rate_value,
                            log_step,
                            context.current_time,
                        )

    def process_and_log(
        self,
        raw_data: dict[int, dict[str, list[RawMetricEvent]]],
        context: LogContext,
    ):
        """
        Public method to process and log, called by the actor.
        Delegates to the internal processing logic.
        """
        self._process_and_log_internal(raw_data, context)
