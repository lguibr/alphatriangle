# File: alphatriangle/stats/collector.py
import logging
import threading
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any

import ray
from torch.utils.tensorboard import SummaryWriter

from .processor import StatsProcessor
from .stats_types import LogContext, RawMetricEvent

if TYPE_CHECKING:
    from trianglengin.core.environment import GameState

    from ..config import StatsConfig

logger = logging.getLogger(__name__)


@ray.remote
class StatsCollectorActor:
    """
    Ray actor for asynchronously collecting raw metric events and triggering
    processing and logging based on StatsConfig.
    """

    def __init__(
        self,
        stats_config: "StatsConfig",
        run_name: str,
        tb_log_dir: str | None,
        mlflow_run_id: str | None,
    ):
        global logger
        logger = logging.getLogger(__name__)

        self._config = stats_config
        self._run_name = run_name
        self._tb_log_dir = tb_log_dir
        self._mlflow_run_id = mlflow_run_id

        # Buffer now stores lists of RawMetricEvent objects
        self._raw_data_buffer: dict[int, dict[str, list[RawMetricEvent]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._latest_values: dict[str, tuple[int, float]] = {}
        self._event_timestamps: dict[str, deque[tuple[float, int]]] = defaultdict(
            lambda: deque(maxlen=200)
        )
        # Initialize to -1, indicating no steps processed yet
        self._last_processed_step = -1
        self._last_processed_time = time.monotonic()

        self._latest_worker_states: dict[int, GameState] = {}
        self._last_state_update_time: dict[int, float] = {}

        self.tb_writer: SummaryWriter | None = None
        if self._tb_log_dir:
            try:
                self.tb_writer = SummaryWriter(log_dir=self._tb_log_dir)
                logger.info(
                    f"StatsCollectorActor: TensorBoard writer initialized at {self._tb_log_dir}"
                )
            except Exception as e:
                logger.error(
                    f"StatsCollectorActor: Failed to initialize TensorBoard writer: {e}"
                )

        self.processor = StatsProcessor(
            self._config,
            self._run_name,
            self.tb_writer,
            self._mlflow_run_id,
        )

        self._lock = threading.Lock()
        logger.info(
            f"StatsCollectorActor initialized for run '{self._run_name}'. Processing interval: {self._config.processing_interval_seconds}s. MLflow Run ID: {self._mlflow_run_id}"
        )

    def get_config(self) -> "StatsConfig":
        return self._config

    def get_run_name(self) -> str:
        return self._run_name

    def get_tb_log_dir(self) -> str | None:
        return self._tb_log_dir

    def log_event(self, event: RawMetricEvent):
        """Logs a single raw metric event."""
        if not isinstance(event, RawMetricEvent):
            logger.error(f"Received invalid event type: {type(event)}")
            return
        if not isinstance(event.name, str) or not event.name:
            logger.error(f"Invalid event name: {event.name}")
            return
        if not isinstance(event.value, int | float) or not isinstance(
            event.global_step, int
        ):
            logger.error(
                f"Invalid event value or step type: value={type(event.value)}, step={type(event.global_step)}"
            )
            return
        if not isinstance(event.context, dict):
            logger.error(f"Invalid event context type: {type(event.context)}")
            return

        if not event.is_valid():
            logger.warning(
                f"Received non-finite value for event '{event.name}': {event.value}. Skipping."
            )
            return

        with self._lock:
            step_data = self._raw_data_buffer[event.global_step]
            # Store the entire RawMetricEvent object
            step_data[event.name].append(event)
            # Still store the latest float value for quick access if needed elsewhere
            self._latest_values[event.name] = (event.global_step, float(event.value))
            is_rate_numerator = any(
                mc.rate_numerator_event == event.name
                for mc in self._config.metrics
                if mc.aggregation == "rate"
            )
            if is_rate_numerator:
                self._event_timestamps[event.name].append(
                    (time.monotonic(), event.global_step)
                )

        logger.debug(
            f"Logged event: {event.name}, Value: {event.value}, Step: {event.global_step}"
        )

    def log_batch_events(self, events: list[RawMetricEvent]):
        """Logs a batch of raw metric events."""
        logger.debug(f"Received batch of {len(events)} events.")
        for event in events:
            self.log_event(event)

    def _process_and_log_internal(self, current_global_step: int):
        """Internal logic shared by process_and_log and force_process_and_log."""
        now = time.monotonic()
        logger.debug(f"Processing stats up to step {current_global_step}...")
        # Prepare data with the correct type for the processor
        processed_data: dict[int, dict[str, list[RawMetricEvent]]] = {}
        steps_to_process = []
        timestamp_data = {}
        latest_values_copy = {}
        max_step_processed_in_batch = self._last_processed_step

        with self._lock:
            # Process all steps <= current_global_step that haven't been processed
            # This now includes step 0 if it hasn't been processed.
            steps_to_process = sorted(
                [step for step in self._raw_data_buffer if step <= current_global_step]
            )
            if not steps_to_process:
                logger.debug("No new steps to process.")
                return

            # Prepare data for the processor (copy the lists of RawMetricEvent)
            for step in steps_to_process:
                processed_data[step] = {
                    key: list(event_list)  # Create a copy of the list
                    for key, event_list in self._raw_data_buffer[step].items()
                }
                # Track the maximum step number we are actually processing
                max_step_processed_in_batch = max(max_step_processed_in_batch, step)

            timestamp_data = {
                name: list(dq) for name, dq in self._event_timestamps.items()
            }
            latest_values_copy = self._latest_values.copy()

        # Process and log outside the lock
        try:
            context = LogContext(
                latest_step=current_global_step,  # Use the loop's current step for context
                last_log_time=self._last_processed_time,
                current_time=now,
                event_timestamps=timestamp_data,
                latest_values=latest_values_copy,
            )
            if self.processor:
                # Pass the correctly typed data
                self.processor.process_and_log(processed_data, context)
                logger.debug(f"Processing complete for steps {steps_to_process}.")
            else:
                logger.error("StatsProcessor not initialized, cannot process stats.")

        except Exception as e:
            logger.error(f"Error during stats processing/logging: {e}", exc_info=True)

        # Clean up processed steps from the buffer (inside lock)
        with self._lock:
            for step in steps_to_process:
                if step in self._raw_data_buffer:
                    del self._raw_data_buffer[step]
            # Update last processed step to the maximum step handled in this batch
            self._last_processed_step = max_step_processed_in_batch

    def process_and_log(self, current_global_step: int):
        """
        Processes buffered raw data up to the current global step and logs
        aggregated metrics according to the configuration, respecting the
        processing interval.
        """
        now = time.monotonic()
        if now - self._last_processed_time < self._config.processing_interval_seconds:
            return

        self._process_and_log_internal(current_global_step)
        with self._lock:
            self._last_processed_time = now

    def force_process_and_log(self, current_global_step: int):
        """
        Forces processing and logging of buffered raw data, bypassing the
        time interval check. Intended for testing or final flush.
        """
        logger.info(
            f"Forcing stats processing up to step {current_global_step} (bypassing time interval)."
        )
        self._process_and_log_internal(current_global_step)
        with self._lock:
            self._last_processed_time = time.monotonic()

    def update_worker_game_state(self, worker_id: int, game_state: "GameState"):
        """Stores the latest game state received from a worker."""
        if not isinstance(worker_id, int):
            logger.error(f"Invalid worker_id type: {type(worker_id)}")
            return
        if not hasattr(game_state, "grid_data") or not hasattr(game_state, "shapes"):
            logger.error(
                f"Invalid game_state object received from worker {worker_id}: type={type(game_state)}"
            )
            return
        with self._lock:
            self._latest_worker_states[worker_id] = game_state
            self._last_state_update_time[worker_id] = time.time()
        logger.debug(
            f"Updated game state for worker {worker_id} (Step: {game_state.current_step})"
        )

    def get_latest_worker_states(self) -> dict[int, "GameState"]:
        """Returns a shallow copy of the latest worker states dictionary."""
        with self._lock:
            states = self._latest_worker_states.copy()
        logger.debug(
            f"get_latest_worker_states called. Returning states for workers: {list(states.keys())}"
        )
        return states

    def get_state(self) -> dict[str, Any]:
        """Returns the internal state for saving (minimal state needed)."""
        with self._lock:
            state = {
                "last_processed_step": self._last_processed_step,
                "last_processed_time": self._last_processed_time,
            }
        logger.info(f"get_state called. Returning state: {state}")
        return state

    def set_state(self, state: dict[str, Any]):
        """Restores the internal state from saved data."""
        with self._lock:
            self._last_processed_step = state.get("last_processed_step", -1)
            self._last_processed_time = state.get(
                "last_processed_time", time.monotonic()
            )
            self._raw_data_buffer.clear()
            self._latest_values = {}
            self._event_timestamps.clear()
            self._latest_worker_states = {}
            self._last_state_update_time = {}
        logger.info(f"State restored. Last processed step: {self._last_processed_step}")

    def close_tb_writer(self):
        """Closes the TensorBoard writer if it exists."""
        if hasattr(self, "tb_writer") and self.tb_writer:
            try:
                self.tb_writer.flush()
                self.tb_writer.close()
                logger.info("StatsCollectorActor: TensorBoard writer closed.")
                self.tb_writer = None
            except Exception as e:
                logger.error(
                    f"StatsCollectorActor: Error closing TensorBoard writer: {e}"
                )

    def _get_internal_state_for_testing(self) -> dict[str, Any]:
        """Returns copies of internal state for testing purposes ONLY."""
        logger.warning(
            "_get_internal_state_for_testing called. THIS SHOULD ONLY HAPPEN IN TESTS."
        )
        with self._lock:
            # Return copies to avoid modifying internal state directly
            return {
                "raw_data_buffer": self._raw_data_buffer.copy(),
                "latest_values": self._latest_values.copy(),
                "event_timestamps": self._event_timestamps.copy(),
                "last_processed_step": self._last_processed_step,
                "last_processed_time": self._last_processed_time,
                "mlflow_run_id": self._mlflow_run_id,
            }

    def __del__(self):
        """Ensure TensorBoard writer is closed when actor is destroyed."""
        if hasattr(self, "close_tb_writer"):
            self.close_tb_writer()
