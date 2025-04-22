# File: alphatriangle/stats/collector.py
import logging
import threading
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any  # Import Optional

import ray
from torch.utils.tensorboard import SummaryWriter

from .processor import StatsProcessor

# Update import to use the new filename
from .stats_types import LogContext, RawMetricEvent

if TYPE_CHECKING:
    from trianglengin.core.environment import GameState

    from ..config import StatsConfig

# Initialize logger at the module level
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
    ):
        # Move global logger assignment here before any logging calls in __init__
        global logger
        logger = logging.getLogger(
            __name__
        )  # Ensure actor uses the correct logger instance

        self._config = stats_config  # Store internally
        self._run_name = run_name  # Store internally
        self._tb_log_dir = tb_log_dir  # Store internally

        # Internal state for raw data buffering
        self._raw_data_buffer: dict[int, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._latest_values: dict[str, tuple[int, float]] = {}
        self._event_timestamps: dict[str, deque[tuple[float, int]]] = defaultdict(
            lambda: deque(maxlen=200)
        )
        self._last_processed_step = -1
        self._last_processed_time = time.monotonic()

        # Game state tracking
        self._latest_worker_states: dict[int, GameState] = {}
        self._last_state_update_time: dict[int, float] = {}

        # Processor and logging setup
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

        # Always instantiate the processor internally
        self.processor = StatsProcessor(self._config, self._run_name, self.tb_writer)

        self._lock = threading.Lock()

        logger.info(
            f"StatsCollectorActor initialized for run '{self._run_name}'. Processing interval: {self._config.processing_interval_seconds}s"
        )

    # --- Getters for Testability ---
    def get_config(self) -> "StatsConfig":
        return self._config

    def get_run_name(self) -> str:
        return self._run_name

    def get_tb_log_dir(self) -> str | None:
        return self._tb_log_dir

    # --- Event Logging ---
    def log_event(self, event: RawMetricEvent):
        """Logs a single raw metric event."""
        if not isinstance(event, RawMetricEvent):
            logger.error(f"Received invalid event type: {type(event)}")
            return

        # Basic validation
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

        # Use float for consistency
        value = float(event.value)

        if not event.is_valid():
            logger.warning(
                f"Received non-finite value for event '{event.name}': {value}. Skipping."
            )
            return

        with self._lock:
            step_data = self._raw_data_buffer[event.global_step]
            step_data[event.name].append(value)
            self._latest_values[event.name] = (event.global_step, value)
            # Record timestamp for rate calculation events
            is_rate_numerator = any(
                mc.rate_numerator_event == event.name
                for mc in self._config.metrics
                if mc.aggregation == "rate"  # Use self._config
            )
            if is_rate_numerator:
                self._event_timestamps[event.name].append(
                    (time.monotonic(), event.global_step)
                )

        logger.debug(
            f"Logged event: {event.name}, Value: {value}, Step: {event.global_step}"
        )

    def log_batch_events(self, events: list[RawMetricEvent]):
        """Logs a batch of raw metric events."""
        logger.debug(f"Received batch of {len(events)} events.")
        for event in events:
            self.log_event(event)  # Delegate to single event logging

    def _process_and_log_internal(self, current_global_step: int):
        """Internal logic shared by process_and_log and force_process_and_log."""
        now = time.monotonic()
        logger.debug(f"Processing stats up to step {current_global_step}...")
        processed_data = {}
        steps_to_process = []
        timestamp_data = {}
        latest_values_copy = {}

        with self._lock:
            # Collect steps to process (from last processed up to current)
            steps_to_process = sorted(
                [
                    step
                    for step in self._raw_data_buffer
                    if step > self._last_processed_step and step <= current_global_step
                ]
            )
            if not steps_to_process:
                logger.debug("No new steps to process.")
                # Update time even if no steps processed, only if called via regular process_and_log
                # self._last_processed_time = now # Moved to caller
                return

            # Prepare data for the processor
            for step in steps_to_process:
                processed_data[step] = dict(self._raw_data_buffer[step])  # Copy data

            # Prepare timestamp data for rate calculation
            timestamp_data = {
                name: list(dq) for name, dq in self._event_timestamps.items()
            }
            latest_values_copy = self._latest_values.copy()

        # Process and log outside the lock
        try:
            context = LogContext(
                latest_step=current_global_step,
                last_log_time=self._last_processed_time,
                current_time=now,
                event_timestamps=timestamp_data,
                latest_values=latest_values_copy,
            )
            # Ensure processor exists before calling
            if self.processor:
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
            self._last_processed_step = steps_to_process[-1]
            # self._last_processed_time = now # Moved to caller

    def process_and_log(self, current_global_step: int):
        """
        Processes buffered raw data up to the current global step and logs
        aggregated metrics according to the configuration, respecting the
        processing interval.
        """
        now = time.monotonic()
        # Check time interval condition *before* acquiring the lock
        if (
            now - self._last_processed_time < self._config.processing_interval_seconds
        ):  # Use self._config
            return  # Avoid processing too frequently

        self._process_and_log_internal(current_global_step)

        # Update last processed time only if the time check passed and processing occurred
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
        # Update last processed time after forced processing
        with self._lock:
            self._last_processed_time = time.monotonic()

    # --- Game State Handling (Remains the same) ---
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

    # --- State Management for Checkpointing ---
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
            # Clear transient data on restore
            self._raw_data_buffer.clear()
            self._latest_values = {}
            self._event_timestamps.clear()
            self._latest_worker_states = {}
            self._last_state_update_time = {}
        logger.info(f"State restored. Last processed step: {self._last_processed_step}")

    def close_tb_writer(self):
        """Closes the TensorBoard writer if it exists."""
        # Check if tb_writer exists before accessing
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

    # --- Test-only method ---
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
            }

    def __del__(self):
        """Ensure TensorBoard writer is closed when actor is destroyed."""
        # Add check to prevent AttributeError if __init__ failed early
        if hasattr(self, "close_tb_writer"):
            self.close_tb_writer()
