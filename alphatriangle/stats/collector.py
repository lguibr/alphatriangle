# File: alphatriangle/stats/collector.py
import logging
import time
from collections import deque
from typing import TYPE_CHECKING, Any, cast  # Added cast

import numpy as np
import ray

from ..utils.types import StatsCollectorData, StepInfo

if TYPE_CHECKING:
    from ..environment import GameState

logger = logging.getLogger(__name__)


@ray.remote
class StatsCollectorActor:
    """
    Ray actor for collecting time-series statistics and latest worker game states.
    Stores metrics as (StepInfo, value) tuples.
    """

    def __init__(self, max_history: int | None = 1000):
        self.max_history = max_history
        self._data: StatsCollectorData = {}
        # Store the latest GameState reported by each worker
        self._latest_worker_states: dict[int, GameState] = {}
        self._last_state_update_time: dict[int, float] = {}

        # Ensure logger is configured for the actor process
        log_level = logging.INFO
        # Check if runtime_context is available before using it
        actor_id_str = "UnknownActor"
        try:
            if ray.is_initialized():
                actor_id_str = ray.get_runtime_context().get_actor_id()
        except Exception:
            pass  # Ignore if context cannot be retrieved
        log_format = f"%(asctime)s [%(levelname)s] [StatsCollectorActor pid={actor_id_str}] %(name)s: %(message)s"
        logging.basicConfig(level=log_level, format=log_format, force=True)
        global logger  # Re-assign logger after config
        logger = logging.getLogger(__name__)

        logger.info(f"Initialized with max_history={max_history}.")

    # --- Metric Logging ---

    def log(self, metric_name: str, value: float, step_info: StepInfo):
        """Logs a single metric value with its associated step information."""
        logger.debug(
            f"Attempting to log metric='{metric_name}', value={value}, step_info={step_info}"
        )
        if not isinstance(metric_name, str):
            logger.error(f"Invalid metric_name type: {type(metric_name)}")
            return
        if not isinstance(step_info, dict):
            logger.error(f"Invalid step_info type: {type(step_info)}")
            return
        if not np.isfinite(value):
            logger.warning(
                f"Received non-finite value for metric '{metric_name}': {value}. Skipping log."
            )
            return

        try:
            if metric_name not in self._data:
                logger.debug(f"Creating new deque for metric: '{metric_name}'")
                self._data[metric_name] = deque(maxlen=self.max_history)

            # Ensure value is float for consistency
            value_float = float(value)
            # Store the StepInfo dict directly
            self._data[metric_name].append((step_info, value_float))
            logger.debug(
                f"Successfully logged metric='{metric_name}', value={value_float}, step_info={step_info}. Deque size: {len(self._data[metric_name])}"
            )
        except (ValueError, TypeError) as e:
            logger.error(
                f"Could not log metric '{metric_name}'. Invalid value conversion: {e}"
            )
        except Exception as e:
            logger.error(
                f"Unexpected error logging metric '{metric_name}' (value={value}, step_info={step_info}): {e}",
                exc_info=True,
            )

    def log_batch(self, metrics: dict[str, tuple[float, StepInfo]]):
        """Logs a batch of metrics, each with its StepInfo."""
        received_keys = list(metrics.keys())
        logger.debug(
            f"Log batch received with {len(metrics)} metrics. Keys: {received_keys}"
        )
        for name, (value, step_info) in metrics.items():
            self.log(name, value, step_info)  # Delegate to single log method

    # --- Game State Handling (No change needed) ---

    def update_worker_game_state(self, worker_id: int, game_state: "GameState"):
        """Stores the latest game state received from a worker."""
        if not isinstance(worker_id, int):
            logger.error(f"Invalid worker_id type: {type(worker_id)}")
            return
        # Basic check if it looks like a GameState object (can add more checks if needed)
        if not hasattr(game_state, "grid_data") or not hasattr(game_state, "shapes"):
            logger.error(
                f"Invalid game_state object received from worker {worker_id}: type={type(game_state)}"
            )
            return
        # Store the received state (it should be a copy from the worker)
        self._latest_worker_states[worker_id] = game_state
        self._last_state_update_time[worker_id] = time.time()
        logger.debug(
            f"Updated game state for worker {worker_id} (Step: {game_state.current_step})"
        )

    def get_latest_worker_states(self) -> dict[int, "GameState"]:
        """Returns a shallow copy of the latest worker states dictionary."""
        logger.debug(
            f"get_latest_worker_states called. Returning states for workers: {list(self._latest_worker_states.keys())}"
        )
        return self._latest_worker_states.copy()

    # --- Data Retrieval & Management ---

    def get_data(self) -> StatsCollectorData:
        """Returns a copy of the collected statistics data."""
        logger.debug(f"get_data called. Returning {len(self._data)} metrics.")
        # Return copies of deques to prevent external modification
        return {k: dq.copy() for k, dq in self._data.items()}

    def get_metric_data(self, metric_name: str) -> deque[tuple[StepInfo, float]] | None:
        """Returns a copy of the data deque for a specific metric."""
        dq = self._data.get(metric_name)
        return dq.copy() if dq else None

    def clear(self):
        """Clears all collected statistics and worker states."""
        self._data = {}
        self._latest_worker_states = {}
        self._last_state_update_time = {}
        logger.info("Data and worker states cleared.")

    def get_state(self) -> dict[str, Any]:
        """Returns the internal state for saving."""
        # Convert deques to lists for serialization compatibility with cloudpickle/json
        # The items in the list are now (StepInfo, float) tuples
        serializable_metrics = {key: list(dq) for key, dq in self._data.items()}

        state = {
            "max_history": self.max_history,
            "_metrics_data_list": serializable_metrics,  # Use the list version
        }
        logger.info(
            f"get_state called. Returning state for {len(serializable_metrics)} metrics. Worker states NOT included."
        )
        return state

    def set_state(self, state: dict[str, Any]):
        """Restores the internal state from saved data."""
        self.max_history = state.get("max_history", self.max_history)
        loaded_metrics_list = state.get("_metrics_data_list", {})
        self._data = {}
        restored_metrics_count = 0
        for key, items_list in loaded_metrics_list.items():
            if isinstance(items_list, list) and all(
                isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], dict)
                for item in items_list
            ):
                # Ensure items are (StepInfo, float)
                valid_items: list[tuple[StepInfo, float]] = []
                for item in items_list:
                    try:
                        # Basic check for StepInfo structure (can be enhanced)
                        if not isinstance(item[0], dict):
                            raise TypeError("StepInfo is not a dict")
                        # Ensure value is float
                        value = float(item[1])
                        # Cast the dict to StepInfo for type safety
                        step_info = cast("StepInfo", item[0])
                        valid_items.append((step_info, value))
                    except (ValueError, TypeError, IndexError) as e:
                        logger.warning(
                            f"Skipping invalid item {item} in metric '{key}' during state restore: {e}"
                        )
                # Convert list back to deque with maxlen
                # Cast valid_items to the expected type for deque
                self._data[key] = deque(
                    cast("list[tuple[StepInfo, float]]", valid_items),
                    maxlen=self.max_history,
                )
                restored_metrics_count += 1
            else:
                logger.warning(
                    f"Skipping restore for metric '{key}'. Invalid data format: {type(items_list)}"
                )
        # Clear worker states on restore, as they are transient
        self._latest_worker_states = {}
        self._last_state_update_time = {}
        logger.info(
            f"State restored. Restored {restored_metrics_count} metrics. Max history: {self.max_history}. Worker states cleared."
        )
