# File: src/stats/collector.py
import logging
import time  # Import time
from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np
import ray

from src.utils.types import StatsCollectorData

if TYPE_CHECKING:
    from src.environment import GameState  # Import GameState for type hint

# Get logger instance for this module
logger = logging.getLogger(__name__)


@ray.remote
class StatsCollectorActor:
    """
    Ray actor for collecting time-series statistics and latest worker game states.
    """

    def __init__(self, max_history: int | None = 1000):
        self.max_history = max_history
        self._data: StatsCollectorData = {}
        # Store the latest GameState reported by each worker
        self._latest_worker_states: dict[int, GameState] = {}
        self._last_state_update_time: dict[int, float] = {}  # Track update times

        print(f"[StatsCollectorActor] Initialized with max_history={max_history}.")
        logger.info(f"Initialized with max_history={max_history}.")

    # --- Metric Logging ---

    def log(self, metric_name: str, value: float, step: int):
        """Logs a single metric value."""
        logger.debug(
            f"Log received: metric='{metric_name}', value={value}, step={step}"
        )
        if not isinstance(metric_name, str):
            logger.error(f"Invalid metric_name type: {type(metric_name)}")
            return
        if not np.isfinite(value):
            logger.warning(
                f"Received non-finite value for metric '{metric_name}': {value}. Skipping log."
            )
            return
        if metric_name not in self._data:
            self._data[metric_name] = deque(maxlen=self.max_history)
        try:
            # Ensure step is int and value is float for consistency
            self._data[metric_name].append((int(step), float(value)))
        except (ValueError, TypeError) as e:
            logger.error(
                f"Could not log metric '{metric_name}'. Invalid step/value: {e}"
            )

    def log_batch(self, metrics: dict[str, tuple[float, int]]):
        """Logs a batch of metrics."""
        logger.debug(
            f"Log batch received with {len(metrics)} metrics: {list(metrics.keys())}"
        )
        for name, (value, step) in metrics.items():
            self.log(name, value, step)  # Delegate to single log method

    # --- Game State Handling ---

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
        # Return a copy to prevent external modification of the internal dict
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

    def get_metric_data(self, metric_name: str) -> deque[tuple[int, float]] | None:
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
        # --- CHANGE: Convert deques to lists for serialization ---
        serializable_metrics = {key: list(dq) for key, dq in self._data.items()}
        # --- END CHANGE ---

        # Note: GameState objects are complex and might be large.
        # Saving them frequently might be slow and consume disk space.
        # Consider if saving the latest worker states is truly necessary for resuming,
        # or if just saving metrics/model/buffer is sufficient.
        # For now, we exclude worker states from the checkpoint state.
        state = {
            "max_history": self.max_history,
            "_metrics_data_list": serializable_metrics,  # Use the list version
            # "_latest_worker_states": self._latest_worker_states # Excluded for now
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
                isinstance(item, tuple) and len(item) == 2 for item in items_list
            ):
                # Ensure items are (int, float)
                valid_items = []
                for item in items_list:
                    try:
                        valid_items.append((int(item[0]), float(item[1])))
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Skipping invalid item {item} in metric '{key}' during state restore."
                        )
                # --- CHANGE: Convert list back to deque ---
                self._data[key] = deque(valid_items, maxlen=self.max_history)
                # --- END CHANGE ---
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
