# File: alphatriangle/rl/core/visual_state_actor.py
import logging
import time
from typing import Any

import ray

from ...environment import GameState

logger = logging.getLogger(__name__)


@ray.remote
class VisualStateActor:
    """A simple Ray actor to hold the latest game states from workers for visualization."""

    def __init__(self) -> None:
        self.worker_states: dict[int, GameState] = {}
        self.global_stats: dict[str, Any] = {}
        self.last_update_times: dict[int, float] = {}

    def update_state(self, worker_id: int, game_state: GameState):
        """Workers call this to update their latest state."""
        self.worker_states[worker_id] = game_state
        self.last_update_times[worker_id] = time.time()

    def update_global_stats(self, stats: dict[str, Any]):
        """Orchestrator calls this to update global stats."""
        # Ensure stats is a dictionary
        if isinstance(stats, dict):
            # Use update to merge instead of direct assignment
            self.global_stats.update(stats)
        else:
            # Handle error or log warning if stats is not a dict
            logger.error(
                f"VisualStateActor received non-dict type for global stats: {type(stats)}"
            )
            # Don't reset, just ignore the update
            # self.global_stats = {}

    def get_all_states(self) -> dict[int, Any]:
        """
        Called by the orchestrator to get states for the visual queue.
        Key -1 holds the global_stats dictionary.
        Other keys hold GameState objects.
        """
        # Use dict() constructor instead of comprehension for ruff C416
        # Cast worker_states to dict[int, Any] before combining
        combined_states: dict[int, Any] = dict(self.worker_states)
        combined_states[-1] = self.global_stats.copy()
        return combined_states

    def get_state(self, worker_id: int) -> GameState | None:
        """Get state for a specific worker (unused currently)."""
        return self.worker_states.get(worker_id)
