# File: src/rl/core/visual_state_actor.py
import time
from typing import TYPE_CHECKING, Any

import ray

if TYPE_CHECKING:
    from src.environment import GameState


@ray.remote
class VisualStateActor:
    """A simple Ray actor to hold the latest game states from workers for visualization."""

    def __init__(self):
        self.worker_states: dict[int, GameState] = {}
        self.global_stats: dict[str, Any] = {}
        self.last_update_times: dict[int, float] = {}

    def update_state(self, worker_id: int, game_state: "GameState"):
        """Workers call this to update their latest state."""
        self.worker_states[worker_id] = game_state
        self.last_update_times[worker_id] = time.time()

    def update_global_stats(self, stats: dict[str, Any]):
        """Orchestrator calls this to update global stats."""
        # Ensure stats is a dictionary
        if isinstance(stats, dict):
            self.global_stats = stats
        else:
            # Handle error or log warning if stats is not a dict
            # For mypy, we assume it's a dict based on usage,
            # but runtime check might be good.
            pass  # Or log error

    def get_all_states(self) -> dict[int, Any]:
        """Called by the orchestrator to get states for the visual queue."""
        # Use dict() constructor instead of comprehension for ruff C416
        combined_states = dict(self.worker_states)
        combined_states[-1] = self.global_stats.copy()
        return combined_states

    def get_state(self, worker_id: int) -> "GameState" | None:
        """Get state for a specific worker (unused currently)."""
        return self.worker_states.get(worker_id)
