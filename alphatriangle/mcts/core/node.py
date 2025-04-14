from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alphatriangle.environment import GameState
    from alphatriangle.utils.types import ActionType

logger = logging.getLogger(__name__)


class Node:
    """Represents a node in the Monte Carlo Search Tree."""

    def __init__(
        self,
        state: GameState,
        parent: Node | None = None,
        action_taken: ActionType | None = None,
        prior_probability: float = 0.0,
    ):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken

        self.children: dict[ActionType, Node] = {}

        self.visit_count: int = 0
        self.total_action_value: float = 0.0
        self.prior_probability: float = prior_probability

    @property
    def is_expanded(self) -> bool:
        """Checks if the node has been expanded (i.e., children generated)."""
        return bool(self.children)

    @property
    def is_leaf(self) -> bool:
        """Checks if the node is a leaf (not expanded)."""
        return not self.is_expanded

    @property
    def value_estimate(self) -> float:
        """
        Calculates the Q-value (average action value) estimate for this node's state.
        This is the average value observed from simulations starting from this state.
        Refactored for clarity and safety.
        """
        if self.visit_count == 0:
            return 0.0

        visits = max(1, self.visit_count)
        q_value = self.total_action_value / visits

        return q_value

    def __repr__(self) -> str:
        return (
            f"Node(StateStep={self.state.current_step}, "
            f"FromAction={self.action_taken}, Visits={self.visit_count}, "
            f"Value={self.value_estimate:.3f}, Prior={self.prior_probability:.4f}, "
            f"Children={len(self.children)})"
        )
