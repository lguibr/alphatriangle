from __future__ import annotations
import math
import logging
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.environment import GameState
    from src.utils.types import ActionType

logger = logging.getLogger(__name__)


class Node:
    """Represents a node in the Monte Carlo Search Tree."""

    def __init__(
        self,
        state: "GameState",
        parent: Optional[Node] = None,
        action_taken: Optional["ActionType"] = None,
        prior_probability: float = 0.0,
    ):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken

        self.children: Dict["ActionType", Node] = {}

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

        if self.visit_count == 0:
            logger.warning(
                f"Node {self} had visit_count=0 but value_estimate was accessed. Returning 0."
            )

        return q_value

    def __repr__(self) -> str:
        parent_action = self.parent.action_taken if self.parent else "Root"
        return (
            f"Node(StateStep={self.state.current_step}, "
            f"FromAction={self.action_taken}, Visits={self.visit_count}, "
            f"Value={self.value_estimate:.3f}, Prior={self.prior_probability:.4f}, "
            f"Children={len(self.children)})"
        )
