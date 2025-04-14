from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol

from ...utils.types import PolicyValueOutput

if TYPE_CHECKING:
    from ...environment import GameState
    from ...utils.types import ActionType

ActionPolicyMapping = Mapping["ActionType", float]


class ActionPolicyValueEvaluator(Protocol):
    """Defines the interface for evaluating a game state using a neural network."""

    def evaluate(self, state: "GameState") -> PolicyValueOutput:
        """
        Evaluates a single game state using the neural network.

        Args:
            state: The GameState object to evaluate.

        Returns:
            A tuple containing:
                - ActionPolicyMapping: A mapping from valid action indices
                    to their prior probabilities (output by the policy head).
                - float: The estimated value of the state (output by the value head).
        """
        ...

    def evaluate_batch(self, states: list["GameState"]) -> list[PolicyValueOutput]:
        """
        Evaluates a batch of game states using the neural network.
        (Optional but recommended for performance if MCTS supports batch evaluation).

        Args:
            states: A list of GameState objects to evaluate.

        Returns:
            A list of tuples, where each tuple corresponds to an input state and contains:
                - ActionPolicyMapping: Action probabilities for that state.
                - float: The estimated value of that state.
        """
        ...
