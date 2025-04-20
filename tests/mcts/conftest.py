# File: tests/mcts/conftest.py
import random
from collections.abc import Mapping

import numpy as np
import pytest

# Import EnvConfig from trianglengin
from trianglengin.config import EnvConfig

# Keep alphatriangle imports
from alphatriangle.mcts.core.node import Node
from alphatriangle.utils.types import ActionType, PolicyValueOutput

rng = np.random.default_rng()


# --- Mock GameState (using trianglengin.EnvConfig) ---
class MockGameState:
    """A simplified mock GameState for testing MCTS logic."""

    def __init__(
        self,
        current_step: int = 0,
        is_terminal: bool = False,
        outcome: float = 0.0,
        valid_actions: list[ActionType] | None = None,
        env_config: EnvConfig | None = None,  # Expects trianglengin.EnvConfig
    ):
        self.current_step = current_step
        self._is_over = is_terminal
        self._outcome = outcome
        # Use trianglengin.EnvConfig
        self.env_config = env_config if env_config else EnvConfig()
        action_dim_int = int(self.env_config.ACTION_DIM)
        self._valid_actions = (
            valid_actions if valid_actions is not None else list(range(action_dim_int))
        )
        self._game_over_reason: str | None = None  # Add reason attribute

    def is_over(self) -> bool:
        return self._is_over

    def get_outcome(self) -> float:
        if not self._is_over:
            # MCTS expects 0 for non-terminal, not an error
            return 0.0
        return self._outcome

    def valid_actions(self) -> set[ActionType]:  # Return set to match trianglengin
        return set(self._valid_actions)

    def copy(self) -> "MockGameState":
        return MockGameState(
            self.current_step,
            self._is_over,
            self._outcome,
            list(self._valid_actions),
            self.env_config,  # Pass trianglengin.EnvConfig
        )

    def step(self, action: ActionType) -> tuple[float, bool]:
        if action not in self.valid_actions():
            raise ValueError(
                f"Invalid action {action} for mock state. Valid: {self.valid_actions()}"
            )
        self.current_step += 1
        self._is_over = self.current_step >= 10 or len(self._valid_actions) == 0
        self._outcome = -1.0 if self._is_over else 0.0  # Match trianglengin outcome
        if action in self._valid_actions:
            self._valid_actions.remove(action)
        elif self._valid_actions and random.random() < 0.5:
            self._valid_actions.pop(random.randrange(len(self._valid_actions)))
        return 0.0, self._is_over  # Return dummy reward

    def force_game_over(self, reason: str):  # Add method
        self._is_over = True
        self._game_over_reason = reason
        self._valid_actions = []

    def __hash__(self):
        return hash(
            (self.current_step, self._is_over, tuple(sorted(self._valid_actions)))
        )

    def __eq__(self, other):
        if not isinstance(other, MockGameState):
            return NotImplemented
        return (
            self.current_step == other.current_step
            and self._is_over == other._is_over
            and sorted(self._valid_actions) == sorted(other._valid_actions)
            and self.env_config == other.env_config
        )


# ... (MockNetworkEvaluator remains the same, uses MockGameState) ...
class MockNetworkEvaluator:
    """A mock network evaluator for testing MCTS."""

    def __init__(
        self,
        default_policy: Mapping[ActionType, float] | None = None,
        default_value: float = 0.5,
        action_dim: int = 9,
    ):
        self._default_policy = default_policy
        self._default_value = default_value
        self._action_dim = action_dim
        self.evaluation_history: list[MockGameState] = []
        self.batch_evaluation_history: list[list[MockGameState]] = []

    def _get_policy(self, state: MockGameState) -> Mapping[ActionType, float]:
        if self._default_policy is not None:
            valid_actions = state.valid_actions()
            policy = {
                a: p for a, p in self._default_policy.items() if a in valid_actions
            }
            policy_sum = sum(policy.values())
            if policy_sum > 1e-9 and abs(policy_sum - 1.0) > 1e-6:
                policy = {a: p / policy_sum for a, p in policy.items()}
            elif not policy and valid_actions:
                prob = 1.0 / len(valid_actions)
                policy = dict.fromkeys(valid_actions, prob)
            return policy

        valid_actions = state.valid_actions()
        if not valid_actions:
            return {}
        prob = 1.0 / len(valid_actions)
        return dict.fromkeys(valid_actions, prob)

    def evaluate(self, state: MockGameState) -> PolicyValueOutput:
        self.evaluation_history.append(state)
        self._action_dim = int(state.env_config.ACTION_DIM)
        policy = self._get_policy(state)
        full_policy = dict.fromkeys(range(self._action_dim), 0.0)
        full_policy.update(policy)
        return full_policy, self._default_value

    def evaluate_batch(self, states: list[MockGameState]) -> list[PolicyValueOutput]:
        self.batch_evaluation_history.append(states)
        results = []
        for state in states:
            results.append(self.evaluate(state))
        return results


@pytest.fixture
def mock_evaluator(mock_env_config: EnvConfig) -> MockNetworkEvaluator:
    """Provides a MockNetworkEvaluator instance configured with the mock EnvConfig."""
    action_dim_int = int(mock_env_config.ACTION_DIM)
    return MockNetworkEvaluator(action_dim=action_dim_int)


@pytest.fixture
def root_node_mock_state(mock_env_config: EnvConfig) -> Node:
    """Provides a root Node with a MockGameState using the mock EnvConfig."""
    action_dim_int = int(mock_env_config.ACTION_DIM)
    # Pass trianglengin.EnvConfig to MockGameState
    state = MockGameState(
        valid_actions=list(range(action_dim_int)),
        env_config=mock_env_config,
    )
    return Node(state=state)  # type: ignore [arg-type]


# ... (expanded_node_mock_state remains the same, using MockGameState) ...
@pytest.fixture
def expanded_node_mock_state(
    root_node_mock_state: Node, mock_evaluator: MockNetworkEvaluator
) -> Node:
    """Provides an expanded root node with mock children using mock EnvConfig."""
    root = root_node_mock_state
    mock_state: MockGameState = root.state  # type: ignore [assignment]
    mock_evaluator._action_dim = int(mock_state.env_config.ACTION_DIM)
    policy, value = mock_evaluator.evaluate(mock_state)
    if not policy:
        policy = (
            dict.fromkeys(
                mock_state.valid_actions(), 1.0 / len(mock_state.valid_actions())
            )
            if mock_state.valid_actions()
            else {}
        )

    for action in mock_state.valid_actions():
        prior = policy.get(action, 0.0)
        child_state = MockGameState(
            current_step=1, valid_actions=[0, 1], env_config=mock_state.env_config
        )
        child = Node(
            state=child_state,  # type: ignore [arg-type]
            parent=root,
            action_taken=action,
            prior_probability=prior,
        )
        root.children[action] = child
    root.visit_count = 1
    root.total_action_value = value
    return root


@pytest.fixture
def deep_expanded_node_mock_state(
    expanded_node_mock_state: Node,
    mock_evaluator: MockNetworkEvaluator,
    mock_env_config: EnvConfig,
) -> Node:
    """
    Provides a root node expanded two levels deep, specifically configured
    to encourage traversal down the path leading to action 0, then action 1.
    """
    root = expanded_node_mock_state
    mock_evaluator._action_dim = int(mock_env_config.ACTION_DIM)

    if 0 not in root.children or 1 not in root.children:
        pytest.skip("Actions 0 or 1 not available in expanded node children")

    root.visit_count = 100
    child0 = root.children[0]

    child0.visit_count = 80
    child0.total_action_value = 40
    child0.prior_probability = 0.8

    for action, child in root.children.items():
        if action != 0:
            child.visit_count = 2
            child.total_action_value = 0
            child.prior_probability = 0.01

    mock_child0_state: MockGameState = child0.state  # type: ignore [assignment]
    policy_gc, value_gc = mock_evaluator.evaluate(mock_child0_state)
    if not policy_gc:
        policy_gc = (
            dict.fromkeys(
                mock_child0_state.valid_actions(),
                1.0 / len(mock_child0_state.valid_actions()),
            )
            if mock_child0_state.valid_actions()
            else {}
        )

    # Convert set to list before checking/indexing
    valid_gc_actions_list = list(mock_child0_state.valid_actions())
    if 1 in valid_gc_actions_list:
        preferred_gc_action = 1
    elif valid_gc_actions_list:
        # If action 1 is not available, pick the first available action
        preferred_gc_action = valid_gc_actions_list[0]
    else:
        pytest.skip("Child 0 has no valid actions to create grandchildren")

    for action_gc in valid_gc_actions_list:
        prior_gc = policy_gc.get(action_gc, 0.0)
        grandchild_state = MockGameState(
            current_step=2, valid_actions=[0], env_config=mock_child0_state.env_config
        )
        grandchild = Node(
            state=grandchild_state,  # type: ignore [arg-type]
            parent=child0,
            action_taken=action_gc,
            prior_probability=prior_gc,
        )
        child0.children[action_gc] = grandchild

    preferred_grandchild = child0.children.get(preferred_gc_action)
    if preferred_grandchild:
        preferred_grandchild.visit_count = 60
        preferred_grandchild.total_action_value = 30
        preferred_grandchild.prior_probability = 0.7

    for action_gc, grandchild in child0.children.items():
        if action_gc != preferred_gc_action:
            grandchild.visit_count = 1
            grandchild.total_action_value = 0
            grandchild.prior_probability = 0.05

    return root
