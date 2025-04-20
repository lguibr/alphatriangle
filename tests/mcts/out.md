File: conftest.py
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


# ... (expanded_node_mock_state, deep_expanded_node_mock_state remain the same, using MockGameState) ...
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

    valid_gc_actions = mock_child0_state.valid_actions()
    if 1 not in valid_gc_actions and valid_gc_actions:
        preferred_gc_action = valid_gc_actions[0]
    elif not valid_gc_actions:
        pytest.skip("Child 0 has no valid actions to create grandchildren")
    else:
        preferred_gc_action = 1

    for action_gc in valid_gc_actions:
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


File: fixtures.py
from collections.abc import Mapping

import pytest

# Use relative imports for alphatriangle components if running tests from project root
# or absolute imports if package is installed
try:
    # Try absolute imports first (for installed package)
    from alphatriangle.config import EnvConfig, MCTSConfig
    from alphatriangle.mcts.core.node import Node
    from alphatriangle.utils.types import ActionType, PolicyValueOutput
except ImportError:
    # Fallback to relative imports (for running tests directly)
    from alphatriangle.config import EnvConfig, MCTSConfig
    from alphatriangle.mcts.core.node import Node
    from alphatriangle.utils.types import ActionType, PolicyValueOutput


# --- Mock GameState ---
class MockGameState:
    """A simplified mock GameState for testing MCTS logic."""

    def __init__(
        self,
        current_step: int = 0,
        is_terminal: bool = False,
        outcome: float = 0.0,
        valid_actions: list[ActionType] | None = None,
        env_config: EnvConfig | None = None,
    ):
        self.current_step = current_step
        self._is_over = is_terminal
        self._outcome = outcome
        # Use a default EnvConfig if none provided, needed for action dim
        # Pydantic models with defaults can be instantiated without args
        self.env_config = env_config if env_config else EnvConfig()
        # Cast ACTION_DIM to int
        action_dim_int = int(self.env_config.ACTION_DIM)
        self._valid_actions = (
            valid_actions if valid_actions is not None else list(range(action_dim_int))
        )

    def is_over(self) -> bool:
        return self._is_over

    def get_outcome(self) -> float:
        if not self._is_over:
            raise ValueError("Cannot get outcome of non-terminal state.")
        return self._outcome

    def valid_actions(self) -> list[ActionType]:
        return self._valid_actions

    def copy(self) -> "MockGameState":
        # Simple copy for testing, doesn't need full state copy
        return MockGameState(
            self.current_step,
            self._is_over,
            self._outcome,
            list(self._valid_actions),
            self.env_config,
        )

    def step(self, action: ActionType) -> tuple[float, bool]:
        # Mock step: advances step, returns dummy values, becomes terminal sometimes
        if action not in self._valid_actions:
            raise ValueError(f"Invalid action {action} for mock state.")
        self.current_step += 1
        # Simple logic: become terminal after 5 steps for testing
        self._is_over = self.current_step >= 5
        self._outcome = 1.0 if self._is_over else 0.0
        # Return dummy reward and done status
        return 0.0, self._is_over

    def __hash__(self):
        # Simple hash for testing purposes
        return hash((self.current_step, self._is_over, tuple(self._valid_actions)))

    def __eq__(self, other):
        if not isinstance(other, MockGameState):
            return NotImplemented
        return (
            self.current_step == other.current_step
            and self._is_over == other._is_over
            and self._valid_actions == other._valid_actions
        )


# --- Mock Network Evaluator ---
class MockNetworkEvaluator:
    """A mock network evaluator for testing MCTS."""

    def __init__(
        self,
        default_policy: Mapping[ActionType, float] | None = None,
        default_value: float = 0.5,
        action_dim: int = 3,  # Default action dim
    ):
        self._default_policy = default_policy
        self._default_value = default_value
        self._action_dim = action_dim  # Already int
        self.evaluation_history: list[MockGameState] = []
        self.batch_evaluation_history: list[list[MockGameState]] = []

    def _get_policy(self, state: MockGameState) -> Mapping[ActionType, float]:
        if self._default_policy is not None:
            return self._default_policy
        # Default uniform policy over valid actions
        valid_actions = state.valid_actions()
        if not valid_actions:
            return {}
        prob = 1.0 / len(valid_actions)
        # Return policy only for valid actions
        return dict.fromkeys(valid_actions, prob)

    def evaluate(self, state: MockGameState) -> PolicyValueOutput:
        self.evaluation_history.append(state)
        policy = self._get_policy(state)
        # Ensure policy sums to 1 if not empty
        if policy:
            policy_sum = sum(policy.values())
            if abs(policy_sum - 1.0) > 1e-6:
                policy = {a: p / policy_sum for a, p in policy.items()}

        # Create full policy map for the action dimension
        full_policy = dict.fromkeys(range(self._action_dim), 0.0)
        full_policy.update(policy)

        return full_policy, self._default_value

    def evaluate_batch(self, states: list[MockGameState]) -> list[PolicyValueOutput]:
        self.batch_evaluation_history.append(states)
        results = []
        for state in states:
            results.append(self.evaluate(state))  # Reuse single evaluate logic
        return results


# --- Pytest Fixtures ---
@pytest.fixture
def mock_env_config() -> EnvConfig:
    """Provides a default EnvConfig for tests."""
    # Pydantic models with defaults can be instantiated without args
    return EnvConfig()


@pytest.fixture
def mock_mcts_config() -> MCTSConfig:
    """Provides a default MCTSConfig for tests."""
    # Pydantic models with defaults can be instantiated without args
    return MCTSConfig()


@pytest.fixture
def mock_evaluator(mock_env_config: EnvConfig) -> MockNetworkEvaluator:
    """Provides a MockNetworkEvaluator instance."""
    # Cast ACTION_DIM to int
    action_dim_int = int(mock_env_config.ACTION_DIM)
    return MockNetworkEvaluator(action_dim=action_dim_int)


@pytest.fixture
def root_node_mock_state(mock_env_config: EnvConfig) -> Node:
    """Provides a root Node with a MockGameState."""
    # Cast ACTION_DIM to int
    action_dim_int = int(mock_env_config.ACTION_DIM)
    state = MockGameState(
        valid_actions=list(range(action_dim_int)),
        env_config=mock_env_config,
    )
    # Cast MockGameState to Any temporarily to satisfy Node's type hint
    return Node(state=state)  # type: ignore [arg-type]


@pytest.fixture
def expanded_node_mock_state(
    root_node_mock_state: Node, mock_evaluator: MockNetworkEvaluator
) -> Node:
    """Provides an expanded root node with mock children."""
    root = root_node_mock_state
    # Cast root.state back to MockGameState for the evaluator
    mock_state: MockGameState = root.state  # type: ignore [assignment]
    policy, value = mock_evaluator.evaluate(mock_state)
    # Manually expand for testing setup
    for action in mock_state.valid_actions():
        prior = policy.get(action, 0.0)
        # Create mock child state (doesn't need to be accurate step)
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
    # Simulate one backpropagation
    root.visit_count = 1
    root.total_action_value = value
    return root


File: test_expansion.py
from typing import Any

import pytest

# Import EnvConfig from trianglengin
from trianglengin.config import EnvConfig

# Keep alphatriangle imports
from alphatriangle.mcts.core.node import Node
from alphatriangle.mcts.strategy import expansion

# Import fixtures from local conftest
from .conftest import MockGameState


# ... (tests remain the same, using MockGameState which now uses trianglengin.EnvConfig) ...
def test_expand_node_with_policy_basic(root_node_mock_state: Node):
    """Test basic node expansion with a valid policy."""
    node = root_node_mock_state
    mock_state: MockGameState = node.state  # type: ignore [assignment]
    valid_actions = mock_state.valid_actions()
    policy = {action: 1.0 / len(valid_actions) for action in valid_actions}

    assert not node.is_expanded
    expansion.expand_node_with_policy(node, policy)

    assert node.is_expanded
    assert len(node.children) == len(valid_actions)
    for action in valid_actions:
        assert action in node.children
        child = node.children[action]
        assert child.parent is node
        assert child.action_taken == action
        assert child.prior_probability == pytest.approx(1.0 / len(valid_actions))
        assert child.state.current_step == node.state.current_step + 1
        assert not child.is_expanded
        assert child.visit_count == 0
        assert child.total_action_value == 0.0


def test_expand_node_with_policy_partial(root_node_mock_state: Node):
    """Test expansion when policy doesn't cover all valid actions (should assign 0 prior)."""
    node = root_node_mock_state
    mock_state: MockGameState = node.state  # type: ignore [assignment]
    valid_actions = mock_state.valid_actions()
    policy = {0: 0.6, 1: 0.4}

    expansion.expand_node_with_policy(node, policy)

    assert node.is_expanded
    assert len(node.children) == len(valid_actions)

    assert 0 in node.children
    assert node.children[0].prior_probability == pytest.approx(0.6)
    assert 1 in node.children
    assert node.children[1].prior_probability == pytest.approx(0.4)
    if 2 in valid_actions:
        assert 2 in node.children
        assert node.children[2].prior_probability == 0.0


def test_expand_node_with_policy_empty_valid_actions(root_node_mock_state: Node):
    """Test expansion when the node's state has no valid actions (but isn't terminal yet)."""
    node = root_node_mock_state
    mock_state: MockGameState = node.state  # type: ignore [assignment]
    mock_state._valid_actions = []
    policy = {0: 1.0}

    expansion.expand_node_with_policy(node, policy)

    assert not node.is_expanded
    assert not node.children
    # Check if the state was forced to game over
    assert node.state.is_over()
    assert "Expansion found no valid actions" in node.state._game_over_reason  # type: ignore


def test_expand_node_with_policy_already_expanded(root_node_mock_state: Node):
    """Test that expanding an already expanded node does nothing."""
    node = root_node_mock_state
    policy = {0: 1.0}
    node.children[0] = Node(
        state=MockGameState(current_step=1, env_config=node.state.env_config),  # type: ignore [arg-type]
        parent=node,
        action_taken=0,
    )

    assert node.is_expanded
    original_children = node.children.copy()
    expansion.expand_node_with_policy(node, policy)
    assert node.children == original_children


def test_expand_node_with_policy_terminal_node(root_node_mock_state: Node):
    """Test that expanding a terminal node does nothing."""
    node = root_node_mock_state
    mock_state: MockGameState = node.state  # type: ignore [assignment]
    mock_state._is_over = True
    policy = {0: 1.0}

    assert not node.is_expanded
    expansion.expand_node_with_policy(node, policy)
    assert not node.is_expanded


def test_expand_node_with_invalid_policy_content(root_node_mock_state: Node):
    """Test expansion handles policy with invalid content (e.g., negative priors)."""
    node = root_node_mock_state
    mock_state: MockGameState = node.state  # type: ignore [assignment]
    valid_actions = mock_state.valid_actions()
    policy = {0: 1.5, 1: -0.5}

    expansion.expand_node_with_policy(node, policy)

    assert node.is_expanded
    assert len(node.children) == len(valid_actions)
    assert node.children[0].prior_probability == pytest.approx(1.5)
    assert node.children[1].prior_probability == 0.0
    if 2 in valid_actions:
        assert node.children[2].prior_probability == 0.0


File: test_selection.py
import math

import pytest

# Import EnvConfig from trianglengin
from trianglengin.config import EnvConfig

# Keep alphatriangle imports
from alphatriangle.config import MCTSConfig
from alphatriangle.mcts.core.node import Node
from alphatriangle.mcts.strategy import selection

# Import fixtures from local conftest
from .conftest import MockGameState


# ... (tests remain the same, using MockGameState which now uses trianglengin.EnvConfig) ...
def test_puct_calculation_unvisited_child(
    mock_mcts_config: MCTSConfig, mock_env_config: EnvConfig
):
    """Test PUCT score for an unvisited child node."""
    parent = Node(state=MockGameState(env_config=mock_env_config))  # type: ignore [arg-type]
    parent.visit_count = 10
    child = Node(
        state=MockGameState(current_step=1, env_config=mock_env_config),  # type: ignore [arg-type]
        parent=parent,
        action_taken=0,
        prior_probability=0.5,
    )
    child.visit_count = 0
    child.total_action_value = 0.0

    score, q_value, exploration = selection.calculate_puct_score(
        child, parent.visit_count, mock_mcts_config
    )

    assert q_value == 0.0
    expected_exploration = (
        mock_mcts_config.puct_coefficient * 0.5 * (math.sqrt(10) / (1 + 0))
    )
    assert exploration == pytest.approx(expected_exploration)
    assert score == pytest.approx(q_value + exploration)


def test_puct_calculation_visited_child(
    mock_mcts_config: MCTSConfig, mock_env_config: EnvConfig
):
    """Test PUCT score for a visited child node."""
    parent = Node(state=MockGameState(env_config=mock_env_config))  # type: ignore [arg-type]
    parent.visit_count = 25
    child = Node(
        state=MockGameState(current_step=1, env_config=mock_env_config),  # type: ignore [arg-type]
        parent=parent,
        action_taken=1,
        prior_probability=0.2,
    )
    child.visit_count = 5
    child.total_action_value = 3.0

    score, q_value, exploration = selection.calculate_puct_score(
        child, parent.visit_count, mock_mcts_config
    )

    assert q_value == pytest.approx(3.0 / 5.0)
    expected_exploration = (
        mock_mcts_config.puct_coefficient * 0.2 * (math.sqrt(25) / (1 + 5))
    )
    assert exploration == pytest.approx(expected_exploration)
    assert score == pytest.approx(q_value + exploration)


def test_puct_calculation_zero_parent_visits(
    mock_mcts_config: MCTSConfig, mock_env_config: EnvConfig
):
    """Test PUCT score when parent visit count is zero (should use sqrt(1))."""
    parent = Node(state=MockGameState(env_config=mock_env_config))  # type: ignore [arg-type]
    parent.visit_count = 0
    child = Node(
        state=MockGameState(current_step=1, env_config=mock_env_config),  # type: ignore [arg-type]
        parent=parent,
        action_taken=0,
        prior_probability=0.6,
    )
    child.visit_count = 0
    child.total_action_value = 0.0

    score, q_value, exploration = selection.calculate_puct_score(
        child, 0, mock_mcts_config
    )

    assert q_value == 0.0
    expected_exploration = (
        mock_mcts_config.puct_coefficient * 0.6 * (math.sqrt(1) / (1 + 0))
    )
    assert exploration == pytest.approx(expected_exploration)
    assert score == pytest.approx(q_value + exploration)


def test_select_child_node_basic(
    expanded_node_mock_state: Node, mock_mcts_config: MCTSConfig
):
    """Test basic child selection based on PUCT."""
    parent = expanded_node_mock_state
    parent.visit_count = 10

    if 0 not in parent.children or 1 not in parent.children or 2 not in parent.children:
        pytest.skip("Required children (0, 1, 2) not present in fixture")

    child0 = parent.children[0]
    child0.visit_count = 1
    child0.total_action_value = 0.8
    child0.prior_probability = 0.1

    child1 = parent.children[1]
    child1.visit_count = 5
    child1.total_action_value = 0.5
    child1.prior_probability = 0.6

    child2 = parent.children[2]
    child2.visit_count = 3
    child2.total_action_value = 1.5
    child2.prior_probability = 0.3

    selected_child = selection.select_child_node(parent, mock_mcts_config)
    assert selected_child is child0


def test_select_child_node_no_children(
    root_node_mock_state: Node, mock_mcts_config: MCTSConfig
):
    """Test selection raises error if node has no children."""
    parent = root_node_mock_state
    assert not parent.children
    with pytest.raises(selection.SelectionError):
        selection.select_child_node(parent, mock_mcts_config)


def test_select_child_node_tie_breaking(
    expanded_node_mock_state: Node, mock_mcts_config: MCTSConfig
):
    """Test that selection handles ties (implementation detail, usually selects first max)."""
    parent = expanded_node_mock_state
    parent.visit_count = 10

    if 0 not in parent.children or 1 not in parent.children or 2 not in parent.children:
        pytest.skip("Required children (0, 1, 2) not present in fixture")

    child0 = parent.children[0]
    child0.visit_count = 1
    child0.total_action_value = 0.9
    child0.prior_probability = 0.4

    child1 = parent.children[1]
    child1.visit_count = 1
    child1.total_action_value = 0.9
    child1.prior_probability = 0.4

    child2 = parent.children[2]
    child2.visit_count = 5
    child2.total_action_value = 0.1
    child2.prior_probability = 0.1

    selected_child = selection.select_child_node(parent, mock_mcts_config)
    assert selected_child is child0 or selected_child is child1


def test_add_dirichlet_noise(
    expanded_node_mock_state: Node, mock_mcts_config: MCTSConfig
):
    """Test that Dirichlet noise modifies prior probabilities correctly."""
    node = expanded_node_mock_state
    config_copy = mock_mcts_config.model_copy(deep=True)
    config_copy.dirichlet_alpha = 0.5
    config_copy.dirichlet_epsilon = 0.25

    n_children = len(node.children)
    if n_children == 0:
        pytest.skip("Node has no children to add noise to.")
    original_priors = {a: c.prior_probability for a, c in node.children.items()}

    selection.add_dirichlet_noise(node, config_copy)

    new_priors = {a: c.prior_probability for a, c in node.children.items()}
    mixed_sum = sum(new_priors.values())

    assert len(new_priors) == n_children
    priors_changed = False
    for action, new_p in new_priors.items():
        assert action in original_priors
        assert 0.0 <= new_p <= 1.0
        if abs(new_p - original_priors[action]) > 1e-9:
            priors_changed = True

    assert priors_changed
    assert mixed_sum == pytest.approx(1.0, abs=1e-6)


def test_add_dirichlet_noise_disabled(
    expanded_node_mock_state: Node, mock_mcts_config: MCTSConfig
):
    """Test that noise is not added if alpha or epsilon is zero."""
    node = expanded_node_mock_state
    if not node.children:
        pytest.skip("Node has no children.")
    original_priors = {a: c.prior_probability for a, c in node.children.items()}

    config_alpha_zero = mock_mcts_config.model_copy(deep=True)
    config_alpha_zero.dirichlet_alpha = 0.0
    config_alpha_zero.dirichlet_epsilon = 0.25

    config_eps_zero = mock_mcts_config.model_copy(deep=True)
    config_eps_zero.dirichlet_alpha = 0.5
    config_eps_zero.dirichlet_epsilon = 0.0

    selection.add_dirichlet_noise(node, config_alpha_zero)
    priors_after_alpha_zero = {a: c.prior_probability for a, c in node.children.items()}
    assert priors_after_alpha_zero == original_priors

    for a, p in original_priors.items():
        node.children[a].prior_probability = p

    selection.add_dirichlet_noise(node, config_eps_zero)
    priors_after_eps_zero = {a: c.prior_probability for a, c in node.children.items()}
    assert priors_after_eps_zero == original_priors


def test_traverse_to_leaf_unexpanded(
    root_node_mock_state: Node, mock_mcts_config: MCTSConfig
):
    """Test traversal stops immediately at an unexpanded root."""
    leaf, depth = selection.traverse_to_leaf(root_node_mock_state, mock_mcts_config)
    assert leaf is root_node_mock_state
    assert depth == 0


def test_traverse_to_leaf_expanded(
    expanded_node_mock_state: Node, mock_mcts_config: MCTSConfig
):
    """Test traversal selects a child from an expanded node and stops (depth 1)."""
    root = expanded_node_mock_state
    for child in root.children.values():
        assert not child.is_expanded

    leaf, depth = selection.traverse_to_leaf(root, mock_mcts_config)

    assert leaf in root.children.values()
    assert depth == 1


def test_traverse_to_leaf_max_depth(
    expanded_node_mock_state: Node, mock_mcts_config: MCTSConfig
):
    """Test traversal stops at max depth."""
    root = expanded_node_mock_state
    config_copy = mock_mcts_config.model_copy(deep=True)
    config_copy.max_search_depth = 0

    leaf, depth = selection.traverse_to_leaf(root, config_copy)

    assert leaf is root
    assert depth == 0

    config_copy.max_search_depth = 1
    if not root.children:
        pytest.skip("Root node has no children for max depth 1 test")

    child0 = next(iter(root.children.values()))
    child0.children[0] = Node(
        state=MockGameState(current_step=2, env_config=root.state.env_config),  # type: ignore [arg-type]
        parent=child0,
        action_taken=0,
    )

    leaf, depth = selection.traverse_to_leaf(root, config_copy)

    assert leaf in root.children.values()
    assert depth == 1


def test_traverse_to_terminal_leaf(
    expanded_node_mock_state: Node, mock_mcts_config: MCTSConfig
):
    """Test traversal stops at a terminal node."""
    root = expanded_node_mock_state
    if 1 not in root.children:
        pytest.skip("Child 1 not present in fixture")
    child1 = root.children[1]
    mock_child1_state: MockGameState = child1.state  # type: ignore [assignment]
    mock_child1_state._is_over = True

    root.visit_count = 10
    for action, child in root.children.items():
        if action == 1:
            child.visit_count = 5
            child.total_action_value = 4
            child.prior_probability = 0.8
        else:
            child.visit_count = 1
            child.total_action_value = 0
            child.prior_probability = 0.1

    leaf, depth = selection.traverse_to_leaf(root, mock_mcts_config)

    assert leaf is child1
    assert depth == 1


def test_traverse_to_leaf_deeper(
    deep_expanded_node_mock_state: Node, mock_mcts_config: MCTSConfig
):
    """Test traversal goes deeper than 1 level using the specifically configured fixture."""
    root = deep_expanded_node_mock_state
    config_copy = mock_mcts_config.model_copy(deep=True)
    config_copy.max_search_depth = 10

    assert 0 in root.children
    child0 = root.children[0]
    assert child0.is_expanded
    assert child0.children

    mock_child0_state: MockGameState = child0.state  # type: ignore [assignment]
    valid_gc_actions = mock_child0_state.valid_actions()
    if 1 in valid_gc_actions:
        preferred_gc_action = 1
    elif valid_gc_actions:
        preferred_gc_action = valid_gc_actions[0]
    else:
        pytest.fail("Fixture error: Child 0 has no valid actions for grandchildren")

    expected_grandchild = child0.children.get(preferred_gc_action)
    assert expected_grandchild is not None

    leaf, depth = selection.traverse_to_leaf(root, config_copy)

    assert leaf is expected_grandchild
    assert depth == 2


File: __init__.py


