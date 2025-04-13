# File: tests/mcts/fixtures.py
from collections.abc import Mapping

import pytest

from src.config import MCTSConfig

# Import necessary classes from the source code
from src.environment import EnvConfig
from src.mcts.core.node import Node
from src.utils.types import ActionType, PolicyValueOutput


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
        self._valid_actions = valid_actions if valid_actions is not None else [0, 1, 2]
        # Use a default EnvConfig if none provided, needed for action dim
        self.env_config = env_config if env_config else EnvConfig(ROWS=3, COLS=3)

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

    def step(self, action: ActionType) -> tuple[float, float, bool]:
        # Mock step: advances step, returns dummy values, becomes terminal sometimes
        if action not in self._valid_actions:
            raise ValueError(f"Invalid action {action} for mock state.")
        self.current_step += 1
        # Simple logic: become terminal after 5 steps for testing
        self._is_over = self.current_step >= 5
        self._outcome = 1.0 if self._is_over else 0.0
        return 0.0, 0.0, self._is_over  # placeholder_val, reward, done

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
        self._action_dim = action_dim
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
    # Simple 3x3 grid for easier testing
    return EnvConfig(ROWS=3, COLS=3, COLS_PER_ROW=[3, 3, 3], NUM_SHAPE_SLOTS=1)


@pytest.fixture
def mock_mcts_config() -> MCTSConfig:
    """Provides a default MCTSConfig for tests."""
    return MCTSConfig(
        num_simulations=10,  # Fewer sims for testing
        puct_coefficient=1.0,
        temperature_initial=1.0,
        temperature_final=0.1,
        temperature_anneal_steps=5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        max_search_depth=10,
    )


@pytest.fixture
def mock_evaluator(mock_env_config: EnvConfig) -> MockNetworkEvaluator:
    """Provides a MockNetworkEvaluator instance."""
    return MockNetworkEvaluator(action_dim=mock_env_config.ACTION_DIM)


@pytest.fixture
def root_node_mock_state(mock_env_config: EnvConfig) -> Node:
    """Provides a root Node with a MockGameState."""
    state = MockGameState(valid_actions=[0, 1, 2], env_config=mock_env_config)
    return Node(state=state)


@pytest.fixture
def expanded_node_mock_state(
    root_node_mock_state: Node, mock_evaluator: MockNetworkEvaluator
) -> Node:
    """Provides an expanded root node with mock children."""
    root = root_node_mock_state
    policy, value = mock_evaluator.evaluate(root.state)
    # Manually expand for testing setup
    for action in root.state.valid_actions():
        prior = policy.get(action, 0.0)
        # Create mock child state (doesn't need to be accurate step)
        child_state = MockGameState(
            current_step=1, valid_actions=[0, 1], env_config=root.state.env_config
        )
        child = Node(
            state=child_state, parent=root, action_taken=action, prior_probability=prior
        )
        root.children[action] = child
    # Simulate one backpropagation
    root.visit_count = 1
    root.total_action_value = value
    return root
