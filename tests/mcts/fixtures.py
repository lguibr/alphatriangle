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
