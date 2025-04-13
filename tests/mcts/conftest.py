# File: tests/mcts/conftest.py
import pytest
import numpy as np
from typing import Dict, List, Tuple, Mapping, Optional
import random
import torch
import torch.optim as optim

# Import necessary classes from the source code
import sys
import os

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.environment import GameState, EnvConfig
from src.mcts.core.node import Node

# ADDED: Import ModelConfig and TrainConfig here as they are used in fixtures below
from src.config import MCTSConfig, ModelConfig, TrainConfig
from src.utils.types import (
    ActionType,
    PolicyValueOutput,
    StateType,
    Experience,
)  # Added StateType, Experience
from src.nn import NeuralNetwork, AlphaTriangleNet  # Added NN imports
from src.rl import ExperienceBuffer, Trainer  # Added RL imports


# --- Mock GameState ---
class MockGameState:
    """A simplified mock GameState for testing MCTS logic."""

    def __init__(
        self,
        current_step: int = 0,
        is_terminal: bool = False,
        outcome: float = 0.0,
        valid_actions: Optional[List[ActionType]] = None,
        env_config: Optional[EnvConfig] = None,
    ):
        self.current_step = current_step
        self._is_over = is_terminal
        self._outcome = outcome
        if env_config:
            self.env_config = env_config
        else:
            # Use a smaller default for faster tests if not specified
            self.env_config = EnvConfig(ROWS=3, COLS=3, COLS_PER_ROW=[3, 3, 3])

        default_valid = list(range(self.env_config.ACTION_DIM))
        if valid_actions is None:
            self._valid_actions = default_valid
        else:
            self._valid_actions = [
                a for a in valid_actions if 0 <= a < self.env_config.ACTION_DIM
            ]

    def is_over(self) -> bool:
        return self._is_over

    def get_outcome(self) -> float:
        if not self._is_over:
            raise ValueError("Cannot get outcome of non-terminal state.")
        return self._outcome

    def valid_actions(self) -> List[ActionType]:
        return self._valid_actions

    def copy(self) -> "MockGameState":
        return MockGameState(
            self.current_step,
            self._is_over,
            self._outcome,
            list(self._valid_actions),
            self.env_config,
        )

    def step(
        self, action: ActionType
    ) -> Tuple[float, bool]:  # MODIFIED: Return signature changed
        """
        Simulates taking a step. Returns (reward, done).
        Matches the real GameState.step signature.
        """
        if action not in self.valid_actions():
            raise ValueError(
                f"Invalid action {action} for mock state. Valid: {self.valid_actions()}"
            )
        self.current_step += 1
        # Make terminal condition slightly more complex for testing
        self._is_over = self.current_step >= 10 or len(self._valid_actions) == 0
        self._outcome = 1.0 if self._is_over else 0.0
        # Simulate removing the taken action from valid actions
        if action in self._valid_actions:
            self._valid_actions.remove(action)
        # Simulate removing another random action sometimes
        elif self._valid_actions and random.random() < 0.5:
            self._valid_actions.pop(random.randrange(len(self._valid_actions)))

        # Return dummy reward and the 'done' status
        return 0.0, self._is_over  # MODIFIED: Return only two values

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


# --- Mock Network Evaluator ---
class MockNetworkEvaluator:
    """A mock network evaluator for testing MCTS."""

    def __init__(
        self,
        default_policy: Optional[Mapping[ActionType, float]] = None,
        default_value: float = 0.5,
        action_dim: int = 9,
    ):
        self._default_policy = default_policy
        self._default_value = default_value
        self._action_dim = action_dim
        self.evaluation_history: List[MockGameState] = []
        self.batch_evaluation_history: List[List[MockGameState]] = []

    def _get_policy(self, state: MockGameState) -> Mapping[ActionType, float]:
        if self._default_policy is not None:
            valid_actions = state.valid_actions()
            policy = {
                a: p for a, p in self._default_policy.items() if a in valid_actions
            }
            # Normalize if specified policy doesn't sum to 1 over valid actions
            policy_sum = sum(policy.values())
            if policy_sum > 1e-9 and abs(policy_sum - 1.0) > 1e-6:
                policy = {a: p / policy_sum for a, p in policy.items()}
            elif not policy and valid_actions:  # Handle empty policy for valid actions
                prob = 1.0 / len(valid_actions)
                policy = {a: prob for a in valid_actions}
            return policy

        # Default uniform policy
        valid_actions = state.valid_actions()
        if not valid_actions:
            return {}
        prob = 1.0 / len(valid_actions)
        return {a: prob for a in valid_actions}

    def evaluate(self, state: MockGameState) -> PolicyValueOutput:
        self.evaluation_history.append(state)
        self._action_dim = state.env_config.ACTION_DIM
        policy = self._get_policy(state)
        # Create full policy map respecting action_dim
        full_policy = {a: 0.0 for a in range(self._action_dim)}
        full_policy.update(policy)
        return full_policy, self._default_value

    def evaluate_batch(self, states: List[MockGameState]) -> List[PolicyValueOutput]:
        self.batch_evaluation_history.append(states)
        results = []
        for state in states:
            # Use single evaluate logic for consistency
            results.append(self.evaluate(state))
        return results


# --- Pytest Fixtures ---
# Session-scoped fixtures moved to top-level tests/conftest.py


@pytest.fixture
def mock_evaluator(mock_env_config: EnvConfig) -> MockNetworkEvaluator:
    """Provides a MockNetworkEvaluator instance configured with the mock EnvConfig."""
    return MockNetworkEvaluator(action_dim=mock_env_config.ACTION_DIM)


@pytest.fixture
def root_node_mock_state(mock_env_config: EnvConfig) -> Node:
    """Provides a root Node with a MockGameState using the mock EnvConfig."""
    state = MockGameState(
        valid_actions=list(range(mock_env_config.ACTION_DIM)),
        env_config=mock_env_config,
    )
    return Node(state=state)


@pytest.fixture
def expanded_node_mock_state(
    root_node_mock_state: Node, mock_evaluator: MockNetworkEvaluator
) -> Node:
    """Provides an expanded root node with mock children using mock EnvConfig."""
    root = root_node_mock_state
    mock_evaluator._action_dim = root.state.env_config.ACTION_DIM
    policy, value = mock_evaluator.evaluate(root.state)
    # Ensure policy is not empty before expanding
    if not policy:
        policy = (
            {
                a: 1.0 / len(root.state.valid_actions())
                for a in root.state.valid_actions()
            }
            if root.state.valid_actions()
            else {}
        )

    for action in root.state.valid_actions():
        prior = policy.get(action, 0.0)
        child_state = MockGameState(
            current_step=1, valid_actions=[0, 1], env_config=root.state.env_config
        )
        child = Node(
            state=child_state, parent=root, action_taken=action, prior_probability=prior
        )
        root.children[action] = child
    root.visit_count = 1  # Simulate one visit to root after expansion
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
    mock_evaluator._action_dim = (
        mock_env_config.ACTION_DIM
    )  # Ensure evaluator has correct action dim

    # Ensure children exist
    if 0 not in root.children or 1 not in root.children:
        pytest.skip("Actions 0 or 1 not available in expanded node children")

    # --- Configure Root Node to strongly prefer Action 0 ---
    root.visit_count = 100  # Give root significant visits
    child0 = root.children[0]
    child1 = root.children[1]

    # Child 0: High visit count, good value, high prior (after potential noise)
    child0.visit_count = 80
    child0.total_action_value = 40  # Q = 0.5
    child0.prior_probability = 0.8

    # Other children: Low visits, low value, low prior
    for action, child in root.children.items():
        if action != 0:
            child.visit_count = 2
            child.total_action_value = 0  # Q = 0.0
            child.prior_probability = 0.01

    # --- Configure Child 0 to strongly prefer Action 1 ---
    # Ensure Child 0 has children (expand it manually)
    # Use evaluator to get a policy, then manually create children
    policy_gc, value_gc = mock_evaluator.evaluate(child0.state)
    if not policy_gc:  # Handle case where mock state has no valid actions
        policy_gc = (
            {
                a: 1.0 / len(child0.state.valid_actions())
                for a in child0.state.valid_actions()
            }
            if child0.state.valid_actions()
            else {}
        )

    valid_gc_actions = child0.state.valid_actions()
    if (
        1 not in valid_gc_actions and valid_gc_actions
    ):  # If action 1 not valid, pick first valid one
        preferred_gc_action = valid_gc_actions[0]
    elif not valid_gc_actions:
        pytest.skip("Child 0 has no valid actions to create grandchildren")
    else:
        preferred_gc_action = 1

    # Create grandchild nodes
    for action_gc in valid_gc_actions:
        prior_gc = policy_gc.get(action_gc, 0.0)
        grandchild_state = MockGameState(
            current_step=2, valid_actions=[0], env_config=child0.state.env_config
        )
        grandchild = Node(
            state=grandchild_state,
            parent=child0,
            action_taken=action_gc,
            prior_probability=prior_gc,
        )
        child0.children[action_gc] = grandchild

    # Now configure grandchild stats
    preferred_grandchild = child0.children.get(preferred_gc_action)
    if preferred_grandchild:
        # Preferred Grandchild: High visits, good value, high prior
        preferred_grandchild.visit_count = 60
        preferred_grandchild.total_action_value = 30  # Q = 0.5
        preferred_grandchild.prior_probability = 0.7

    # Other grandchildren: Low visits, low value, low prior
    for action_gc, grandchild in child0.children.items():
        if action_gc != preferred_gc_action:
            grandchild.visit_count = 1
            grandchild.total_action_value = 0  # Q = 0.0
            grandchild.prior_probability = 0.05

    return root


# --- Fixtures for NN/RL Tests (Defined here for potential use in MCTS tests if needed) ---


@pytest.fixture
def mock_state_type(
    mock_model_config: ModelConfig, mock_env_config: EnvConfig
) -> StateType:
    """Creates a mock StateType dictionary with correct shapes."""
    grid_shape = (
        mock_model_config.GRID_INPUT_CHANNELS,
        mock_env_config.ROWS,
        mock_env_config.COLS,
    )
    other_shape = (mock_model_config.OTHER_NN_INPUT_FEATURES_DIM,)
    return {
        "grid": np.random.rand(*grid_shape).astype(np.float32),
        "other_features": np.random.rand(*other_shape).astype(np.float32),
    }


@pytest.fixture
def mock_experience(
    mock_state_type: StateType, mock_env_config: EnvConfig
) -> Experience:
    """Creates a mock Experience tuple."""
    policy_target = (
        {a: 1.0 / mock_env_config.ACTION_DIM for a in range(mock_env_config.ACTION_DIM)}
        if mock_env_config.ACTION_DIM > 0
        else {0: 1.0}
    )
    value_target = random.uniform(-1, 1)
    return (mock_state_type, policy_target, value_target)


@pytest.fixture
def mock_nn_interface(
    mock_model_config: ModelConfig,
    mock_env_config: EnvConfig,
    mock_train_config: TrainConfig,
) -> NeuralNetwork:
    """Provides a NeuralNetwork instance with a mock model for testing."""
    device = torch.device("cpu")  # Use CPU for testing
    nn_interface = NeuralNetwork(
        mock_model_config, mock_env_config, mock_train_config, device
    )
    # Optionally replace internal model with a simpler mock if needed,
    # but using the actual AlphaTriangleNet with simple config is often better.
    return nn_interface


@pytest.fixture
def mock_trainer(
    mock_nn_interface: NeuralNetwork,
    mock_train_config: TrainConfig,
    mock_env_config: EnvConfig,
) -> Trainer:
    """Provides a Trainer instance."""
    return Trainer(mock_nn_interface, mock_train_config, mock_env_config)


@pytest.fixture
def mock_optimizer(mock_trainer: Trainer) -> optim.Optimizer:
    """Provides the optimizer from the mock_trainer."""
    return mock_trainer.optimizer


@pytest.fixture
def mock_experience_buffer(mock_train_config: TrainConfig) -> ExperienceBuffer:
    """Provides an ExperienceBuffer instance."""
    return ExperienceBuffer(mock_train_config)


@pytest.fixture
def filled_mock_buffer(
    mock_experience_buffer: ExperienceBuffer, mock_experience: Experience
) -> ExperienceBuffer:
    """Provides a buffer filled with some mock experiences."""
    for i in range(mock_experience_buffer.min_size_to_train + 5):
        # Create slightly different experiences
        state_copy = {k: v.copy() for k, v in mock_experience[0].items()}
        # Ensure grid is writeable before modifying
        if not state_copy["grid"].flags.writeable:
            state_copy["grid"] = state_copy["grid"].copy()
        state_copy["grid"] += (
            np.random.randn(*state_copy["grid"].shape).astype(np.float32) * 0.1
        )
        exp_copy = (state_copy, mock_experience[1], random.uniform(-1, 1))
        mock_experience_buffer.add(exp_copy)
    return mock_experience_buffer
