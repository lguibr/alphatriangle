from typing import Any

import pytest

from alphatriangle.mcts.core.node import Node

# Import necessary components and fixtures
from alphatriangle.mcts.strategy import expansion

# Import session-scoped fixtures implicitly via pytest injection
# from alphatriangle.config import MCTSConfig # REMOVED - Provided by top-level conftest
from .conftest import (  # Import from conftest (local fixtures)
    # mock_env_config, # REMOVED - Provided by top-level conftest
    MockGameState,
)


def test_expand_node_with_policy_basic(root_node_mock_state: Node):
    """Test basic node expansion with a valid policy."""
    node = root_node_mock_state
    # Cast node.state to Any temporarily to access mock method
    mock_state: Any = node.state
    valid_actions = mock_state.valid_actions()
    # Simple policy: uniform over valid actions
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
        assert (
            child.state.current_step == node.state.current_step + 1
        )  # Check state stepped
        assert not child.is_expanded
        assert child.visit_count == 0
        assert child.total_action_value == 0.0


def test_expand_node_with_policy_partial(root_node_mock_state: Node):
    """Test expansion when policy doesn't cover all valid actions (should assign 0 prior)."""
    node = root_node_mock_state
    # Cast node.state to Any temporarily to access mock method
    mock_state: Any = node.state
    valid_actions = mock_state.valid_actions()  # e.g., [0, 1, ..., 8] for 3x3
    # Policy only covers action 0 and 1
    policy = {0: 0.6, 1: 0.4}

    expansion.expand_node_with_policy(node, policy)

    assert node.is_expanded
    assert len(node.children) == len(
        valid_actions
    )  # Should still create nodes for all valid actions

    assert 0 in node.children
    assert node.children[0].prior_probability == pytest.approx(0.6)
    assert 1 in node.children
    assert node.children[1].prior_probability == pytest.approx(0.4)
    # Check an action not in the policy but valid
    if 2 in valid_actions:
        assert 2 in node.children
        assert node.children[2].prior_probability == 0.0  # Prior should default to 0


def test_expand_node_with_policy_empty_valid_actions(root_node_mock_state: Node):
    """Test expansion when the node's state has no valid actions (but isn't terminal yet)."""
    node = root_node_mock_state
    # Cast node.state to Any temporarily to access mock attribute
    mock_state: Any = node.state
    mock_state._valid_actions = []  # No valid actions
    policy = {0: 1.0}  # Policy doesn't matter here

    expansion.expand_node_with_policy(node, policy)

    assert not node.is_expanded  # Should not expand
    assert not node.children
    # The function should log a warning in this case
    # The node's state should be marked as terminal by the expansion function
    assert node.state.is_over()


def test_expand_node_with_policy_already_expanded(root_node_mock_state: Node):
    """Test that expanding an already expanded node does nothing."""
    node = root_node_mock_state
    policy = {0: 1.0}
    # Manually add a child to simulate expansion
    # Pass the env_config from the root node's state
    node.children[0] = Node(
        state=MockGameState(current_step=1, env_config=node.state.env_config),  # type: ignore [arg-type]
        parent=node,
        action_taken=0,
    )

    assert node.is_expanded
    original_children = node.children.copy()

    expansion.expand_node_with_policy(node, policy)

    assert node.children == original_children  # Children should not change


def test_expand_node_with_policy_terminal_node(root_node_mock_state: Node):
    """Test that expanding a terminal node does nothing."""
    node = root_node_mock_state
    # Cast node.state to Any temporarily to access mock attribute
    mock_state: Any = node.state
    mock_state._is_over = True  # Mark as terminal
    policy = {0: 1.0}

    assert not node.is_expanded
    expansion.expand_node_with_policy(node, policy)
    assert not node.is_expanded  # Should not expand


def test_expand_node_with_invalid_policy_content(root_node_mock_state: Node):
    """Test expansion handles policy with invalid content (e.g., negative priors)."""
    # Note: The main search loop should validate policy *before* calling expand.
    # This test checks if expand handles it defensively (it currently clamps).
    node = root_node_mock_state
    # Cast node.state to Any temporarily to access mock method
    mock_state: Any = node.state
    valid_actions = mock_state.valid_actions()
    policy = {0: 1.5, 1: -0.5}  # Invalid priors

    expansion.expand_node_with_policy(node, policy)

    assert node.is_expanded
    assert len(node.children) == len(valid_actions)
    assert node.children[0].prior_probability == pytest.approx(
        1.5
    )  # Currently doesn't clamp > 1
    assert node.children[1].prior_probability == 0.0  # Clamps negative to 0
    if 2 in valid_actions:
        assert node.children[2].prior_probability == 0.0
