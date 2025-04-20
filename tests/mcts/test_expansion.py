import pytest

# Import EnvConfig from trianglengin
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
