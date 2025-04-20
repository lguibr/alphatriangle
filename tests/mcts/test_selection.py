# File: tests/mcts/test_selection.py
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
    # Convert set to list before checking/indexing
    valid_gc_actions_list = list(valid_gc_actions)
    if 1 in valid_gc_actions_list:
        preferred_gc_action = 1
    elif valid_gc_actions_list:
        # Index the list, not the set
        preferred_gc_action = valid_gc_actions_list[0]
    else:
        pytest.fail("Fixture error: Child 0 has no valid actions for grandchildren")

    expected_grandchild = child0.children.get(preferred_gc_action)
    assert expected_grandchild is not None

    leaf, depth = selection.traverse_to_leaf(root, config_copy)

    assert leaf is expected_grandchild
    assert depth == 2
