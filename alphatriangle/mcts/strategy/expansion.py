import logging
from typing import TYPE_CHECKING

# Import GameState from trianglengin
# Keep alphatriangle utils types for now
from ..core.node import Node
from ..core.types import ActionPolicyMapping

if TYPE_CHECKING:
    from ...utils.types import ActionType

logger = logging.getLogger(__name__)


def expand_node_with_policy(node: Node, action_policy: ActionPolicyMapping):
    """
    Expands a node by creating children for valid actions using the
    pre-computed action policy priors from the network.
    Assumes the node is not terminal and not already expanded.
    Marks the node's state as game_over if no valid actions are found.
    """
    if node.is_expanded:
        logger.debug(f"[Expand] Attempted to expand an already expanded node: {node}")
        return
    # Use is_over() method from trianglengin.GameState
    if node.state.is_over():
        logger.warning(f"[Expand] Attempted to expand a terminal node: {node}")
        return

    logger.debug(f"[Expand] Expanding Node: {node}")

    # Use valid_actions() method from trianglengin.GameState
    valid_actions: set[ActionType] = node.state.valid_actions()
    logger.debug(
        f"[Expand] Found {len(valid_actions)} valid actions for state step {node.state.current_step}."
    )
    logger.debug(
        f"[Expand] Received action policy (first 5): {list(action_policy.items())[:5]}"
    )

    if not valid_actions:
        logger.warning(
            f"[Expand] Expanding node at step {node.state.current_step} with no valid actions but not terminal? Marking state as game over."
        )
        # Use force_game_over method from trianglengin.GameState
        node.state.force_game_over("Expansion found no valid actions")
        return

    children_created = 0
    for action in valid_actions:
        prior = action_policy.get(action, 0.0)
        if prior < 0.0:
            logger.warning(
                f"[Expand] Received negative prior ({prior}) for action {action}. Clamping to 0."
            )
            prior = 0.0
        elif prior == 0.0:
            logger.debug(
                f"[Expand] Valid action {action} received prior=0 from network."
            )

        # Use copy() method from trianglengin.GameState
        next_state_copy = node.state.copy()
        try:
            # Use step() method from trianglengin.GameState
            _, done = next_state_copy.step(action)
        except Exception as e:
            logger.error(
                f"[Expand] Error stepping state for child node expansion (action {action}): {e}",
                exc_info=True,
            )
            continue

        child = Node(
            state=next_state_copy,  # Already a trianglengin.GameState
            parent=node,
            action_taken=action,
            prior_probability=prior,
        )
        node.children[action] = child
        logger.debug(
            f"  [Expand] Created Child Node: Action={action}, Prior={prior:.4f}, StateStep={next_state_copy.current_step}, Done={done}"
        )
        children_created += 1

    logger.debug(f"[Expand] Expanded node {node} with {children_created} children.")
