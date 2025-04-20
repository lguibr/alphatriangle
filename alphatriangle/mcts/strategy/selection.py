# File: alphatriangle/mcts/strategy/selection.py
import logging
import math

import numpy as np

# Import GameState from trianglengin (only needed for type hint if used)
# from trianglengin.core.environment import GameState
# Keep alphatriangle imports
from ...config import MCTSConfig
from ..core.node import Node

logger = logging.getLogger(__name__)
rng = np.random.default_rng()


class SelectionError(Exception):
    """Custom exception for errors during node selection."""

    pass


def calculate_puct_score(
    child_node: Node,
    parent_visit_count: int,
    config: MCTSConfig,
) -> tuple[float, float, float]:
    """Calculates the PUCT score and its components for a child node."""
    q_value = child_node.value_estimate
    prior = child_node.prior_probability
    child_visits = child_node.visit_count
    parent_visits_sqrt = math.sqrt(max(1, parent_visit_count))

    exploration_term = (
        config.puct_coefficient * prior * (parent_visits_sqrt / (1 + child_visits))
    )
    score = q_value + exploration_term

    if not np.isfinite(score):
        logger.warning(
            f"Non-finite PUCT score calculated (Q={q_value}, P={prior}, ChildN={child_visits}, ParentN={parent_visit_count}, Exp={exploration_term}). Defaulting to Q-value."
        )
        score = q_value
        exploration_term = 0.0

    return score, q_value, exploration_term


def add_dirichlet_noise(node: Node, config: MCTSConfig):
    """Adds Dirichlet noise to the prior probabilities of the children of this node."""
    if (
        config.dirichlet_alpha <= 0.0
        or config.dirichlet_epsilon <= 0.0
        or not node.children
        or len(node.children) <= 1
    ):
        return

    actions = list(node.children.keys())
    noise = rng.dirichlet([config.dirichlet_alpha] * len(actions))
    eps = config.dirichlet_epsilon

    noisy_priors_sum = 0.0
    for i, action in enumerate(actions):
        child = node.children[action]
        original_prior = child.prior_probability
        child.prior_probability = (1 - eps) * child.prior_probability + eps * noise[i]
        noisy_priors_sum += child.prior_probability
        logger.debug(
            f"  [Noise] Action {action}: OrigP={original_prior:.4f}, Noise={noise[i]:.4f} -> NewP={child.prior_probability:.4f}"
        )

    if abs(noisy_priors_sum - 1.0) > 1e-6:
        logger.debug(
            f"Re-normalizing priors after Dirichlet noise (Sum={noisy_priors_sum:.6f})"
        )
        for action in actions:
            if noisy_priors_sum > 1e-9:
                node.children[action].prior_probability /= noisy_priors_sum
            else:
                logger.warning(
                    "Sum of priors after noise is near zero. Cannot normalize."
                )
                node.children[action].prior_probability = 0.0

    logger.debug(
        f"[Noise] Added Dirichlet noise (alpha={config.dirichlet_alpha}, eps={eps}) to {len(actions)} root node priors."
    )


def select_child_node(node: Node, config: MCTSConfig) -> Node:
    """
    Selects the child node with the highest PUCT score. Assumes noise already added if root.
    Raises SelectionError if no valid child can be selected.
    Includes detailed logging of all child scores if DEBUG level is enabled.
    """
    if not node.children:
        raise SelectionError(f"Cannot select child from node {node} with no children.")

    best_score = -float("inf")
    best_child: Node | None = None
    child_scores_log = []

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"  [Select] Selecting child for Node (Visits={node.visit_count}, Children={len(node.children)}, StateStep={node.state.current_step}):"
        )

    parent_visit_count = node.visit_count

    for action, child in node.children.items():
        score, q, exp_term = calculate_puct_score(child, parent_visit_count, config)

        if logger.isEnabledFor(logging.DEBUG):
            log_entry = (
                f"    Act={action}, Score={score:.4f} "
                f"(Q={q:.3f}, P={child.prior_probability:.4f}, N={child.visit_count}, Exp={exp_term:.4f})"
            )
            child_scores_log.append(log_entry)

        if not np.isfinite(score):
            logger.warning(
                f"    [Select] Non-finite PUCT score ({score}) calculated for child action {action}. Skipping."
            )
            continue

        if score > best_score:
            best_score = score
            best_child = child

    if logger.isEnabledFor(logging.DEBUG) and child_scores_log:
        try:

            def get_score_from_log(log_str):
                parts = log_str.split(",")
                for part in parts:
                    if "Score=" in part:
                        return float(part.split("=")[1].split(" ")[0])
                return -float("inf")

            child_scores_log.sort(key=get_score_from_log, reverse=True)
        except Exception as sort_err:
            logger.warning(f"Could not sort child score logs: {sort_err}")
        logger.debug("    [Select] All Child Scores Considered (Top 5):")
        for log_line in child_scores_log[:5]:
            logger.debug(f"      {log_line}")

    if best_child is None:
        child_details = [
            f"Act={a}, N={c.visit_count}, P={c.prior_probability:.4f}, Q={c.value_estimate:.3f}"
            for a, c in node.children.items()
        ]
        logger.error(
            f"Could not select best child for node step {node.state.current_step}. Child details: {child_details}"
        )
        raise SelectionError(
            f"Could not select best child for node step {node.state.current_step}. Check scores and children."
        )

    logger.debug(
        f"  [Select] --> Selected Child: Action {best_child.action_taken}, Score {best_score:.4f}, Q-value {best_child.value_estimate:.3f}"
    )
    return best_child


def traverse_to_leaf(root_node: Node, config: MCTSConfig) -> tuple[Node, int]:
    """
    Traverses the tree from root to a leaf node using PUCT selection.
    A leaf is defined as a node that is not expanded OR is terminal.
    Stops also if the maximum search depth has been reached.
    Raises SelectionError if child selection fails during traversal.
    Returns the leaf node and the depth reached.
    """
    current_node = root_node
    depth = 0
    logger.debug(f"[Traverse] --- Start Traverse (Root Node: {root_node}) ---")
    stop_reason = "Unknown"

    while True:
        logger.debug(
            f"  [Traverse] Depth {depth}: Considering Node: {current_node} (Expanded={current_node.is_expanded}, Terminal={current_node.state.is_over()})"
        )

        # Use is_over() method from trianglengin.GameState
        if current_node.state.is_over():
            stop_reason = "Terminal State"
            logger.debug(
                f"  [Traverse] Depth {depth}: Node is TERMINAL. Stopping traverse."
            )
            break
        if not current_node.is_expanded:
            stop_reason = "Unexpanded Leaf"
            logger.debug(
                f"  [Traverse] Depth {depth}: Node is LEAF (not expanded). Stopping traverse."
            )
            break
        if config.max_search_depth is not None and depth >= config.max_search_depth:
            stop_reason = "Max Depth Reached"
            logger.debug(
                f"  [Traverse] Depth {depth}: Hit MAX DEPTH ({config.max_search_depth}). Stopping traverse."
            )
            break

        try:
            selected_child = select_child_node(current_node, config)
            logger.debug(
                f"  [Traverse] Depth {depth}: Selected child with action {selected_child.action_taken}"
            )
            current_node = selected_child
            depth += 1
        except SelectionError as e:
            stop_reason = f"Child Selection Error: {e}"
            logger.error(
                f"  [Traverse] Depth {depth}: Error during child selection: {e}. Breaking traverse.",
                exc_info=False,
            )
            logger.warning(
                f"  [Traverse] Returning node {current_node} due to SelectionError."
            )
            break
        except Exception as e:
            stop_reason = f"Unexpected Error: {e}"
            logger.error(
                f"  [Traverse] Depth {depth}: Unexpected error during child selection: {e}. Breaking traverse.",
                exc_info=True,
            )
            logger.warning(
                f"  [Traverse] Returning node {current_node} due to Unexpected Error."
            )
            break

    logger.debug(
        f"[Traverse] --- End Traverse: Reached Node at Depth {depth}. Reason: {stop_reason}. Final Node: {current_node} ---"
    )
    return current_node, depth
