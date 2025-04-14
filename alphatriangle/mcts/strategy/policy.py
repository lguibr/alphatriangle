import logging
import random

import numpy as np

from ...utils.types import ActionType
from ..core.node import Node
from ..core.types import ActionPolicyMapping

logger = logging.getLogger(__name__)
rng = np.random.default_rng()


class PolicyGenerationError(Exception):
    """Custom exception for errors during policy generation or action selection."""

    pass


def select_action_based_on_visits(root_node: Node, temperature: float) -> ActionType:
    """
    Selects an action from the root node based on visit counts and temperature.
    Raises PolicyGenerationError if selection is not possible.
    """
    if not root_node.children:
        raise PolicyGenerationError(
            f"Cannot select action: Root node (Step {root_node.state.current_step}) has no children."
        )

    actions = list(root_node.children.keys())
    visit_counts = np.array(
        [root_node.children[action].visit_count for action in actions],
        dtype=np.float64,
    )

    if len(actions) == 0:
        raise PolicyGenerationError(
            f"Cannot select action: No actions available in children for root node (Step {root_node.state.current_step})."
        )

    total_visits = np.sum(visit_counts)
    logger.debug(
        f"[PolicySelect] Selecting action for node step {root_node.state.current_step}. Total child visits: {total_visits}. Num children: {len(actions)}"
    )

    if total_visits == 0:
        logger.warning(
            f"[PolicySelect] Total visit count for children is zero at root node (Step {root_node.state.current_step}). MCTS might have failed. Selecting uniformly."
        )
        selected_action = random.choice(actions)
        logger.debug(
            f"[PolicySelect] Uniform random action selected: {selected_action}"
        )
        return selected_action

    if temperature == 0.0:
        max_visits = np.max(visit_counts)
        logger.debug(
            f"[PolicySelect] Greedy selection (temp=0). Max visits: {max_visits}"
        )
        best_action_indices = np.where(visit_counts == max_visits)[0]
        logger.debug(
            f"[PolicySelect] Greedy selection. Best action indices: {best_action_indices}"
        )
        # Use standard library random for tie-breaking
        chosen_index = random.choice(best_action_indices)
        selected_action = actions[chosen_index]
        logger.debug(f"[PolicySelect] Greedy action selected: {selected_action}")
        return selected_action

    else:
        logger.debug(f"[PolicySelect] Probabilistic selection: Temp={temperature:.4f}")
        logger.debug(f"  Visit Counts: {visit_counts}")
        log_visits = np.log(np.maximum(visit_counts, 1e-9))
        scaled_log_visits = log_visits / temperature
        scaled_log_visits -= np.max(scaled_log_visits)
        probabilities = np.exp(scaled_log_visits)
        sum_probs = np.sum(probabilities)

        if sum_probs < 1e-9 or not np.isfinite(sum_probs):
            raise PolicyGenerationError(
                f"Could not normalize visit probabilities (sum={sum_probs}). Visits: {visit_counts}"
            )
        else:
            probabilities /= sum_probs

        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
            raise PolicyGenerationError(
                f"Invalid probabilities generated after normalization: {probabilities}"
            )
        if abs(np.sum(probabilities) - 1.0) > 1e-5:
            logger.warning(
                f"[PolicySelect] Probabilities sum to {np.sum(probabilities):.6f} after normalization. Attempting re-normalization."
            )
            probabilities /= np.sum(probabilities)
            if abs(np.sum(probabilities) - 1.0) > 1e-5:
                raise PolicyGenerationError(
                    f"Probabilities still do not sum to 1 after re-normalization: {probabilities}, Sum: {np.sum(probabilities)}"
                )

        logger.debug(f"  Final Probabilities (normalized): {probabilities}")
        logger.debug(f"  Final Probabilities Sum: {np.sum(probabilities):.6f}")

        try:
            # Use NumPy's default_rng for weighted choice
            selected_action = rng.choice(actions, p=probabilities)
            logger.debug(
                f"[PolicySelect] Sampled action (temp={temperature:.2f}): {selected_action}"
            )
            # Ensure return type is ActionType (int)
            return int(selected_action)
        except ValueError as e:
            raise PolicyGenerationError(
                f"Error during np.random.choice: {e}. Probs: {probabilities}, Sum: {np.sum(probabilities)}"
            ) from e


def get_policy_target(root_node: Node, temperature: float = 1.0) -> ActionPolicyMapping:
    """
    Calculates the policy target distribution based on MCTS visit counts.
    Raises PolicyGenerationError if target cannot be generated.
    """
    action_dim = int(root_node.state.env_config.ACTION_DIM)  # type: ignore[call-overload]
    full_target = dict.fromkeys(range(action_dim), 0.0)

    if not root_node.children or root_node.visit_count == 0:
        logger.warning(
            f"[PolicyTarget] Cannot compute policy target: Root node (Step {root_node.state.current_step}) has no children or zero visits. Returning zero target."
        )
        return full_target

    child_visits = {
        action: child.visit_count for action, child in root_node.children.items()
    }
    actions = list(child_visits.keys())
    visits = np.array(list(child_visits.values()), dtype=np.float64)
    total_visits = np.sum(visits)

    if not actions:
        logger.warning(
            "[PolicyTarget] Cannot compute policy target: No actions found in children."
        )
        return full_target

    if temperature == 0.0:
        max_visits = np.max(visits)
        if max_visits == 0:
            logger.warning(
                "[PolicyTarget] Temperature is 0 but max visits is 0. Returning zero target."
            )
            return full_target

        best_actions = [actions[i] for i, v in enumerate(visits) if v == max_visits]
        prob = 1.0 / len(best_actions)
        for a in best_actions:
            if 0 <= a < action_dim:
                full_target[a] = prob
            else:
                logger.warning(
                    f"[PolicyTarget] Best action {a} is out of bounds ({action_dim}). Skipping."
                )

    else:
        visit_probs = visits ** (1.0 / temperature)
        sum_probs = np.sum(visit_probs)

        if sum_probs < 1e-9 or not np.isfinite(sum_probs):
            raise PolicyGenerationError(
                f"Could not normalize policy target probabilities (sum={sum_probs}). Visits: {visits}"
            )

        probabilities = visit_probs / sum_probs
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
            raise PolicyGenerationError(
                f"Invalid probabilities generated for policy target: {probabilities}"
            )
        if abs(np.sum(probabilities) - 1.0) > 1e-5:
            logger.warning(
                f"[PolicyTarget] Target probabilities sum to {np.sum(probabilities):.6f}. Re-normalizing."
            )
            probabilities /= np.sum(probabilities)
            if abs(np.sum(probabilities) - 1.0) > 1e-5:
                raise PolicyGenerationError(
                    f"Target probabilities still do not sum to 1 after re-normalization: {probabilities}, Sum: {np.sum(probabilities)}"
                )

        raw_policy = {action: probabilities[i] for i, action in enumerate(actions)}
        for action, prob in raw_policy.items():
            if 0 <= action < action_dim:
                full_target[action] = prob
            else:
                logger.warning(
                    f"[PolicyTarget] Action {action} from MCTS children is out of bounds ({action_dim}). Skipping."
                )

    final_sum = sum(full_target.values())
    if abs(final_sum - 1.0) > 1e-5 and total_visits > 0:
        logger.error(
            f"[PolicyTarget] Final policy target does not sum to 1 ({final_sum:.6f}). Target: {full_target}"
        )

    return full_target
