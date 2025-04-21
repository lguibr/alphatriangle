# File: alphatriangle/rl/self_play/mcts_helpers.py
import logging
import random

import numpy as np

from ...utils.types import ActionType, PolicyTargetMapping

logger = logging.getLogger(__name__)
rng = np.random.default_rng()


class PolicyGenerationError(Exception):
    """Custom exception for errors during policy generation or action selection from visit counts."""

    pass


def select_action_from_visits(
    visit_counts: dict[ActionType, int], temperature: float
) -> ActionType:
    """
    Selects an action based on visit counts and temperature.
    Operates on the dictionary returned by trimcts.run_mcts.
    Raises PolicyGenerationError if selection is not possible.
    """
    if not visit_counts:
        raise PolicyGenerationError(
            "Cannot select action: Visit counts dictionary is empty."
        )

    actions = list(visit_counts.keys())
    counts = np.array(list(visit_counts.values()), dtype=np.float64)

    total_visits = np.sum(counts)
    logger.debug(
        f"[PolicySelect] Selecting action from visits. Total visits: {total_visits}. Num actions: {len(actions)}"
    )

    if total_visits == 0:
        logger.warning(
            "[PolicySelect] Total visit count is zero. Selecting uniformly from available actions."
        )
        selected_action = random.choice(actions)
        logger.debug(
            f"[PolicySelect] Uniform random action selected: {selected_action}"
        )
        return selected_action

    if temperature == 0.0:
        max_visits = np.max(counts)
        logger.debug(
            f"[PolicySelect] Greedy selection (temp=0). Max visits: {max_visits}"
        )
        best_action_indices = np.where(counts == max_visits)[0]
        # Removed redundant log: logger.debug(f"[PolicySelect] Greedy selection. Best action indices: {best_action_indices}")
        chosen_index = random.choice(best_action_indices)
        selected_action = actions[chosen_index]
        logger.debug(f"[PolicySelect] Greedy action selected: {selected_action}")
        return selected_action
    else:
        logger.debug(f"[PolicySelect] Probabilistic selection: Temp={temperature:.4f}")
        # Removed print: logger.debug(f"  Visit Counts: {counts}")
        # Use counts directly, avoid log for stability if counts can be zero
        # Ensure counts are positive before raising to power
        powered_counts = np.maximum(counts, 1e-9) ** (1.0 / temperature)
        sum_powered_counts = np.sum(powered_counts)

        if sum_powered_counts < 1e-9 or not np.isfinite(sum_powered_counts):
            raise PolicyGenerationError(
                f"Could not normalize visit probabilities (sum={sum_powered_counts}). Visits: {counts}"
            )
        else:
            probabilities = powered_counts / sum_powered_counts

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

        # Removed print: logger.debug(f"  Final Probabilities (normalized): {probabilities}")
        # Removed print: logger.debug(f"  Final Probabilities Sum: {np.sum(probabilities):.6f}")

        try:
            selected_action = rng.choice(actions, p=probabilities)
            logger.debug(
                f"[PolicySelect] Sampled action (temp={temperature:.2f}): {selected_action}"
            )
            return int(selected_action)
        except ValueError as e:
            raise PolicyGenerationError(
                f"Error during np.random.choice: {e}. Probs: {probabilities}, Sum: {np.sum(probabilities)}"
            ) from e


def get_policy_target_from_visits(
    visit_counts: dict[ActionType, int], action_dim: int, temperature: float = 1.0
) -> PolicyTargetMapping:
    """
    Calculates the policy target distribution based on MCTS visit counts.
    Operates on the dictionary returned by trimcts.run_mcts.
    Raises PolicyGenerationError if target cannot be generated.
    """
    full_target = dict.fromkeys(range(action_dim), 0.0)

    if not visit_counts:
        logger.warning(
            "[PolicyTarget] Cannot compute policy target: Visit counts dictionary is empty."
        )
        return full_target

    actions = list(visit_counts.keys())
    counts = np.array(list(visit_counts.values()), dtype=np.float64)
    total_visits = np.sum(counts)

    if total_visits == 0:
        logger.warning(
            "[PolicyTarget] Cannot compute policy target: Total visits is zero."
        )
        return full_target

    if temperature == 0.0:
        max_visits = np.max(counts)
        if max_visits == 0:
            logger.warning(
                "[PolicyTarget] Temperature is 0 but max visits is 0. Returning zero target."
            )
            return full_target

        best_actions = [actions[i] for i, v in enumerate(counts) if v == max_visits]
        prob = 1.0 / len(best_actions)
        for a in best_actions:
            if 0 <= a < action_dim:
                full_target[a] = prob
            else:
                logger.warning(
                    f"[PolicyTarget] Best action {a} is out of bounds ({action_dim}). Skipping."
                )
    else:
        # Use counts directly, avoid potential issues with log(0)
        powered_counts = np.maximum(counts, 1e-9) ** (1.0 / temperature)
        sum_powered_counts = np.sum(powered_counts)

        if sum_powered_counts < 1e-9 or not np.isfinite(sum_powered_counts):
            raise PolicyGenerationError(
                f"Could not normalize policy target probabilities (sum={sum_powered_counts}). Visits: {counts}"
            )

        probabilities = powered_counts / sum_powered_counts
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
                    f"[PolicyTarget] Action {action} from MCTS is out of bounds ({action_dim}). Skipping."
                )

    final_sum = sum(full_target.values())
    if abs(final_sum - 1.0) > 1e-5:
        # Keep this error log as it indicates a potential problem
        logger.error(
            f"[PolicyTarget] Final policy target does not sum to 1 ({final_sum:.6f}). Target: {full_target}"
        )

    return full_target
