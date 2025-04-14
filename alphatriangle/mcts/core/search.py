# File: src/mcts/core/search.py
import logging
import time

import numpy as np

# Use relative imports
from ...config import MCTSConfig
from ..strategy import backpropagation, expansion, selection
from .node import Node
from .types import ActionPolicyValueEvaluator

logger = logging.getLogger(__name__)

MCTS_BATCH_SIZE = 8


class MCTSExecutionError(Exception):
    """Custom exception for errors during MCTS execution."""

    pass


def run_mcts_simulations(
    root_node: Node,
    config: MCTSConfig,
    network_evaluator: ActionPolicyValueEvaluator,
) -> int:
    """
    Runs the specified number of MCTS simulations from the root node.
    Uses BATCHED evaluation of leaf nodes but expands immediately after selection.
    Raises MCTSExecutionError if critical errors occur.

    Returns:
        The maximum tree depth reached during the simulations.
    """
    if root_node.state.is_over():
        logger.warning("[MCTS] MCTS started on a terminal state. No simulations run.")
        return 0

    max_depth_overall = 0
    sim_success_count = 0
    sim_error_count = 0
    eval_error_count = 0

    if not root_node.is_expanded:
        logger.debug("[MCTS] Root node not expanded, performing initial evaluation...")
        try:
            action_policy, root_value = network_evaluator.evaluate(root_node.state)
            if not isinstance(action_policy, dict) or not action_policy:
                raise MCTSExecutionError(
                    f"Initial evaluation returned invalid or empty policy format: {type(action_policy)}"
                )
            if not all(
                isinstance(k, int)
                and isinstance(v, float)
                and np.isfinite(v)
                and v >= 0
                for k, v in action_policy.items()
            ):
                raise MCTSExecutionError(
                    f"Initial evaluation returned invalid policy content: {action_policy}"
                )
            if not isinstance(root_value, float) or not np.isfinite(root_value):
                raise MCTSExecutionError(
                    f"Initial evaluation returned invalid value: {root_value}"
                )

            logger.debug(f"[MCTS] Initial root policy priors from NN: {action_policy}")
            expansion.expand_node_with_policy(root_node, action_policy)

            if not root_node.is_expanded and not root_node.state.is_over():
                raise MCTSExecutionError(
                    "Initial root expansion failed unexpectedly (no children, not terminal)."
                )

            if root_node.is_expanded or root_node.state.is_over():
                depth_bp = backpropagation.backpropagate_value(root_node, root_value)
                max_depth_overall = max(max_depth_overall, depth_bp)
                logger.debug(
                    f"[MCTS] Initial root expansion/backprop complete. Value: {root_value:.3f}, Depth: {depth_bp}"
                )
                selection.add_dirichlet_noise(root_node, config)
            else:
                logger.warning(
                    "[MCTS] Initial root expansion did not result in children and state is not terminal. Check expansion logic or valid actions."
                )

        except Exception as e:
            logger.error(
                f"[MCTS] Initial root evaluation/expansion failed: {e}", exc_info=True
            )
            raise MCTSExecutionError(
                f"Initial root evaluation/expansion failed: {e}"
            ) from e

    elif root_node.visit_count == 0:
        logger.warning(
            "[MCTS] Root node expanded but visit_count is 0. Backpropagating current estimate and applying noise."
        )
        depth_bp = backpropagation.backpropagate_value(
            root_node, root_node.value_estimate
        )
        max_depth_overall = max(max_depth_overall, depth_bp)
        selection.add_dirichlet_noise(root_node, config)

    logger.debug(
        f"[MCTS] Starting MCTS loop for {config.num_simulations} simulations (Batch Size: {MCTS_BATCH_SIZE}). Root state step: {root_node.state.current_step}"
    )
    sim_count = 0
    while sim_count < config.num_simulations:
        leaves_to_evaluate: list[Node] = []
        paths_to_backprop: list[tuple[Node, float]] = []
        start_time_batch_select = time.monotonic()
        num_collected_for_batch = 0
        while (
            num_collected_for_batch < MCTS_BATCH_SIZE
            and sim_count < config.num_simulations
        ):
            current_sim_idx = sim_count
            sim_count += 1
            leaf_node: Node | None = None
            try:
                logger.debug(
                    f"  [MCTS Select] Starting Sim {current_sim_idx + 1} Selection..."
                )
                leaf_node, selection_depth = selection.traverse_to_leaf(
                    root_node, config
                )

                if leaf_node.state.is_over():
                    outcome = leaf_node.state.get_outcome()
                    logger.debug(
                        f"  [MCTS Select] Sim {current_sim_idx + 1}: Selected TERMINAL leaf at depth {selection_depth}. Outcome: {outcome:.3f}. Adding to backprop queue."
                    )
                    paths_to_backprop.append((leaf_node, outcome))
                elif not leaf_node.is_expanded:
                    logger.debug(
                        f"  [MCTS Select] Sim {current_sim_idx + 1}: Selected leaf for EVALUATION at depth {selection_depth}. Node: {leaf_node}."
                    )
                    leaves_to_evaluate.append(leaf_node)
                    num_collected_for_batch += 1
                else:
                    logger.debug(
                        f"  [MCTS Select] Sim {current_sim_idx + 1}: Traversal stopped at EXPANDED node (likely max depth {selection_depth}). Value: {leaf_node.value_estimate:.3f}. Adding to backprop queue."
                    )
                    paths_to_backprop.append((leaf_node, leaf_node.value_estimate))

            except Exception as e:
                sim_error_count += 1
                logger.error(
                    f"[MCTS Select] Error during MCTS selection phase (Sim {current_sim_idx + 1}): {e}",
                    exc_info=True,
                )

        selection_duration = time.monotonic() - start_time_batch_select
        logger.debug(
            f"[MCTS Select] Batch selection phase finished. Collected {len(leaves_to_evaluate)} leaves for NN eval. Duration: {selection_duration:.4f}s"
        )

        evaluation_start_time = time.monotonic()
        if leaves_to_evaluate:
            logger.debug(
                f"  [MCTS Eval] Evaluating batch of {len(leaves_to_evaluate)} leaves..."
            )
            try:
                leaf_states = [node.state for node in leaves_to_evaluate]
                batch_results = network_evaluator.evaluate_batch(leaf_states)

                if batch_results is None or len(batch_results) != len(
                    leaves_to_evaluate
                ):
                    raise MCTSExecutionError(
                        f"Network evaluation returned invalid results. Expected {len(leaves_to_evaluate)}, got {len(batch_results) if batch_results else 'None'}"
                    )

                for i, node in enumerate(leaves_to_evaluate):
                    action_policy, value = batch_results[i]
                    if not isinstance(action_policy, dict) or not all(
                        isinstance(k, int)
                        and isinstance(v, float)
                        and np.isfinite(v)
                        and v >= 0
                        for k, v in action_policy.items()
                    ):
                        logger.error(
                            f"  [MCTS Eval] Invalid policy format received post-evaluation for leaf {i}. Policy: {action_policy}"
                        )
                        value = 0.0
                    if not isinstance(value, float) or not np.isfinite(value):
                        logger.warning(
                            f"  [MCTS Eval] Invalid value received post-evaluation for leaf {i}: {value}. Using 0."
                        )
                        value = 0.0

                    if not node.is_expanded and not node.state.is_over():
                        expansion.expand_node_with_policy(node, action_policy)
                        logger.debug(
                            f"  [MCTS Eval/Expand] Expanded evaluated leaf node {i}: {node}"
                        )
                    else:
                        logger.debug(
                            f"  [MCTS Eval/Expand] Evaluated leaf node {i} was already expanded or terminal. Skipping expansion."
                        )

                    paths_to_backprop.append((node, value))

            except Exception as e:
                eval_error_count += len(leaves_to_evaluate)
                logger.error(
                    f"[MCTS Eval] Error during MCTS batch evaluation/expansion: {e}",
                    exc_info=True,
                )

                logger.warning(
                    f"[MCTS Eval] Skipping backpropagation for {len(leaves_to_evaluate)} leaves due to evaluation error."
                )

        evaluation_duration = time.monotonic() - evaluation_start_time
        logger.debug(
            f"[MCTS Eval] Evaluation/Expansion phase finished. Duration: {evaluation_duration:.4f}s"
        )

        backprop_start_time = time.monotonic()
        logger.debug(
            f"  [MCTS Backprop] Backpropagating {len(paths_to_backprop)} paths..."
        )
        for i, (leaf_node, value_to_prop) in enumerate(paths_to_backprop):
            try:
                depth_bp = backpropagation.backpropagate_value(leaf_node, value_to_prop)
                max_depth_overall = max(max_depth_overall, depth_bp)
                sim_success_count += 1
                logger.debug(
                    f"    [Backprop] Path {i}: Value={value_to_prop:.4f}, Depth={depth_bp}, Node={leaf_node}"
                )
            except Exception as bp_err:
                logger.error(
                    f"    [Backprop] Error backpropagating path {i} (Value={value_to_prop:.4f}, Node={leaf_node}): {bp_err}",
                    exc_info=True,
                )
                sim_error_count += 1

        backprop_duration = time.monotonic() - backprop_start_time
        logger.debug(
            f"[MCTS Backprop] Backpropagation phase finished. Duration: {backprop_duration:.4f}s"
        )

    final_log_level = logging.INFO
    total_sims_attempted = sim_count
    logger.log(
        final_log_level,
        f"[MCTS] MCTS loop finished. Target Sims: {config.num_simulations}. Attempted: {total_sims_attempted}. "
        f"Successful Backprops: {sim_success_count}. Selection/BP Errors: {sim_error_count}. Eval Errors: {eval_error_count}. "
        f"Root visits: {root_node.visit_count}. Max depth reached: {max_depth_overall}",
    )
    if root_node.children:
        child_visits_log = {a: c.visit_count for a, c in root_node.children.items()}
        logger.info(f"[MCTS] Root children visit counts: {child_visits_log}")
    elif not root_node.state.is_over():
        logger.warning("[MCTS] MCTS finished but root node still has no children.")

    if sim_error_count + eval_error_count > config.num_simulations * 0.5:
        raise MCTSExecutionError(
            f"MCTS failed: Too many errors ({sim_error_count} selection/BP, {eval_error_count} eval) during {total_sims_attempted} simulations."
        )

    return max_depth_overall
