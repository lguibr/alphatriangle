# File: alphatriangle/mcts/core/search.py
import concurrent.futures
import logging
import time

import numpy as np

from ...config import MCTSConfig
from ..strategy import backpropagation, expansion, selection
from .node import Node
from .types import ActionPolicyValueEvaluator

logger = logging.getLogger(__name__)

# --- CHANGED: Default batch size, can be adjusted ---
MCTS_PARALLEL_TRAVERSALS = 16  # Number of traversals to run in parallel
# --- END CHANGED ---


class MCTSExecutionError(Exception):
    """Custom exception for errors during MCTS execution."""

    pass


def _run_single_traversal(root_node: Node, config: MCTSConfig) -> tuple[Node, int]:
    """Helper function to run a single MCTS traversal (selection phase)."""
    # This function is designed to be thread-safe as selection reads node stats
    # but doesn't modify them until backpropagation.
    try:
        leaf_node, selection_depth = selection.traverse_to_leaf(root_node, config)
        return leaf_node, selection_depth
    except Exception as e:
        # Log error within the thread/task for better context
        logger.error(
            f"[MCTS Traversal Task] Error during traversal: {e}", exc_info=True
        )
        # Re-raise or return an indicator? Re-raising is cleaner for future handling.
        raise MCTSExecutionError(f"Traversal failed: {e}") from e


def run_mcts_simulations(
    root_node: Node,
    config: MCTSConfig,
    network_evaluator: ActionPolicyValueEvaluator,
) -> int:
    """
    Runs the specified number of MCTS simulations from the root node.
    Uses a ThreadPoolExecutor to run selection traversals in parallel.
    Neural network evaluations are batched. Backpropagation is sequential.

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
    total_sims_run = 0

    # --- Initial Root Expansion (if needed) ---
    if not root_node.is_expanded:
        logger.debug("[MCTS] Root node not expanded, performing initial evaluation...")
        try:
            action_policy, root_value = network_evaluator.evaluate(root_node.state)
            # Basic validation (can be enhanced)
            if not isinstance(action_policy, dict) or not isinstance(root_value, float):
                raise MCTSExecutionError("Initial evaluation returned invalid type.")
            if not np.all(np.isfinite(list(action_policy.values()))):
                raise MCTSExecutionError(
                    "Initial evaluation returned non-finite policy."
                )
            if not np.isfinite(root_value):
                raise MCTSExecutionError(
                    "Initial evaluation returned non-finite value."
                )

            expansion.expand_node_with_policy(root_node, action_policy)
            if root_node.is_expanded or root_node.state.is_over():
                depth_bp = backpropagation.backpropagate_value(root_node, root_value)
                max_depth_overall = max(max_depth_overall, depth_bp)
                selection.add_dirichlet_noise(
                    root_node, config
                )  # Apply noise after first expansion/backprop
            else:
                logger.warning("[MCTS] Initial root expansion failed.")
        except Exception as e:
            logger.error(
                f"[MCTS] Initial root evaluation/expansion failed: {e}", exc_info=True
            )
            raise MCTSExecutionError(
                f"Initial root evaluation/expansion failed: {e}"
            ) from e
    elif root_node.visit_count == 0:  # Apply noise if root is expanded but unvisited
        selection.add_dirichlet_noise(root_node, config)
    # --- End Initial Root Expansion ---

    logger.info(
        f"[MCTS] Starting MCTS loop for {config.num_simulations} simulations "
        f"(Parallel Traversals: {MCTS_PARALLEL_TRAVERSALS}). Root state step: {root_node.state.current_step}"
    )

    # Use ThreadPoolExecutor for parallel traversals
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MCTS_PARALLEL_TRAVERSALS
    ) as executor:
        pending_simulations = config.num_simulations
        processed_simulations = 0

        while pending_simulations > 0:
            num_to_launch = min(pending_simulations, MCTS_PARALLEL_TRAVERSALS)
            logger.debug(
                f"[MCTS Batch] Launching {num_to_launch} parallel traversals..."
            )

            # --- Submit Traversal Tasks ---
            futures_to_leaf: dict[concurrent.futures.Future, int] = {}
            for i in range(num_to_launch):
                future = executor.submit(_run_single_traversal, root_node, config)
                futures_to_leaf[future] = processed_simulations + i  # Store sim index

            leaves_to_evaluate: list[Node] = []
            paths_to_backprop: list[tuple[Node, float]] = []
            traversal_results: list[tuple[Node | None, int, Exception | None]] = []

            # --- Collect Traversal Results ---
            for future in concurrent.futures.as_completed(futures_to_leaf):
                sim_idx = futures_to_leaf[future]
                try:
                    leaf_node, selection_depth = future.result()
                    traversal_results.append((leaf_node, selection_depth, None))
                    logger.debug(
                        f"  [MCTS Traversal] Sim {sim_idx + 1} completed. Depth: {selection_depth}, Leaf: {leaf_node}"
                    )
                except Exception as exc:
                    sim_error_count += 1
                    traversal_results.append((None, 0, exc))
                    logger.error(f"  [MCTS Traversal] Sim {sim_idx + 1} failed: {exc}")

            # --- Process Traversal Results ---
            for leaf_node_optional, selection_depth, error in traversal_results:
                # --- CHANGED: Explicit check and assignment ---
                if error or leaf_node_optional is None:
                    continue  # Skip failed traversals

                # Now we know leaf_node_optional is not None, assign to typed variable
                valid_leaf_node: Node = leaf_node_optional
                # --- END CHANGED ---

                max_depth_overall = max(max_depth_overall, selection_depth)

                # --- Use valid_leaf_node ---
                if valid_leaf_node.state.is_over():
                    outcome = valid_leaf_node.state.get_outcome()
                    logger.debug(
                        f"    [Process] Sim result: TERMINAL leaf. Outcome: {outcome:.3f}. Adding to backprop."
                    )
                    paths_to_backprop.append((valid_leaf_node, outcome))
                elif not valid_leaf_node.is_expanded:
                    logger.debug(
                        "    [Process] Sim result: Leaf needs EVALUATION. Adding to batch."
                    )
                    leaves_to_evaluate.append(valid_leaf_node)
                else:  # Hit max depth or encountered selection error resulting in expanded node
                    logger.debug(
                        f"    [Process] Sim result: EXPANDED leaf (likely max depth). Value: {valid_leaf_node.value_estimate:.3f}. Adding to backprop."
                    )
                    paths_to_backprop.append(
                        (valid_leaf_node, valid_leaf_node.value_estimate)
                    )
                # --- END Use valid_leaf_node ---

            # --- Batch Evaluate Leaves ---
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
                            "Network evaluation returned invalid results."
                        )

                    for i, node in enumerate(leaves_to_evaluate):
                        action_policy, value = batch_results[i]
                        # Basic validation
                        if (
                            not isinstance(action_policy, dict)
                            or not isinstance(value, float)
                            or not np.isfinite(value)
                        ):
                            logger.error(
                                f"    [MCTS Eval] Invalid policy/value received for leaf {i}. Policy: {type(action_policy)}, Value: {value}. Using 0 value."
                            )
                            value = 0.0  # Use neutral value on error
                            action_policy = {}  # Use empty policy on error

                        if not node.is_expanded and not node.state.is_over():
                            expansion.expand_node_with_policy(node, action_policy)
                            logger.debug(
                                f"    [MCTS Eval/Expand] Expanded evaluated leaf node {i}: {node}"
                            )

                        paths_to_backprop.append(
                            (node, value)
                        )  # Add evaluated node for backprop

                except Exception as e:
                    eval_error_count += len(leaves_to_evaluate)
                    logger.error(
                        f"  [MCTS Eval] Error during batch evaluation/expansion: {e}",
                        exc_info=True,
                    )
                    # Skip backprop for these leaves if eval failed

            evaluation_duration = time.monotonic() - evaluation_start_time
            if leaves_to_evaluate:
                logger.debug(
                    f"  [MCTS Eval] Evaluation/Expansion phase finished. Duration: {evaluation_duration:.4f}s"
                )

            # --- Sequential Backpropagation ---
            backprop_start_time = time.monotonic()
            logger.debug(
                f"  [MCTS Backprop] Backpropagating {len(paths_to_backprop)} paths..."
            )
            for i, (leaf_node_bp, value_to_prop) in enumerate(paths_to_backprop):
                try:
                    depth_bp = backpropagation.backpropagate_value(
                        leaf_node_bp, value_to_prop
                    )
                    max_depth_overall = max(max_depth_overall, depth_bp)
                    sim_success_count += 1
                    logger.debug(
                        f"    [Backprop] Path {i}: Value={value_to_prop:.4f}, Depth={depth_bp}, Node={leaf_node_bp}"
                    )
                except Exception as bp_err:
                    logger.error(
                        f"    [Backprop] Error backpropagating path {i} (Value={value_to_prop:.4f}, Node={leaf_node_bp}): {bp_err}",
                        exc_info=True,
                    )
                    sim_error_count += 1  # Count backprop errors separately

            backprop_duration = time.monotonic() - backprop_start_time
            logger.debug(
                f"  [MCTS Backprop] Backpropagation phase finished. Duration: {backprop_duration:.4f}s"
            )

            # --- Update Loop Control ---
            processed_simulations += num_to_launch
            pending_simulations -= num_to_launch
            total_sims_run = (
                processed_simulations  # Track total sims attempted in this run
            )

            logger.debug(
                f"[MCTS Batch] Finished batch. Processed: {processed_simulations}/{config.num_simulations}. Pending: {pending_simulations}"
            )

    # --- Final Logging ---
    final_log_level = logging.INFO
    logger.log(
        final_log_level,
        f"[MCTS] MCTS loop finished. Target Sims: {config.num_simulations}. Attempted: {total_sims_run}. "
        f"Successful Backprops: {sim_success_count}. Traversal Errors: {sim_error_count}. Eval Errors: {eval_error_count}. "
        f"Root visits: {root_node.visit_count}. Max depth reached: {max_depth_overall}",
    )
    if root_node.children:
        child_visits_log = {a: c.visit_count for a, c in root_node.children.items()}
        logger.info(f"[MCTS] Root children visit counts: {child_visits_log}")
    elif not root_node.state.is_over():
        logger.warning("[MCTS] MCTS finished but root node still has no children.")

    # --- Error Check ---
    total_errors = sim_error_count + eval_error_count
    if total_errors > config.num_simulations * 0.01:  # Example threshold: 50% errors
        raise MCTSExecutionError(
            f"MCTS failed: High error rate ({total_errors} errors in {total_sims_run} simulations)."
        )
    elif total_errors > 0:
        logger.warning(f"[MCTS] Completed with {total_errors} errors.")

    return max_depth_overall
