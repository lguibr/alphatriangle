File: README.md
# File: alphatriangle/mcts/README.md
# Monte Carlo Tree Search Module (`alphatriangle.mcts`)

## Purpose and Architecture

This module implements the Monte Carlo Tree Search algorithm, a core component of the AlphaZero-style reinforcement learning agent. MCTS is used during self-play to explore the game tree and determine the next best move and generate training targets for the neural network.

-   **Core Components ([`core/README.md`](core/README.md)):**
    -   `Node`: Represents a state in the search tree, storing visit counts, value estimates, prior probabilities, and child nodes. Holds a `GameState` object.
    -   `search`: Contains the main `run_mcts_simulations` function orchestrating the selection, expansion, and backpropagation phases. **This version uses batched neural network evaluation (`evaluate_batch`) for potentially improved performance.** It collects multiple leaf nodes before calling the network.
    -   `config`: Defines the `MCTSConfig` class holding hyperparameters like the number of simulations, PUCT coefficient, temperature settings, and Dirichlet noise parameters.
    -   `types`: Defines necessary type hints and protocols, notably `ActionPolicyValueEvaluator` which specifies the interface required for the neural network evaluator used by MCTS.
-   **Strategy Components ([`strategy/README.md`](strategy/README.md)):**
    -   `selection`: Implements the tree traversal logic (PUCT calculation, Dirichlet noise addition, leaf selection).
    -   `expansion`: Handles expanding leaf nodes **using pre-computed policy priors** obtained from batched network evaluation.
    -   `backpropagation`: Implements the process of updating node statistics back up the tree after a simulation.
    -   `policy`: Provides functions to select the final action based on visit counts (`select_action_based_on_visits`) and to generate the policy target vector for training (`get_policy_target`).

## Exposed Interfaces

-   **Core:**
    -   `Node`: The tree node class.
    -   `MCTSConfig`: Configuration class (defined in [`alphatriangle.config`](../config/README.md)).
    -   `run_mcts_simulations(root_node: Node, config: MCTSConfig, network_evaluator: ActionPolicyValueEvaluator)`: The main function to run MCTS (uses batched evaluation).
    -   `ActionPolicyValueEvaluator`: Protocol defining the NN evaluation interface.
    -   `ActionPolicyMapping`: Type alias for the policy dictionary.
    -   `MCTSExecutionError`: Custom exception for MCTS failures.
-   **Strategy:**
    -   `select_action_based_on_visits(root_node: Node, temperature: float) -> ActionType`: Selects the final move.
    -   `get_policy_target(root_node: Node, temperature: float = 1.0) -> ActionPolicyMapping`: Generates the training policy target.

## Dependencies

-   **[`alphatriangle.environment`](../environment/README.md)**:
    -   `GameState`: Represents the state within each `Node`. MCTS interacts heavily with `GameState` methods like `copy()`, `step()`, `is_over()`, `get_outcome()`, `valid_actions()`.
    -   `EnvConfig`: Accessed via `GameState`.
-   **[`alphatriangle.nn`](../nn/README.md)**:
    -   `NeuralNetwork`: An instance conforming to the `ActionPolicyValueEvaluator` protocol is required by `run_mcts_simulations` and `expansion` to evaluate states (specifically `evaluate_batch`).
-   **[`alphatriangle.config`](../config/README.md)**:
    -   `MCTSConfig`: Provides hyperparameters.
-   **[`alphatriangle.utils`](../utils/README.md)**:
    -   `ActionType`, `PolicyValueOutput`: Used for actions and NN return types.
-   **`numpy`**:
    -   Used for Dirichlet noise generation and potentially in policy calculations.
-   **Standard Libraries:** `typing`, `math`, `logging`, `numpy`, `time`, `concurrent.futures`.

---

**Note:** Please keep this README updated when changing the MCTS algorithm phases (selection, expansion, backpropagation), the node structure, configuration options, or the interaction with the environment or neural network, especially regarding the batched evaluation. Accurate documentation is crucial for maintainability.

File: __init__.py
"""
Monte Carlo Tree Search (MCTS) module.
Provides the core algorithm and components for game tree search.
"""

from alphatriangle.config import MCTSConfig

from .core.node import Node
from .core.search import (
    MCTSExecutionError,
    run_mcts_simulations,
)
from .core.types import ActionPolicyMapping, ActionPolicyValueEvaluator
from .strategy.policy import get_policy_target, select_action_based_on_visits

__all__ = [
    # Core
    "Node",
    "run_mcts_simulations",
    "MCTSConfig",
    "ActionPolicyValueEvaluator",
    "ActionPolicyMapping",
    "MCTSExecutionError",
    # Strategy
    "select_action_based_on_visits",
    "get_policy_target",
]


File: core\node.py
# File: alphatriangle/mcts/core/node.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

# Import GameState from trianglengin
from trianglengin.core.environment import GameState

# Keep ActionType from alphatriangle utils for now
from alphatriangle.utils.types import ActionType

logger = logging.getLogger(__name__)


class Node:
    """Represents a node in the Monte Carlo Search Tree."""

    def __init__(
        self,
        state: GameState,  # Use trianglengin.GameState
        parent: Node | None = None,
        action_taken: ActionType | None = None,
        prior_probability: float = 0.0,
    ):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.children: dict[ActionType, Node] = {}
        self.visit_count: int = 0
        self.total_action_value: float = 0.0
        self.prior_probability: float = prior_probability

    @property
    def is_expanded(self) -> bool:
        """Checks if the node has been expanded (i.e., children generated)."""
        return bool(self.children)

    @property
    def is_leaf(self) -> bool:
        """Checks if the node is a leaf (not expanded)."""
        return not self.is_expanded

    @property
    def value_estimate(self) -> float:
        """
        Calculates the Q-value (average action value) estimate for this node's state.
        This is the average value observed from simulations starting from this state.
        Refactored for clarity and safety.
        """
        if self.visit_count == 0:
            return 0.0

        visits = max(1, self.visit_count)
        q_value = self.total_action_value / visits

        return q_value

    def __repr__(self) -> str:
        return (
            f"Node(StateStep={self.state.current_step}, "
            f"FromAction={self.action_taken}, Visits={self.visit_count}, "
            f"Value={self.value_estimate:.3f}, Prior={self.prior_probability:.4f}, "
            f"Children={len(self.children)})"
        )


File: core\README.md
# File: alphatriangle/mcts/core/README.md
# MCTS Core Submodule (`alphatriangle.mcts.core`)

## Purpose and Architecture

This submodule defines the fundamental building blocks and interfaces for the Monte Carlo Tree Search implementation.

-   **[`Node`](node.py):** The `Node` class is the cornerstone, representing a single state within the search tree. It stores the associated `GameState`, parent/child relationships, the action that led to it, and crucial MCTS statistics (visit count, total action value, prior probability). It provides properties like `value_estimate` (Q-value) and `is_expanded`.
-   **[`search`](search.py):** The `search.py` module contains the high-level `run_mcts_simulations` function. This function orchestrates the core MCTS loop for a given root node: repeatedly selecting leaves, batch-evaluating them using the network, expanding them, and backpropagating the results, using helper functions from the [`alphatriangle.mcts.strategy`](../strategy/README.md) submodule. **It uses a `ThreadPoolExecutor` for parallel traversals and batches network evaluations.**
-   **[`types`](types.py):** The `types.py` module defines essential type hints and protocols for the MCTS module. Most importantly, it defines the `ActionPolicyValueEvaluator` protocol, which specifies the `evaluate` and `evaluate_batch` methods that any neural network interface must implement to be usable by the MCTS expansion phase. It also defines `ActionPolicyMapping`.

## Exposed Interfaces

-   **Classes:**
    -   `Node`: Represents a node in the search tree.
    -   `MCTSExecutionError`: Custom exception for MCTS failures.
-   **Functions:**
    -   `run_mcts_simulations(root_node: Node, config: MCTSConfig, network_evaluator: ActionPolicyValueEvaluator)`: Orchestrates the MCTS process using batched evaluation and parallel traversals.
-   **Protocols/Types:**
    -   `ActionPolicyValueEvaluator`: Defines the interface for the NN evaluator.
    -   `ActionPolicyMapping`: Type alias for the policy dictionary (mapping action index to probability).

## Dependencies

-   **[`alphatriangle.environment`](../../environment/README.md)**:
    -   `GameState`: Used within `Node` to represent the state. Methods like `is_over`, `get_outcome`, `valid_actions`, `copy`, `step` are used during the MCTS process (selection, expansion).
-   **[`alphatriangle.mcts.strategy`](../strategy/README.md)**:
    -   `selection`, `expansion`, `backpropagation`: The `run_mcts_simulations` function delegates the core algorithm phases to functions within this submodule.
-   **[`alphatriangle.config`](../../config/README.md)**:
    -   `MCTSConfig`: Provides hyperparameters.
-   **[`alphatriangle.utils`](../../utils/README.md)**:
    -   `ActionType`, `PolicyValueOutput`: Used in type hints and protocols.
-   **Standard Libraries:** `typing`, `math`, `logging`, `concurrent.futures`, `time`.
-   **`numpy`**: Used for validation checks.

---

**Note:** Please keep this README updated when modifying the `Node` structure, the `run_mcts_simulations` logic (especially parallelism and batching), or the `ActionPolicyValueEvaluator` interface definition. Accurate documentation is crucial for maintainability.

File: core\search.py
# File: alphatriangle/mcts/core/search.py
import concurrent.futures
import logging
import time

import numpy as np

# Import GameState from trianglengin
from trianglengin.core.environment import GameState

# Keep alphatriangle imports
from ...config import MCTSConfig
from ..strategy import backpropagation, expansion, selection
from .node import Node
from .types import ActionPolicyValueEvaluator

logger = logging.getLogger(__name__)

MCTS_PARALLEL_TRAVERSALS = 16


class MCTSExecutionError(Exception):
    """Custom exception for errors during MCTS execution."""

    pass


def _run_single_traversal(root_node: Node, config: MCTSConfig) -> tuple[Node, int]:
    """Helper function to run a single MCTS traversal (selection phase)."""
    try:
        leaf_node, selection_depth = selection.traverse_to_leaf(root_node, config)
        return leaf_node, selection_depth
    except Exception as e:
        logger.error(
            f"[MCTS Traversal Task] Error during traversal: {e}", exc_info=True
        )
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
    # Use is_over() method from trianglengin.GameState
    if root_node.state.is_over():
        logger.warning("[MCTS] MCTS started on a terminal state. No simulations run.")
        return 0

    max_depth_overall = 0
    sim_success_count = 0
    sim_error_count = 0
    eval_error_count = 0
    total_sims_run = 0

    if not root_node.is_expanded:
        logger.debug("[MCTS] Root node not expanded, performing initial evaluation...")
        try:
            # Pass trianglengin.GameState to evaluator
            action_policy, root_value = network_evaluator.evaluate(root_node.state)
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
            # Use is_over() method from trianglengin.GameState
            if root_node.is_expanded or root_node.state.is_over():
                depth_bp = backpropagation.backpropagate_value(root_node, root_value)
                max_depth_overall = max(max_depth_overall, depth_bp)
                selection.add_dirichlet_noise(root_node, config)
            else:
                logger.warning("[MCTS] Initial root expansion failed.")
        except Exception as e:
            logger.error(
                f"[MCTS] Initial root evaluation/expansion failed: {e}", exc_info=True
            )
            raise MCTSExecutionError(
                f"Initial root evaluation/expansion failed: {e}"
            ) from e
    elif root_node.visit_count == 0:
        selection.add_dirichlet_noise(root_node, config)

    logger.info(
        f"[MCTS] Starting MCTS loop for {config.num_simulations} simulations "
        f"(Parallel Traversals: {MCTS_PARALLEL_TRAVERSALS}). Root state step: {root_node.state.current_step}"
    )

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

            futures_to_leaf: dict[concurrent.futures.Future, int] = {}
            for i in range(num_to_launch):
                future = executor.submit(_run_single_traversal, root_node, config)
                futures_to_leaf[future] = processed_simulations + i

            leaves_to_evaluate: list[Node] = []
            paths_to_backprop: list[tuple[Node, float]] = []
            traversal_results: list[tuple[Node | None, int, Exception | None]] = []

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

            for leaf_node_optional, selection_depth, error in traversal_results:
                if error or leaf_node_optional is None:
                    continue
                valid_leaf_node: Node = leaf_node_optional
                max_depth_overall = max(max_depth_overall, selection_depth)

                # Use is_over() and get_outcome() from trianglengin.GameState
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
                else:
                    logger.debug(
                        f"    [Process] Sim result: EXPANDED leaf (likely max depth). Value: {valid_leaf_node.value_estimate:.3f}. Adding to backprop."
                    )
                    paths_to_backprop.append(
                        (valid_leaf_node, valid_leaf_node.value_estimate)
                    )

            evaluation_start_time = time.monotonic()
            if leaves_to_evaluate:
                logger.debug(
                    f"  [MCTS Eval] Evaluating batch of {len(leaves_to_evaluate)} leaves..."
                )
                try:
                    # Pass list of trianglengin.GameState to evaluator
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
                        if (
                            not isinstance(action_policy, dict)
                            or not isinstance(value, float)
                            or not np.isfinite(value)
                        ):
                            logger.error(
                                f"    [MCTS Eval] Invalid policy/value received for leaf {i}. Policy: {type(action_policy)}, Value: {value}. Using 0 value."
                            )
                            value = 0.0
                            action_policy = {}

                        # Use is_over() from trianglengin.GameState
                        if not node.is_expanded and not node.state.is_over():
                            expansion.expand_node_with_policy(node, action_policy)
                            logger.debug(
                                f"    [MCTS Eval/Expand] Expanded evaluated leaf node {i}: {node}"
                            )
                        paths_to_backprop.append((node, value))

                except Exception as e:
                    eval_error_count += len(leaves_to_evaluate)
                    logger.error(
                        f"  [MCTS Eval] Error during batch evaluation/expansion: {e}",
                        exc_info=True,
                    )

            evaluation_duration = time.monotonic() - evaluation_start_time
            if leaves_to_evaluate:
                logger.debug(
                    f"  [MCTS Eval] Evaluation/Expansion phase finished. Duration: {evaluation_duration:.4f}s"
                )

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
                    sim_error_count += 1

            backprop_duration = time.monotonic() - backprop_start_time
            logger.debug(
                f"  [MCTS Backprop] Backpropagation phase finished. Duration: {backprop_duration:.4f}s"
            )

            processed_simulations += num_to_launch
            pending_simulations -= num_to_launch
            total_sims_run = processed_simulations

            logger.debug(
                f"[MCTS Batch] Finished batch. Processed: {processed_simulations}/{config.num_simulations}. Pending: {pending_simulations}"
            )

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
    # Use is_over() from trianglengin.GameState
    elif not root_node.state.is_over():
        logger.warning("[MCTS] MCTS finished but root node still has no children.")

    total_errors = sim_error_count + eval_error_count
    if total_errors > config.num_simulations * 0.5:  # Increased threshold
        raise MCTSExecutionError(
            f"MCTS failed: High error rate ({total_errors} errors in {total_sims_run} simulations)."
        )
    elif total_errors > 0:
        logger.warning(f"[MCTS] Completed with {total_errors} errors.")

    return max_depth_overall


File: core\types.py
# File: alphatriangle/mcts/core/types.py
from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol

# Import GameState from trianglengin
from trianglengin.core.environment import GameState

# Keep alphatriangle utils types for now
from ...utils.types import ActionType, PolicyValueOutput

ActionPolicyMapping = Mapping[ActionType, float]


class ActionPolicyValueEvaluator(Protocol):
    """Defines the interface for evaluating a game state using a neural network."""

    def evaluate(
        self, state: GameState
    ) -> PolicyValueOutput:  # Uses trianglengin.GameState
        """
        Evaluates a single game state using the neural network.

        Args:
            state: The GameState object to evaluate.

        Returns:
            A tuple containing:
                - ActionPolicyMapping: A mapping from valid action indices
                    to their prior probabilities (output by the policy head).
                - float: The estimated value of the state (output by the value head).
        """
        ...

    def evaluate_batch(
        self, states: list[GameState]
    ) -> list[PolicyValueOutput]:  # Uses trianglengin.GameState
        """
        Evaluates a batch of game states using the neural network.
        (Optional but recommended for performance if MCTS supports batch evaluation).

        Args:
            states: A list of GameState objects to evaluate.

        Returns:
            A list of tuples, where each tuple corresponds to an input state and contains:
                - ActionPolicyMapping: Action probabilities for that state.
                - float: The estimated value of that state.
        """
        ...


File: core\__init__.py


File: strategy\backpropagation.py
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.node import Node

logger = logging.getLogger(__name__)


def backpropagate_value(leaf_node: "Node", value: float) -> int:
    """
    Propagates the simulation value back up the tree from the leaf node.
    Returns the depth of the backpropagation path (number of nodes updated).
    """
    current_node: Node | None = leaf_node
    path_str = []
    depth = 0
    logger.debug(
        f"[Backprop] Starting backprop from leaf (Action={leaf_node.action_taken}, StateStep={leaf_node.state.current_step}) with value={value:.4f}"
    )

    while current_node is not None:
        q_before = current_node.value_estimate
        total_val_before = current_node.total_action_value
        visits_before = current_node.visit_count

        current_node.visit_count += 1
        current_node.total_action_value += value

        q_after = current_node.value_estimate
        total_val_after = current_node.total_action_value
        visits_after = current_node.visit_count

        action_str = (
            f"Act={current_node.action_taken}"
            if current_node.action_taken is not None
            else "Root"
        )
        path_str.append(f"N({action_str},V={visits_after},Q={q_after:.3f})")

        logger.debug(
            f"  [Backprop] Depth {depth}: Node({action_str}), "
            f"Visits: {visits_before} -> {visits_after}, "
            f"AddedVal={value:.4f}, "
            f"TotalVal: {total_val_before:.3f} -> {total_val_after:.3f}, "
            f"Q: {q_before:.3f} -> {q_after:.3f}"
        )

        current_node = current_node.parent
        depth += 1

    logger.debug(f"[Backprop] Finished. Path: {' <- '.join(reversed(path_str))}")
    return depth


File: strategy\expansion.py
# File: alphatriangle/mcts/strategy/expansion.py
import logging
from typing import TYPE_CHECKING

# Import GameState from trianglengin
from trianglengin.core.environment import GameState

# Keep alphatriangle utils types for now
from ...utils.types import ActionType
from ..core.node import Node
from ..core.types import ActionPolicyMapping

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


File: strategy\policy.py
# File: alphatriangle/mcts/strategy/policy.py
import logging
import random

import numpy as np

# Import EnvConfig from trianglengin
from trianglengin.config import EnvConfig

# Keep alphatriangle imports
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
            selected_action = rng.choice(actions, p=probabilities)
            logger.debug(
                f"[PolicySelect] Sampled action (temp={temperature:.2f}): {selected_action}"
            )
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
    # Access EnvConfig from the node's state
    env_config: EnvConfig = root_node.state.env_config
    action_dim = int(env_config.ACTION_DIM)  # type: ignore[call-overload]
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


File: strategy\README.md
# File: alphatriangle/mcts/strategy/README.md
# MCTS Strategy Submodule (`alphatriangle.mcts.strategy`)

## Purpose and Architecture

This submodule implements the specific algorithms and heuristics used within the different phases of the Monte Carlo Tree Search, as orchestrated by [`alphatriangle.mcts.core.search.run_mcts_simulations`](../core/search.py).

-   **[`selection`](selection.py):** Contains the logic for traversing the tree from the root to a leaf node.
    -   `calculate_puct_score`: Implements the PUCT (Polynomial Upper Confidence Trees) formula, balancing exploitation (node value) and exploration (prior probability and visit counts).
    -   `add_dirichlet_noise`: Adds noise to the root node's prior probabilities to encourage exploration early in the search, as done in AlphaZero.
    -   `select_child_node`: Chooses the child with the highest PUCT score.
    -   `traverse_to_leaf`: Repeatedly applies `select_child_node` to navigate down the tree.
-   **[`expansion`](expansion.py):** Handles the expansion of a selected leaf node.
    -   `expand_node_with_policy`: Takes a node and a *pre-computed* policy dictionary (obtained from batched network evaluation) and creates child `Node` objects for all valid actions, initializing them with the corresponding prior probabilities.
-   **[`backpropagation`](backpropagation.py):** Implements the update step after a simulation.
    -   `backpropagate_value`: Traverses from the expanded leaf node back up to the root, incrementing the `visit_count` and adding the simulation's resulting `value` to the `total_action_value` of each node along the path.
-   **[`policy`](policy.py):** Provides functions related to action selection and policy target generation after MCTS has run.
    -   `select_action_based_on_visits`: Selects the final action to be played in the game based on the visit counts of the root's children, using a temperature parameter to control exploration vs. exploitation.
    -   `get_policy_target`: Generates the policy target vector (a probability distribution over actions) based on the visit counts, which is used as a training target for the neural network's policy head.

## Exposed Interfaces

-   **Selection:**
    -   `traverse_to_leaf(root_node: Node, config: MCTSConfig) -> Tuple[Node, int]`: Returns leaf node and depth.
    -   `add_dirichlet_noise(node: Node, config: MCTSConfig)`
    -   `select_child_node(node: Node, config: MCTSConfig) -> Node` (Primarily internal use)
    -   `calculate_puct_score(...) -> Tuple[float, float, float]` (Primarily internal use)
    -   `SelectionError`: Custom exception.
-   **Expansion:**
    -   `expand_node_with_policy(node: Node, action_policy: ActionPolicyMapping)`
-   **Backpropagation:**
    -   `backpropagate_value(leaf_node: Node, value: float) -> int`: Returns depth.
-   **Policy:**
    -   `select_action_based_on_visits(root_node: Node, temperature: float) -> ActionType`: Selects the final move.
    -   `get_policy_target(root_node: Node, temperature: float = 1.0) -> ActionPolicyMapping`: Generates the training policy target.
    -   `PolicyGenerationError`: Custom exception.

## Dependencies

-   **[`alphatriangle.mcts.core`](../core/README.md)**:
    -   `Node`: The primary data structure operated upon.
    -   `ActionPolicyMapping`: Used in `expansion` and `policy`.
-   **[`alphatriangle.config`](../../config/README.md)**:
    -   `MCTSConfig`: Provides hyperparameters (PUCT coeff, noise params, etc.).
-   **[`alphatriangle.environment`](../../environment/README.md)**:
    -   `GameState`: Accessed via `Node.state` for methods like `is_over`, `get_outcome`, `valid_actions`, `step`.
-   **[`alphatriangle.utils`](../../utils/README.md)**:
    -   `ActionType`: Used for representing actions.
-   **`numpy`**:
    -   Used for Dirichlet noise and policy/selection calculations.
-   **Standard Libraries:** `typing`, `math`, `logging`, `numpy`, `random`.

---

**Note:** Please keep this README updated when modifying the algorithms within selection, expansion, backpropagation, or policy generation, or changing how they interact with the `Node` structure or `MCTSConfig`. Accurate documentation is crucial for maintainability.

File: strategy\selection.py
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


File: strategy\__init__.py


