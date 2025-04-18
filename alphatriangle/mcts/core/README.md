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