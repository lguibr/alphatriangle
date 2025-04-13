# File: src/mcts/core/README.md
# MCTS Core Submodule (`src.mcts.core`)

## Purpose and Architecture

This submodule defines the fundamental building blocks and interfaces for the Monte Carlo Tree Search implementation.

-   **`Node`:** The `Node` class is the cornerstone, representing a single state within the search tree. It stores the associated `GameState`, parent/child relationships, the action that led to it, and crucial MCTS statistics (visit count, total action value, prior probability). It provides properties like `value_estimate` (Q-value) and `is_expanded`.
-   **`search`:** The `search.py` module contains the high-level `run_mcts_simulations` function. This function orchestrates the core MCTS loop for a given root node: repeatedly selecting leaves, batch-evaluating them using the network, expanding them, and backpropagating the results, using helper functions from the `src.mcts.strategy` submodule.
-   **`config`:** The `config.py` module defines the `MCTSConfig` class, encapsulating all hyperparameters specific to the MCTS algorithm (e.g., simulation count, PUCT constant, temperature, Dirichlet noise parameters). This centralizes MCTS tuning parameters.
-   **`types`:** The `types.py` module defines essential type hints and protocols for the MCTS module. Most importantly, it defines the `ActionPolicyValueEvaluator` protocol, which specifies the `evaluate` and `evaluate_batch` methods that any neural network interface must implement to be usable by the MCTS expansion phase. It also defines `ActionPolicyMapping`.

## Exposed Interfaces

-   **Classes:**
    -   `Node`: Represents a node in the search tree.
    -   `MCTSConfig`: Holds MCTS hyperparameters.
-   **Functions:**
    -   `run_mcts_simulations(root_node: Node, config: MCTSConfig, network_evaluator: ActionPolicyValueEvaluator)`: Orchestrates the MCTS process using batched evaluation.
-   **Protocols/Types:**
    -   `ActionPolicyValueEvaluator`: Defines the interface for the NN evaluator.
    -   `ActionPolicyMapping`: Type alias for the policy dictionary (mapping action index to probability).

## Dependencies

-   **`src.environment`**:
    -   `GameState`: Used within `Node` to represent the state. Methods like `is_over`, `get_outcome`, `valid_actions`, `copy`, `step` are used during the MCTS process (selection, expansion).
-   **`src.mcts.strategy`**:
    -   `selection`, `expansion`, `backpropagation`: The `run_mcts_simulations` function delegates the core algorithm phases to functions within this submodule.
-   **`src.utils.types`**:
    -   `ActionType`, `PolicyValueOutput`: Used in type hints and protocols.
-   **Standard Libraries:** `typing`, `math`, `logging`.

---

**Note:** Please keep this README updated when modifying the `Node` structure, the `MCTSConfig` parameters, the `run_mcts_simulations` logic, or the `ActionPolicyValueEvaluator` interface definition. Accurate documentation is crucial for maintainability.