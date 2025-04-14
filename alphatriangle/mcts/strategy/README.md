# File: src/mcts/strategy/README.md
# MCTS Strategy Submodule (`src.mcts.strategy`)

## Purpose and Architecture

This submodule implements the specific algorithms and heuristics used within the different phases of the Monte Carlo Tree Search, as orchestrated by `src.mcts.core.search.run_mcts_simulations`.

-   **`selection`:** Contains the logic for traversing the tree from the root to a leaf node.
    -   `calculate_puct_score`: Implements the PUCT (Polynomial Upper Confidence Trees) formula, balancing exploitation (node value) and exploration (prior probability and visit counts).
    -   `add_dirichlet_noise`: Adds noise to the root node's prior probabilities to encourage exploration early in the search, as done in AlphaZero.
    -   `select_child_node`: Chooses the child with the highest PUCT score.
    -   `traverse_to_leaf`: Repeatedly applies `select_child_node` to navigate down the tree.
-   **`expansion`:** Handles the expansion of a selected leaf node.
    -   `expand_node_with_policy`: Takes a node and a *pre-computed* policy dictionary (obtained from batched network evaluation) and creates child `Node` objects for all valid actions, initializing them with the corresponding prior probabilities.
-   **`backpropagation`:** Implements the update step after a simulation.
    -   `backpropagate_value`: Traverses from the expanded leaf node back up to the root, incrementing the `visit_count` and adding the simulation's resulting `value` to the `total_action_value` of each node along the path.
-   **`policy`:** Provides functions related to action selection and policy target generation after MCTS has run.
    -   `select_action_based_on_visits`: Selects the final action to be played in the game based on the visit counts of the root's children, using a temperature parameter to control exploration vs. exploitation.
    -   `get_policy_target`: Generates the policy target vector (a probability distribution over actions) based on the visit counts, which is used as a training target for the neural network's policy head.

## Exposed Interfaces

-   **Selection:**
    -   `traverse_to_leaf(root_node: Node, config: MCTSConfig) -> Node`
    -   `add_dirichlet_noise(node: Node, config: MCTSConfig)`
    -   `select_child_node(node: Node, config: MCTSConfig) -> Node` (Primarily internal use)
    -   `calculate_puct_score(...) -> float` (Primarily internal use)
-   **Expansion:**
    -   `expand_node_with_policy(node: Node, action_policy: ActionPolicyMapping)`
-   **Backpropagation:**
    -   `backpropagate_value(leaf_node: Node, value: float)`
-   **Policy:**
    -   `select_action_based_on_visits(root_node: Node, temperature: float) -> Optional[ActionType]`
    -   `get_policy_target(root_node: Node, temperature: float = 1.0) -> ActionPolicyMapping`

## Dependencies

-   **`src.mcts.core`**:
    -   `Node`: The primary data structure operated upon.
    -   `MCTSConfig`: Provides hyperparameters (PUCT coeff, noise params, etc.).
    -   `ActionPolicyMapping`: Used in `expansion` and `policy`.
-   **`src.environment`**:
    -   `GameState`: Accessed via `Node.state` for methods like `is_over`, `get_outcome`, `valid_actions`, `step`.
-   **`src.utils.types`**:
    -   `ActionType`: Used for representing actions.
-   **`numpy`**:
    -   Used for Dirichlet noise and potentially in policy/selection calculations.
-   **Standard Libraries:** `typing`, `math`, `logging`, `numpy`.

---

**Note:** Please keep this README updated when modifying the algorithms within selection, expansion, backpropagation, or policy generation, or changing how they interact with the `Node` structure or `MCTSConfig`. Accurate documentation is crucial for maintainability.