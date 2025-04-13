# File: src/mcts/README.md
# Monte Carlo Tree Search Module (`src.mcts`)

## Purpose and Architecture

This module implements the Monte Carlo Tree Search algorithm, a core component of the AlphaZero-style reinforcement learning agent. MCTS is used during self-play to explore the game tree and determine the next best move and generate training targets for the neural network.

-   **Core Components (`src.mcts.core`):**
    -   `Node`: Represents a state in the search tree, storing visit counts, value estimates, prior probabilities, and child nodes. Holds a `GameState` object.
    -   `search`: Contains the main `run_mcts_simulations` function orchestrating the selection, expansion, and backpropagation phases. **This version uses batched neural network evaluation (`evaluate_batch`) for potentially improved performance.** It collects multiple leaf nodes before calling the network.
    -   `config`: Defines the `MCTSConfig` class holding hyperparameters like the number of simulations, PUCT coefficient, temperature settings, and Dirichlet noise parameters.
    -   `types`: Defines necessary type hints and protocols, notably `ActionPolicyValueEvaluator` which specifies the interface required for the neural network evaluator used by MCTS.
-   **Strategy Components (`src.mcts.strategy`):**
    -   `selection`: Implements the tree traversal logic (PUCT calculation, Dirichlet noise addition, leaf selection).
    -   `expansion`: Handles expanding leaf nodes **using pre-computed policy priors** obtained from batched network evaluation.
    -   `backpropagation`: Implements the process of updating node statistics back up the tree after a simulation.
    -   `policy`: Provides functions to select the final action based on visit counts (`select_action_based_on_visits`) and to generate the policy target vector for training (`get_policy_target`).

## Exposed Interfaces

-   **Core:**
    -   `Node`: The tree node class.
    -   `MCTSConfig`: Configuration class (defined in `src.mcts.core.config`).
    -   `run_mcts_simulations(root_node: Node, config: MCTSConfig, network_evaluator: ActionPolicyValueEvaluator)`: The main function to run MCTS (uses batched evaluation).
    -   `ActionPolicyValueEvaluator`: Protocol defining the NN evaluation interface.
    -   `ActionPolicyMapping`: Type alias for the policy dictionary.
-   **Strategy:**
    -   `select_action_based_on_visits(root_node: Node, temperature: float) -> Optional[ActionType]`: Selects the final move.
    -   `get_policy_target(root_node: Node, temperature: float = 1.0) -> ActionPolicyMapping`: Generates the training policy target.

## Dependencies

-   **`src.environment`**:
    -   `GameState`: Represents the state within each `Node`. MCTS interacts heavily with `GameState` methods like `copy()`, `step()`, `is_over()`, `get_outcome()`, `valid_actions()`.
    -   `EnvConfig`: Accessed via `GameState`.
-   **`src.nn`**:
    -   `NeuralNetwork`: An instance conforming to the `ActionPolicyValueEvaluator` protocol is required by `run_mcts_simulations` and `expansion` to evaluate states (specifically `evaluate_batch`).
-   **`src.utils.types`**:
    -   `ActionType`, `PolicyValueOutput`: Used for actions and NN return types.
-   **`numpy`**:
    -   Used for Dirichlet noise generation and potentially in policy calculations.
-   **Standard Libraries:** `typing`, `math`, `logging`, `numpy`, `time`.

---

**Note:** Please keep this README updated when changing the MCTS algorithm phases (selection, expansion, backpropagation), the node structure, configuration options, or the interaction with the environment or neural network, especially regarding the batched evaluation. Accurate documentation is crucial for maintainability.