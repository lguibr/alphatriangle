# Neural Network Module (`src.nn`)

## Purpose and Architecture

This module defines and manages the neural network used by the AlphaTriangle agent. It follows the AlphaZero paradigm, featuring a shared body and separate heads for policy and value prediction.

-   **Model Definition (`model.py`):**
    -   The `AlphaTriangleNet` class (inheriting from `torch.nn.Module`) defines the network architecture.
    -   It includes convolutional layers for processing the grid state, potentially residual blocks.
    -   **Optionally**, it can include a **Transformer Encoder block** after the CNN/ResNet body to apply self-attention over the spatial features before combining them with other input features. This is controlled by `ModelConfig.USE_TRANSFORMER`.
    -   The output from the CNN/Transformer body is combined with other extracted features (e.g., shape info) and passed through shared fully connected layers.
    -   It splits into two heads:
        -   **Policy Head:** Outputs logits representing the probability distribution over all possible actions.
        -   **Value Head:** Outputs a single scalar value estimating the expected outcome from the current state.
    -   The architecture is configurable via `ModelConfig`.
-   **Network Interface (`network.py`):**
    -   The `NeuralNetwork` class acts as a wrapper around the `AlphaTriangleNet` PyTorch model.
    -   It provides a clean interface for the rest of the system (MCTS, Trainer) to interact with the network, abstracting away PyTorch specifics.
    -   It **internally uses `src.features.extract_state_features`** to convert input `GameState` objects into tensors before feeding them to the underlying `AlphaTriangleNet` model.
    -   Key methods:
        -   `evaluate(state: GameState)`: Takes a `GameState`, extracts features, performs a forward pass, and returns the policy probabilities (as a dictionary) and the scalar value estimate. Conforms to the `ActionPolicyValueEvaluator` protocol required by MCTS.
        -   `evaluate_batch(states: List[GameState])`: Extracts features from a batch of `GameState` objects and performs batched evaluation for efficiency.
        -   `get_weights()`: Returns the model's state dictionary (on CPU).
        -   `set_weights(weights: Dict)`: Loads weights into the model (handles device placement).
    -   It handles device placement (`torch.device`).

## Exposed Interfaces

-   **Classes:**
    -   `AlphaTriangleNet(model_config: ModelConfig, env_config: EnvConfig)`: The PyTorch `nn.Module` defining the architecture.
    -   `NeuralNetwork(model_config: ModelConfig, env_config: EnvConfig, train_config: TrainConfig, device: torch.device)`: The wrapper class providing the primary interface.
        -   `evaluate(state: GameState) -> PolicyValueOutput`
        -   `evaluate_batch(states: List[GameState]) -> List[PolicyValueOutput]`
        -   `get_weights() -> Dict[str, torch.Tensor]`
        -   `set_weights(weights: Dict[str, torch.Tensor])`
        -   `model`: Public attribute to access the underlying `AlphaTriangleNet` instance.
        -   `device`: Public attribute indicating the `torch.device`.
        -   `model_config`: Public attribute.

## Dependencies

-   **`src.config`**:
    -   `ModelConfig`: Defines the network architecture parameters (including expected feature dimensions and Transformer options).
    -   `EnvConfig`: Provides environment dimensions (grid size, action space size) needed by the model.
    -   `TrainConfig`: Used by `NeuralNetwork` init.
-   **`src.environment`**:
    -   `GameState`: Input type for `evaluate` and `evaluate_batch`.
-   **`src.features`**:
    -   `extract_state_features`: Used internally by `NeuralNetwork` to process `GameState` inputs.
-   **`src.utils.types`**:
    -   `ActionType`, `PolicyValueOutput`, `StateType`: Used in method signatures and return types.
-   **`torch`**:
    -   The core deep learning framework (`torch`, `torch.nn`, `torch.nn.functional`).
-   **`numpy`**:
    -   Used for converting state components to tensors.
-   **Standard Libraries:** `typing`, `os`, `logging`, `math`.

---

**Note:** Please keep this README updated when changing the neural network architecture (`AlphaTriangleNet`, including Transformer usage), the `NeuralNetwork` interface methods, or its interaction with configuration or other modules (especially `src.features`). Accurate documentation is crucial for maintainability.