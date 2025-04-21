
# Neural Network Module (`alphatriangle.nn`)

## Purpose and Architecture

This module defines and manages the neural network used by the AlphaTriangle agent. It follows the AlphaZero paradigm, featuring a shared body and separate heads for policy and value prediction.

-   **Model Definition ([`model.py`](model.py)):**
    -   The `AlphaTriangleNet` class (inheriting from `torch.nn.Module`) defines the network architecture.
    -   It includes convolutional layers for processing the grid state, potentially residual blocks.
    -   **Optionally**, it can include a **Transformer Encoder block** after the CNN/ResNet body to apply self-attention over the spatial features before combining them with other input features. This is controlled by `ModelConfig.USE_TRANSFORMER`.
    -   The output from the CNN/Transformer body is combined with other extracted features (e.g., shape info) and passed through shared fully connected layers.
    -   It splits into two heads:
        -   **Policy Head:** Outputs logits representing the probability distribution over all possible actions.
        -   **Value Head:** Outputs logits representing a **distribution** over possible state values (C51 Distributional RL).
    -   The architecture is configurable via [`ModelConfig`](../config/model_config.py). **It uses `trianglengin.EnvConfig` for environment dimensions.**
-   **Network Interface ([`network.py`](network.py)):**
    -   The `NeuralNetwork` class acts as a wrapper around the `AlphaTriangleNet` PyTorch model.
    -   It provides a clean interface for the rest of the system (MCTS, Trainer) to interact with the network, abstracting away PyTorch specifics.
    -   It **internally uses [`alphatriangle.features.extract_state_features`](../features/extractor.py)** to convert input `GameState` objects into tensors before feeding them to the underlying `AlphaTriangleNet` model.
    -   It handles the **distributional value head**, calculating the expected value from the predicted distribution for use by MCTS.
    -   It **optionally compiles** the underlying model using `torch.compile()` based on `TrainConfig.COMPILE_MODEL` for potential performance improvements.
    -   **Crucially, it conforms to the `trimcts.AlphaZeroNetworkInterface` protocol**, providing `evaluate_state` and `evaluate_batch` methods with the expected signatures (returning concrete `dict` for policy).
    -   Key methods:
        -   `evaluate_state(state: GameState) -> tuple[dict[int, float], float]`: Takes a `GameState`, extracts features, performs a forward pass, and returns the policy probabilities (as a dictionary) and the **expected scalar value estimate**.
        -   `evaluate_batch(states: list[GameState]) -> list[tuple[dict[int, float], float]]`: Extracts features from a batch of `GameState` objects and performs batched evaluation for efficiency.
        -   `get_weights()`: Returns the model's state dictionary (on CPU).
        -   `set_weights(weights: Dict)`: Loads weights into the model (handles device placement).
    -   It handles device placement (`torch.device`).

## Exposed Interfaces

-   **Classes:**
    -   `AlphaTriangleNet(model_config: ModelConfig, env_config: EnvConfig)`: The PyTorch `nn.Module` defining the architecture.
    -   `NeuralNetwork(model_config: ModelConfig, env_config: EnvConfig, train_config: TrainConfig, device: torch.device)`: The wrapper class providing the primary interface.
        -   `evaluate_state(state: GameState) -> tuple[dict[int, float], float]`
        -   `evaluate_batch(states: list[GameState]) -> list[tuple[dict[int, float], float]]`
        -   `get_weights() -> Dict[str, torch.Tensor]`
        -   `set_weights(weights: Dict[str, torch.Tensor])`
        -   `model`: Public attribute to access the underlying `AlphaTriangleNet` instance.
        -   `device`: Public attribute indicating the `torch.device`.
        -   `model_config`: Public attribute.
        -   `num_atoms`, `v_min`, `v_max`, `delta_z`, `support`: Attributes related to the distributional value head.

## Dependencies

-   **[`alphatriangle.config`](../config/README.md)**:
    -   `ModelConfig`: Defines the network architecture parameters.
    -   `TrainConfig`: Used by `NeuralNetwork` init (e.g., for `COMPILE_MODEL`).
-   **`trianglengin`**:
    -   `GameState`: Input type for `evaluate_state` and `evaluate_batch`.
    -   `EnvConfig`: Provides environment dimensions (grid size, action space size) needed by the model.
-   **[`alphatriangle.features`](../features/README.md)**:
    -   `extract_state_features`: Used internally by `NeuralNetwork` to process `GameState` inputs.
-   **[`alphatriangle.utils`](../utils/README.md)**:
    -   `ActionType`, `PolicyValueOutput`, `StateType`: Used in method signatures and return types.
-   **`torch`**:
    -   The core deep learning framework (`torch`, `torch.nn`, `torch.nn.functional`).
-   **`numpy`**:
    -   Used for converting state components to tensors.
-   **Standard Libraries:** `typing`, `os`, `logging`, `math`, `sys`.

---

**Note:** Please keep this README updated when changing the neural network architecture (`AlphaTriangleNet`), the `NeuralNetwork` interface methods, or its interaction with configuration or other modules (especially `alphatriangle.features` and `trianglengin`). Accurate documentation is crucial for maintainability.
