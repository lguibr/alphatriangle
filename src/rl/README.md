# File: src/rl/README.md
# Reinforcement Learning Module (`src.rl`)

## Purpose and Architecture

This module contains core components related to the reinforcement learning algorithm itself, specifically the `Trainer` for network updates, the `ExperienceBuffer` for storing data, and the `SelfPlayWorker` actor for generating data. **The overall orchestration of the training process has been moved to the `src.training` module.**

-   **Core Components (`src.rl.core`):**
    -   `Trainer`: Responsible for performing the neural network update steps. It takes batches of experience from the buffer, calculates losses (applying importance sampling weights if using PER), updates the network weights, and calculates TD errors for PER priority updates.
    -   `ExperienceBuffer`: A replay buffer storing `Experience` tuples (`(StateType, policy_target, value_target)`). Supports both uniform sampling and Prioritized Experience Replay (PER).
-   **Self-Play Components (`src.rl.self_play`):**
    -   `worker`: Defines the `SelfPlayWorker` Ray actor. Each actor runs game episodes independently using MCTS and its local copy of the neural network. It collects experiences and returns results via a `SelfPlayResult` object.
-   **Types (`src.rl.types`):**
    -   Defines Pydantic models like `SelfPlayResult` for structured data transfer between Ray actors and the training loop.

## Exposed Interfaces

-   **Core:**
    -   `Trainer`:
        -   `__init__(nn_interface: NeuralNetwork, train_config: TrainConfig, env_config: EnvConfig)`
        -   `train_step(per_sample: PERBatchSample) -> Optional[Tuple[Dict[str, float], np.ndarray]]`: Takes PER sample, returns loss info and TD errors.
        -   `load_optimizer_state(state_dict: dict)`
        -   `get_current_lr() -> float`
    -   `ExperienceBuffer`:
        -   `__init__(config: TrainConfig)`
        -   `add(experience: Experience)`
        -   `add_batch(experiences: List[Experience])`
        -   `sample(batch_size: int, current_train_step: Optional[int] = None) -> Optional[PERBatchSample]`: Samples batch, requires step for PER beta.
        -   `update_priorities(tree_indices: np.ndarray, td_errors: np.ndarray)`: Updates priorities for PER.
        -   `is_ready() -> bool`
        -   `__len__() -> int`
-   **Self-Play:**
    -   `SelfPlayWorker`: Ray actor class.
        -   `run_episode() -> SelfPlayResult`
        -   `set_weights(weights: Dict)`
-   **Types:**
    -   `SelfPlayResult`: Pydantic model for self-play results.

## Dependencies

-   **`src.config`**: `TrainConfig`, `EnvConfig`, `ModelConfig`, `MCTSConfig`.
-   **`src.nn`**: `NeuralNetwork`.
-   **`src.features`**: `extract_state_features`.
-   **`src.mcts`**: Core MCTS components.
-   **`src.environment`**: `GameState`.
-   **`src.stats`**: `StatsCollectorActor` (used indirectly via `src.training`).
-   **`src.utils`**: Types and helpers.
-   **`src.structs`**: Implicitly used via `GameState`.
-   **`torch`**: Used by `Trainer` and `NeuralNetwork`.
-   **`ray`**: Used by `SelfPlayWorker`.
-   **Standard Libraries:** `typing`, `logging`, `collections.deque`, `numpy`.

---

**Note:** Please keep this README updated when changing the responsibilities of the Trainer, Buffer, or SelfPlayWorker.