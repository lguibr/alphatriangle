

*   **Change:** Mention the newly logged metrics.

```markdown
# Reinforcement Learning Module (`alphatriangle.rl`)

## Purpose and Architecture

This module contains core components related to the reinforcement learning algorithm itself, specifically the `Trainer` for network updates, the `ExperienceBuffer` for storing data, and the `SelfPlayWorker` actor for generating data using the `trimcts` library. **The overall orchestration of the training process resides in the [`alphatriangle.training`](../training/README.md) module.**

-   **Core Components ([`core/README.md`](core/README.md)):**
    -   `Trainer`: Responsible for performing the neural network update steps. It takes batches of experience from the buffer, calculates losses (policy cross-entropy, **distributional value cross-entropy**, optional entropy bonus), applies importance sampling weights if using PER, updates the network weights, and calculates TD errors for PER priority updates. **Uses `trianglengin.EnvConfig`.**
    -   `ExperienceBuffer`: A replay buffer storing `Experience` tuples (`(StateType, policy_target, n_step_return)`). The `StateType` contains processed numerical features (`grid`, `other_features`) **and the geometry of available shapes (`available_shapes_geometry`)**. Supports both uniform sampling and Prioritized Experience Replay (PER).
-   **Self-Play Components ([`self_play/README.md`](self_play/README.md)):**
    -   `worker`: Defines the `SelfPlayWorker` Ray actor. Each actor runs game episodes independently using `trimcts.run_mcts` and its local copy of the neural network (`NeuralNetwork` which conforms to `trimcts.AlphaZeroNetworkInterface`). It collects experiences (including calculated n-step returns) and returns results via a `SelfPlayResult` object. It also logs stats (including step rewards, episode scores, and lengths) and game state asynchronously to the `StatsCollectorActor`. **Uses `trianglengin.GameState` and `trianglengin.EnvConfig`.**
-   **Types ([`types.py`](types.py)):**
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
        -   `set_current_trainer_step(global_step: int)`
-   **Types:**
    -   `SelfPlayResult`: Pydantic model for self-play results.

## Dependencies

-   **[`alphatriangle.config`](../config/README.md)**: `TrainConfig`, `ModelConfig`, `AlphaTriangleMCTSConfig`.
-   **`trianglengin`**: `GameState`, `EnvConfig`.
-   **`trimcts`**: `run_mcts`, `SearchConfiguration`, `AlphaZeroNetworkInterface`.
-   **[`alphatriangle.nn`](../nn/README.md)**: `NeuralNetwork`.
-   **[`alphatriangle.features`](../features/README.md)**: `extract_state_features`.
-   **[`alphatriangle.stats`](../stats/README.md)**: `StatsCollectorActor` (used indirectly via `alphatriangle.training`).
-   **[`alphatriangle.utils`](../utils/README.md)**: Types (`Experience`, `StateType`, `PERBatchSample`, `StepInfo`) and helpers (`SumTree`).
-   **`torch`**: Used by `Trainer` and `NeuralNetwork`.
-   **`ray`**: Used by `SelfPlayWorker`.
-   **Standard Libraries:** `typing`, `logging`, `collections.deque`, `numpy`, `random`, `time`.

---

**Note:** Please keep this README updated when changing the responsibilities of the Trainer, Buffer, or SelfPlayWorker.
