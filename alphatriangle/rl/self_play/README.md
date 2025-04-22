
# RL Self-Play Submodule (`alphatriangle.rl.self_play`)

## Purpose and Architecture

This submodule focuses specifically on generating game episodes through self-play, driven by the current neural network and MCTS. It is designed to run in parallel using Ray actors managed by the [`alphatriangle.training.worker_manager`](../../training/worker_manager.py).

-   **[`worker.py`](worker.py):** Defines the `SelfPlayWorker` class, decorated with `@ray.remote`.
    -   Each `SelfPlayWorker` actor runs independently, typically on a separate CPU core.
    -   It initializes its own `GameState` environment and `NeuralNetwork` instance (usually on the CPU).
    -   It receives configuration objects (`EnvConfig`, `MCTSConfig`, `ModelConfig`, `TrainConfig`) during initialization.
    -   It has a `set_weights` method allowing the `TrainingLoop` to periodically update its local neural network with the latest trained weights from the central model. It also has `set_current_trainer_step` to store the global step associated with the current weights, called by the `WorkerManager`.
    -   Its main method, `run_episode`, simulates a complete game episode:
        -   Includes detailed logging for debugging.
        -   Uses its local `NeuralNetwork` evaluator and `MCTSConfig` to run MCTS using `trimcts.run_mcts`.
        -   Selects actions based on MCTS results ([`mcts_helpers.select_action_from_visits`](mcts_helpers.py)).
        -   Generates policy targets ([`mcts_helpers.get_policy_target_from_visits`](mcts_helpers.py)).
        -   Stores `(StateType, policy_target, n_step_return)` tuples (using extracted features and calculated n-step returns).
        -   Steps its local game environment (`GameState.step`).
        -   Returns the collected `Experience` list, final score, episode length, and MCTS statistics via a `SelfPlayResult` object.
        -   **Asynchronously sends raw metric events (`RawMetricEvent`) for step rewards, MCTS simulations, current score, and episode completion (including score, length, simulations) to the `StatsCollectorActor`, tagged with the `current_trainer_step` (global step of its network weights).**
-   **[`mcts_helpers.py`](mcts_helpers.py):** Contains helper functions for processing MCTS visit counts into policy targets and selecting actions based on temperature. Includes `PolicyGenerationError` for specific failures.

## Exposed Interfaces

-   **Classes:**
    -   `SelfPlayWorker`: Ray actor class.
        -   `__init__(...)`
        -   `run_episode() -> SelfPlayResult`: Runs one episode and returns results.
        -   `set_weights(weights: Dict)`: Updates the actor's local network weights.
        -   `set_current_trainer_step(global_step: int)`: Updates the stored trainer step.
-   **Types:**
    -   `SelfPlayResult`: Pydantic model defined in [`alphatriangle.rl.types`](../types.py).
-   **Functions (from `mcts_helpers.py`):**
    -   `select_action_from_visits(...) -> ActionType`
    -   `get_policy_target_from_visits(...) -> PolicyTargetMapping`
    -   `PolicyGenerationError` (Exception)

## Dependencies

-   **[`alphatriangle.config`](../../config/README.md)**:
    -   `EnvConfig`, `AlphaTriangleMCTSConfig`, `ModelConfig`, `TrainConfig`.
-   **`trianglengin`**:
    -   `GameState`, `EnvConfig`.
-   **`trimcts`**:
    -   `run_mcts`, `SearchConfiguration`.
-   **[`alphatriangle.nn`](../../nn/README.md)**:
    -   `NeuralNetwork`: Instantiated locally within the actor.
-   **[`alphatriangle.features`](../../features/README.md)**:
    -   `extract_state_features`: Used to generate `StateType` for experiences.
-   **[`alphatriangle.utils`](../../utils/README.md)**:
    -   `types`: `Experience`, `ActionType`, `PolicyTargetMapping`, `StateType`.
    -   `helpers`: `get_device`, `set_random_seeds`.
-   **[`alphatriangle.rl.types`](../types.py)**:
    -   `SelfPlayResult`: Return type.
-   **[`alphatriangle.stats`](../../stats/README.md)**:
    -   `StatsCollectorActor`, **`RawMetricEvent`**: Handle passed for logging.
-   **`numpy`**:
    -   Used by MCTS strategies and feature extraction.
-   **`ray`**:
    -   The `@ray.remote` decorator makes this a Ray actor.
-   **`torch`**:
    -   Used by the local `NeuralNetwork`.
-   **Standard Libraries:** `typing`, `logging`, `random`, `time`, `collections.deque`, `cProfile`.

---

**Note:** Please keep this README updated when changing the self-play episode generation logic, the data collected, the interaction with MCTS/environment, or the asynchronous logging behavior, especially regarding the sending of `RawMetricEvent` objects. Accurate documentation is crucial for maintainability.
