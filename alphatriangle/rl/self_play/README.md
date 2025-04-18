# File: alphatriangle/rl/self_play/README.md
# RL Self-Play Submodule (`alphatriangle.rl.self_play`)

## Purpose and Architecture

This submodule focuses specifically on generating game episodes through self-play, driven by the current neural network and MCTS. It is designed to run in parallel using Ray actors managed by the [`alphatriangle.training.worker_manager`](../../training/worker_manager.py).

-   **[`worker.py`](worker.py):** Defines the `SelfPlayWorker` class, decorated with `@ray.remote`.
    -   Each `SelfPlayWorker` actor runs independently, typically on a separate CPU core.
    -   It initializes its own `GameState` environment and `NeuralNetwork` instance (usually on the CPU).
    -   It receives configuration objects (`EnvConfig`, `MCTSConfig`, `ModelConfig`, `TrainConfig`) during initialization.
    -   It has a `set_weights` method allowing the `TrainingLoop` to periodically update its local neural network with the latest trained weights from the central model. **It also has `set_current_trainer_step` to store the global step associated with the current weights, called by the `WorkerManager`.**
    -   Its main method, `run_episode`, simulates a complete game episode:
        -   Uses its local `NeuralNetwork` evaluator and `MCTSConfig` to run MCTS ([`alphatriangle.mcts.run_mcts_simulations`](../../mcts/core/search.py)), **reusing the search tree between moves**.
        -   Selects actions based on MCTS results ([`alphatriangle.mcts.strategy.policy.select_action_based_on_visits`](../../mcts/strategy/policy.py)).
        -   Generates policy targets ([`alphatriangle.mcts.strategy.policy.get_policy_target`](../../mcts/strategy/policy.py)).
        -   Stores `(StateType, policy_target, n_step_return)` tuples (using extracted features and calculated n-step returns).
        -   Steps its local game environment (`GameState.step`).
        -   Returns the collected `Experience` list, final score, episode length, and MCTS statistics via a `SelfPlayResult` object.
        -   **Asynchronously logs per-step statistics (score, reward, MCTS visits/depth) to the `StatsCollectorActor`, providing a `StepInfo` dictionary containing the `game_step_index` and the `current_trainer_step` (global step of its current network weights).**
        -   **Asynchronously reports its current `GameState` to the `StatsCollectorActor` for visualization.**

## Exposed Interfaces

-   **Classes:**
    -   `SelfPlayWorker`: Ray actor class.
        -   `__init__(...)`
        -   `run_episode() -> SelfPlayResult`: Runs one episode and returns results.
        -   `set_weights(weights: Dict)`: Updates the actor's local network weights.
        -   `set_current_trainer_step(global_step: int)`: Updates the stored trainer step.
-   **Types:**
    -   `SelfPlayResult`: Pydantic model defined in [`alphatriangle.rl.types`](../types.py).

## Dependencies

-   **[`alphatriangle.config`](../../config/README.md)**:
    -   `EnvConfig`, `MCTSConfig`, `ModelConfig`, `TrainConfig`.
-   **[`alphatriangle.nn`](../../nn/README.md)**:
    -   `NeuralNetwork`: Instantiated locally within the actor.
-   **[`alphatriangle.mcts`](../../mcts/README.md)**:
    -   Core MCTS functions and types. **MCTS uses batched evaluation.**
-   **[`alphatriangle.environment`](../../environment/README.md)**:
    -   `GameState`, `EnvConfig`: Used to instantiate and step through the game simulation locally.
-   **[`alphatriangle.features`](../../features/README.md)**:
    -   `extract_state_features`: Used to generate `StateType` for experiences.
-   **[`alphatriangle.utils`](../../utils/README.md)**:
    -   `types`: `Experience`, `ActionType`, `PolicyTargetMapping`, `StateType`, `StepInfo`.
    -   `helpers`: `get_device`, `set_random_seeds`.
-   **[`alphatriangle.rl.types`](../types.py)**:
    -   `SelfPlayResult`: Return type.
-   **[`alphatriangle.stats`](../../stats/README.md)**:
    -   `StatsCollectorActor`: Handle passed for logging.
-   **`numpy`**:
    -   Used by MCTS strategies.
-   **`ray`**:
    -   The `@ray.remote` decorator makes this a Ray actor.
-   **`torch`**:
    -   Used by the local `NeuralNetwork`.
-   **Standard Libraries:** `typing`, `logging`, `random`, `time`, `collections.deque`.

---

**Note:** Please keep this README updated when changing the self-play episode generation logic, the data collected, the interaction with MCTS/environment, or the asynchronous logging behavior, especially regarding the inclusion of `current_trainer_step` in `StepInfo`. Accurate documentation is crucial for maintainability.