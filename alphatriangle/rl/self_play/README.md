# RL Self-Play Submodule (`alphatriangle.rl.self_play`)

## Purpose and Architecture

This submodule focuses specifically on generating game episodes through self-play, driven by the current neural network and MCTS. It is designed to run in parallel using Ray actors managed by the `TrainingOrchestrator`.

-   **`worker.py`:** Defines the `SelfPlayWorker` class, decorated with `@ray.remote`.
    -   Each `SelfPlayWorker` actor runs independently, typically on a separate CPU core.
    -   It initializes its own `GameState` environment and `NeuralNetwork` instance (usually on the CPU).
    -   It receives configuration objects (`EnvConfig`, `MCTSConfig`, `ModelConfig`, `TrainConfig`) during initialization.
    -   It has a `set_weights` method allowing the `TrainingOrchestrator` to periodically update its local neural network with the latest trained weights from the central model.
    -   Its main method, `run_episode`, simulates a complete game episode:
        -   Uses its local `NeuralNetwork` evaluator and `MCTSConfig` to run MCTS (`alphatriangle.mcts.run_mcts_simulations`), **reusing the search tree between moves**.
        -   Selects actions based on MCTS results (`alphatriangle.mcts.strategy.policy.select_action_based_on_visits`).
        -   Generates policy targets (`alphatriangle.mcts.strategy.policy.get_policy_target`).
        -   Stores `(StateType, policy_target, placeholder_value)` tuples (using extracted features).
        -   Steps its local game environment (`GameState.step`).
        -   Backfills the value target after the episode ends.
        -   Returns the collected `Experience` list, final score, episode length, and **a copy of the final `GameState`** to the orchestrator via a `SelfPlayResult` object.
    -   **Removed:** It no longer pushes intermediate state to a central actor for visualization. The orchestrator uses the final state returned in `SelfPlayResult`.

## Exposed Interfaces

-   **Classes:**
    -   `SelfPlayWorker`: Ray actor class.
        -   `__init__(...)`
        -   `run_episode() -> SelfPlayResult`: Runs one episode and returns results.
        -   `set_weights(weights: Dict)`: Updates the actor's local network weights.
-   **Types:**
    -   `SelfPlayResult`: Pydantic model defined in `alphatriangle.rl.types`.

## Dependencies

-   **`alphatriangle.config`**:
    -   `EnvConfig`, `MCTSConfig`, `ModelConfig`, `TrainConfig`.
-   **`alphatriangle.nn`**:
    -   `NeuralNetwork`: Instantiated locally within the actor.
-   **`alphatriangle.mcts`**:
    -   Core MCTS functions and types. **MCTS uses batched evaluation.**
-   **`alphatriangle.environment`**:
    -   `GameState`, `EnvConfig`: Used to instantiate and step through the game simulation locally.
-   **`alphatriangle.features`**:
    -   `extract_state_features`: Used to generate `StateType` for experiences.
-   **`alphatriangle.utils`**:
    -   `types`: `Experience`, `ActionType`, `PolicyTargetMapping`, `StateType`.
    -   `helpers`: `get_device`, `set_random_seeds`.
-   **`alphatriangle.rl.types`**:
    -   `SelfPlayResult`: Return type.
-   **`numpy`**:
    -   Used by MCTS strategies.
-   **`ray`**:
    -   The `@ray.remote` decorator makes this a Ray actor.
-   **`torch`**:
    -   Used by the local `NeuralNetwork`.
-   **Standard Libraries:** `typing`, `logging`, `random`, `time`.

---

**Note:** Please keep this README updated when changing the self-play episode generation logic within the actor, the data collected (`Experience`), or the interaction with MCTS or the environment, especially regarding tree reuse. Accurate documentation is crucial for maintainability.