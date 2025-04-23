# RL Core Submodule (alphatriangle.rl.core)

## Purpose and Architecture

This submodule contains core classes directly involved in the reinforcement learning update process and data storage. **The orchestration logic resides in the [alphatriangle.training](../../training/README.md) module, and persistence/statistics are handled by the trieye library.**

-   **[Trainer](trainer.py):** This class encapsulates the logic for updating the neural network's weights.
    -   It holds the main NeuralNetwork interface, optimizer, and scheduler.
    -   Its train_step method takes a batch of experiences (potentially with PER indices and weights), performs forward/backward passes, calculates losses (policy cross-entropy, **distributional value cross-entropy**, optional entropy bonus), applies importance sampling weights if using PER, updates weights, and returns **raw loss components** and calculated TD errors for PER priority updates. **Logging of these losses is handled externally by the TrainingLoop sending events to the TrieyeActor.** Uses trianglengin.EnvConfig.
-   **[ExperienceBuffer](buffer.py):** This class implements a replay buffer storing Experience tuples ((StateType, policy_target, n_step_return)). It supports Prioritized Experience Replay (PER) via a SumTree, including prioritized sampling and priority updates, based on configuration. **Its content is saved and loaded via the TrieyeActor.**

## Exposed Interfaces

-   **Classes:**
    -   Trainer:
        -   __init__(nn_interface: NeuralNetwork, train_config: TrainConfig, env_config: EnvConfig)
        -   train_step(per_sample: PERBatchSample) -> Optional[Tuple[Dict[str, float], np.ndarray]]: Takes PER sample, returns raw loss info and TD errors.
        -   load_optimizer_state(state_dict: dict)
        -   get_current_lr() -> float
    -   ExperienceBuffer:
        -   __init__(config: TrainConfig)
        -   add(experience: Experience)
        -   add_batch(experiences: List[Experience])
        -   sample(batch_size: int, current_train_step: Optional[int] = None) -> Optional[PERBatchSample]: Samples batch, requires step for PER beta.
        -   update_priorities(tree_indices: np.ndarray, td_errors: np.ndarray): Updates priorities for PER.
        -   is_ready() -> bool
        -   __len__() -> int

## Dependencies

-   **[alphatriangle.config](../../config/README.md)**: TrainConfig, ModelConfig.
-   **trianglengin**: EnvConfig.
-   **[alphatriangle.nn](../../nn/README.md)**: NeuralNetwork.
-   **[alphatriangle.utils](../../utils/README.md)**: Types (Experience, PERBatchSample, StateType, etc.) and helpers (SumTree).
-   **torch**: Used heavily by Trainer.
-   **Standard Libraries:** typing, logging, collections.deque, numpy, random.

---

**Note:** Please keep this README updated when changing the responsibilities or interfaces of the Trainer or Buffer.
