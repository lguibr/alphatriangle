# File: alphatriangle/training/components.py
from dataclasses import dataclass
from typing import TYPE_CHECKING

# --- ADDED: Import ActorHandle ---
import ray

# --- END ADDED ---

if TYPE_CHECKING:
    # Keep ray import here as well for consistency if needed elsewhere
    # import ray

    from alphatriangle.config import (
        EnvConfig,
        MCTSConfig,
        ModelConfig,
        PersistenceConfig,
        TrainConfig,
    )
    from alphatriangle.data import DataManager
    from alphatriangle.nn import NeuralNetwork
    from alphatriangle.rl import ExperienceBuffer, Trainer

    # --- REMOVED: Import StatsCollectorActor class ---
    # from alphatriangle.stats import StatsCollectorActor
    # --- END REMOVED ---


@dataclass
class TrainingComponents:
    """Holds the initialized core components needed for training."""

    nn: "NeuralNetwork"
    buffer: "ExperienceBuffer"
    trainer: "Trainer"
    data_manager: "DataManager"
    # --- CORRECTED: Use ActorHandle type hint ---
    stats_collector_actor: ray.actor.ActorHandle | None
    # --- END CORRECTED ---
    train_config: "TrainConfig"
    env_config: "EnvConfig"
    model_config: "ModelConfig"
    mcts_config: "MCTSConfig"
    persist_config: "PersistenceConfig"
