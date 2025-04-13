# File: src/training/components.py
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.nn import NeuralNetwork
    from src.rl import ExperienceBuffer, Trainer
    from src.data import DataManager
    from src.stats import StatsCollectorActor
    from src.config import (
        TrainConfig,
        EnvConfig,
        ModelConfig,
        MCTSConfig,
        PersistenceConfig,
    )


@dataclass
class TrainingComponents:
    """Holds the initialized core components needed for training."""

    nn: "NeuralNetwork"
    buffer: "ExperienceBuffer"
    trainer: "Trainer"
    data_manager: "DataManager"
    stats_collector_actor: "StatsCollectorActor"
    train_config: "TrainConfig"
    env_config: "EnvConfig"
    model_config: "ModelConfig"
    mcts_config: "MCTSConfig"
    persist_config: "PersistenceConfig"
