# File: src/training/components.py
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
    from alphatriangle.stats import StatsCollectorActor


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
