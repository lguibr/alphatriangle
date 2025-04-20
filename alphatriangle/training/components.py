from dataclasses import dataclass
from typing import TYPE_CHECKING

import ray

# Import EnvConfig from trianglengin
from trianglengin.config import EnvConfig

# Keep alphatriangle imports
from alphatriangle.config import (
    MCTSConfig,
    ModelConfig,
    PersistenceConfig,
    TrainConfig,
)
from alphatriangle.data import DataManager
from alphatriangle.nn import NeuralNetwork
from alphatriangle.rl import ExperienceBuffer, Trainer
from alphatriangle.stats import StatsCollectorActor  # Keep StatsCollectorActor import

if TYPE_CHECKING:
    pass  # No changes needed here


@dataclass
class TrainingComponents:
    """Holds the initialized core components needed for training."""

    nn: "NeuralNetwork"
    buffer: "ExperienceBuffer"
    trainer: "Trainer"
    data_manager: "DataManager"
    stats_collector_actor: ray.actor.ActorHandle | None  # Keep type hint
    train_config: "TrainConfig"
    env_config: EnvConfig  # Use trianglengin.EnvConfig
    model_config: "ModelConfig"
    mcts_config: "MCTSConfig"
    persist_config: "PersistenceConfig"
