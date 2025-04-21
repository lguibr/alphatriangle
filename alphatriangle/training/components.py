from dataclasses import dataclass
from typing import TYPE_CHECKING

import ray

# Import EnvConfig from trianglengin's top level
from trianglengin import EnvConfig  # UPDATED IMPORT
from trimcts import SearchConfiguration

# Keep alphatriangle imports

if TYPE_CHECKING:
    # Keep AlphaTriangleMCTSConfig for potential future use if needed
    from alphatriangle.config import (
        ModelConfig,
        PersistenceConfig,
        TrainConfig,
    )
    from alphatriangle.data import DataManager
    from alphatriangle.nn import NeuralNetwork
    from alphatriangle.rl import ExperienceBuffer, Trainer

    pass


@dataclass
class TrainingComponents:
    """Holds the initialized core components needed for training."""

    nn: "NeuralNetwork"
    buffer: "ExperienceBuffer"
    trainer: "Trainer"
    data_manager: "DataManager"
    stats_collector_actor: ray.actor.ActorHandle | None
    train_config: "TrainConfig"
    env_config: EnvConfig  # Uses trianglengin.EnvConfig
    model_config: "ModelConfig"
    mcts_config: SearchConfiguration  # Use the trimcts config type here
    persist_config: "PersistenceConfig"
