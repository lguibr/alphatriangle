# File: alphatriangle/training/components.py
from dataclasses import dataclass
from typing import TYPE_CHECKING

import ray

# Import EnvConfig from trianglengin's top level
from trianglengin import EnvConfig
from trieye import Serializer  # Import Serializer from trieye
from trimcts import SearchConfiguration

# Keep alphatriangle imports

if TYPE_CHECKING:
    from trieye import TrieyeConfig  # Import TrieyeConfig

    from alphatriangle.config import ModelConfig, TrainConfig
    from alphatriangle.nn import NeuralNetwork
    from alphatriangle.rl import ExperienceBuffer, Trainer


@dataclass
class TrainingComponents:
    """Holds the initialized core components needed for training."""

    nn: "NeuralNetwork"
    buffer: "ExperienceBuffer"
    trainer: "Trainer"
    trieye_actor: ray.actor.ActorHandle
    trieye_config: "TrieyeConfig"
    serializer: Serializer  # Added Serializer instance
    train_config: "TrainConfig"
    env_config: EnvConfig
    model_config: "ModelConfig"
    mcts_config: SearchConfiguration
    profile_workers: bool
