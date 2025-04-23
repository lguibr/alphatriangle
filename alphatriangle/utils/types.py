# File: alphatriangle/utils/types.py
from collections.abc import Mapping

import numpy as np
from typing_extensions import TypedDict


class StateType(TypedDict):
    """
    Represents the processed state features input to the neural network.
    Contains numerical arrays derived from the raw GameState.
    """

    grid: np.ndarray  # (C, H, W) float32, e.g., occupancy, death cells
    other_features: np.ndarray  # (OtherFeatDim,) float32, e.g., shape info, game stats


# Action representation (integer index)
ActionType = int

# Policy target from MCTS (visit counts distribution)
# Mapping from action index to its probability (normalized visit count)
PolicyTargetMapping = Mapping[ActionType, float]


# REMOVED StepInfo TypedDict (no longer needed directly here)


Experience = tuple[StateType, PolicyTargetMapping, float]
# Represents one unit of experience stored in the replay buffer.
# 1. StateType: The processed features (grid, other_features) of the state s_t.
#               NOTE: This is NOT the raw GameState object.
# 2. PolicyTargetMapping: The MCTS-derived policy target pi(a|s_t) for state s_t.
# 3. float: The calculated N-step return G_t^n starting from state s_t, used
#           as the target for the value head during training.


# Batch of experiences for training
ExperienceBatch = list[Experience]


# Output type from the neural network's evaluate method (for MCTS interaction)
# 1. dict[ActionType, float]: Policy probabilities P(a|s) for the evaluated state.
# 2. float: The expected value V(s) calculated from the value distribution logits.
PolicyValueOutput = tuple[dict[int, float], float]


# REMOVED StatsCollectorData type alias


class PERBatchSample(TypedDict):
    """Output of the PER buffer's sample method."""

    batch: ExperienceBatch  # The sampled experiences
    indices: np.ndarray  # Tree indices of the sampled experiences (for priority update)
    weights: np.ndarray  # Importance sampling weights for the sampled experiences
