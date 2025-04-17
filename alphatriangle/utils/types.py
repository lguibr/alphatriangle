# File: alphatriangle/utils/types.py
from collections import deque
from collections.abc import Mapping

import numpy as np
from typing_extensions import TypedDict


class StateType(TypedDict):
    grid: np.ndarray  # (C, H, W) float32
    other_features: np.ndarray  # (OtherFeatDim,) float32


# Action representation (integer index)
ActionType = int

# Policy target from MCTS (visit counts distribution)
# Mapping from action index to its probability (normalized visit count)
PolicyTargetMapping = Mapping[ActionType, float]

# Experience tuple stored in buffer
# NOW stores the extracted StateType (features) instead of the raw GameState object.
# Kept as Tuple for performance in buffer operations.
# --- CHANGE: Updated comment for n-step returns (used for distributional target) ---
# The third element (float) represents the calculated n-step return (G_t^n)
# starting from the state represented by the first element (StateType).
# This G_t^n is used by the Trainer to construct the target value distribution.
Experience = tuple[StateType, PolicyTargetMapping, float]
# --- END CHANGE ---

# Batch of experiences for training
ExperienceBatch = list[Experience]

# Output type from the neural network's evaluate method
# (Policy Mapping, Value Estimate)
# Kept as Tuple for performance.
# --- CHANGE: Updated comment for expected value ---
# The second element (float) is the EXPECTED value calculated from the
# predicted value distribution (used for MCTS). The Trainer uses the raw logits.
PolicyValueOutput = tuple[Mapping[ActionType, float], float]
# --- END CHANGE ---

# Type alias for the data structure holding collected statistics
# Maps metric name to a deque of (step, value) tuples
# Kept as Dict[Deque] internally in StatsCollectorActor, type alias is sufficient here.
StatsCollectorData = dict[str, deque[tuple[int, float]]]

# --- Pydantic Models for Data Transfer ---
# SelfPlayResult moved to alphatriangle/rl/types.py to resolve circular import


# --- Prioritized Experience Replay Types ---
# TypedDict for the output of the PER buffer's sample method
class PERBatchSample(TypedDict):
    batch: ExperienceBatch
    indices: np.ndarray
    weights: np.ndarray
