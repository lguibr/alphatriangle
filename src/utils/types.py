# File: src/utils/types.py
# File: src/utils/types.py
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
Experience = tuple[StateType, PolicyTargetMapping, float]

# Batch of experiences for training
ExperienceBatch = list[Experience]

# Output type from the neural network's evaluate method
# (Policy Mapping, Value Estimate)
# Kept as Tuple for performance.
PolicyValueOutput = tuple[Mapping[ActionType, float], float]

# Type alias for the data structure holding collected statistics
# Maps metric name to a deque of (step, value) tuples
# Kept as Dict[Deque] internally in StatsCollectorActor, type alias is sufficient here.
StatsCollectorData = dict[str, deque[tuple[int, float]]]

# --- Pydantic Models for Data Transfer ---
# SelfPlayResult moved to src/rl/types.py to resolve circular import


# --- Prioritized Experience Replay Types ---
# TypedDict for the output of the PER buffer's sample method
class PERBatchSample(TypedDict):
    batch: ExperienceBatch
    indices: np.ndarray
    weights: np.ndarray
