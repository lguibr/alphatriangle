from collections import deque
from collections.abc import Mapping

import numpy as np
from typing_extensions import TypedDict

# Define a type alias for the geometric representation of a shape
# List of triangles, where each triangle is (row, col, is_up)
ShapeGeometry = list[tuple[int, int, bool]]
# Store geometry and color ID
SerializableShapeInfo = tuple[ShapeGeometry, int]


class StateType(TypedDict):
    """
    Represents the processed state features input to the neural network
    and stored in the replay buffer.
    """

    grid: np.ndarray  # (C, H, W) float32, e.g., occupancy, death cells
    other_features: np.ndarray  # (OtherFeatDim,) float32, e.g., shape info, game stats
    # ADDED: Store geometry for visualization
    # List corresponds to slots; stores (triangle_list, color_id) or None if slot empty
    available_shapes_geometry: list[SerializableShapeInfo | None]


# Action representation (integer index)
ActionType = int

# Policy target from MCTS (visit counts distribution)
# Mapping from action index to its probability (normalized visit count)
PolicyTargetMapping = Mapping[ActionType, float]


class StepInfo(TypedDict, total=False):
    """Dictionary to hold various step counters associated with a metric."""

    global_step: int  # Overall training step count
    buffer_size: int  # Current size of the experience buffer
    game_step_index: int  # Index within a self-play episode


Experience = tuple[StateType, PolicyTargetMapping, float]
# Represents one unit of experience stored in the replay buffer.
# 1. StateType: The processed features of the state s_t.
#               Includes 'grid' (occupancy), 'other_features' (numerical),
#               and 'available_shapes_geometry' (visual info for shapes in slots).
# 2. PolicyTargetMapping: The MCTS-derived policy target pi(a|s_t) for state s_t.
# 3. float: The calculated N-step return G_t^n starting from state s_t, used
#           as the target for the value head during training.


# Batch of experiences for training
ExperienceBatch = list[Experience]


# Output type from the neural network's evaluate method (for MCTS interaction)
# 1. dict[ActionType, float]: Policy probabilities P(a|s) for the evaluated state.
# 2. float: The expected value V(s) calculated from the value distribution logits.
PolicyValueOutput = tuple[dict[ActionType, float], float]


# Type alias for the data structure holding collected statistics
# Maps metric name to a deque of (step_info_dict, value) tuples
StatsCollectorData = dict[str, deque[tuple[StepInfo, float]]]


class PERBatchSample(TypedDict):
    """Output of the PER buffer's sample method."""

    batch: ExperienceBatch  # The sampled experiences
    indices: np.ndarray  # Tree indices of the sampled experiences (for priority update)
    weights: np.ndarray  # Importance sampling weights for the sampled experiences
