from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
from collections import deque
import numpy as np

from src.utils.types import Experience, StateType

arbitrary_types_config = ConfigDict(arbitrary_types_allowed=True)


class CheckpointData(BaseModel):
    """Pydantic model defining the structure of saved checkpoint data."""

    model_config = arbitrary_types_config

    run_name: str
    global_step: int = Field(..., ge=0)
    episodes_played: int = Field(..., ge=0)
    total_simulations_run: int = Field(..., ge=0)
    model_config_dict: Dict[str, Any]
    env_config_dict: Dict[str, Any]
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Dict[str, Any]
    stats_collector_state: Dict[str, Any]


class BufferData(BaseModel):
    """Pydantic model defining the structure of saved buffer data."""

    model_config = arbitrary_types_config

    buffer_list: List[Experience]


class LoadedTrainingState(BaseModel):
    """Pydantic model representing the fully loaded state."""

    model_config = arbitrary_types_config

    checkpoint_data: Optional[CheckpointData] = None
    buffer_data: Optional[BufferData] = None


BufferData.model_rebuild(force=True)
LoadedTrainingState.model_rebuild(force=True)
