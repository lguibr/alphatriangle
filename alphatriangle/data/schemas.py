from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Use relative import
from ..utils.types import Experience

arbitrary_types_config = ConfigDict(arbitrary_types_allowed=True)


class CheckpointData(BaseModel):
    """Pydantic model defining the structure of saved checkpoint data."""

    model_config = arbitrary_types_config

    run_name: str
    global_step: int = Field(..., ge=0)
    episodes_played: int = Field(..., ge=0)
    total_simulations_run: int = Field(..., ge=0)
    model_config_dict: dict[str, Any]
    env_config_dict: dict[str, Any]
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    stats_collector_state: dict[str, Any]


class BufferData(BaseModel):
    """Pydantic model defining the structure of saved buffer data."""

    model_config = arbitrary_types_config

    buffer_list: list[Experience]


class LoadedTrainingState(BaseModel):
    """Pydantic model representing the fully loaded state."""

    model_config = arbitrary_types_config

    checkpoint_data: CheckpointData | None = None
    buffer_data: BufferData | None = None


BufferData.model_rebuild(force=True)
LoadedTrainingState.model_rebuild(force=True)
