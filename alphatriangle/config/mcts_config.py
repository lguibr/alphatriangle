# File: alphatriangle/config/mcts_config.py
from pydantic import BaseModel, Field, field_validator


class MCTSConfig(BaseModel):
    """
    Configuration for Monte Carlo Tree Search (Pydantic model).
    --- TUNED FOR INCREASED EXPLORATION & DEPTH ---
    """

    num_simulations: int = Field(default=2048, ge=1)
    puct_coefficient: float = Field(default=2.0, gt=0)
    temperature_initial: float = Field(default=1.0, ge=0)
    temperature_final: float = Field(default=0.1, ge=0)
    temperature_anneal_steps: int = Field(default=100, ge=0)
    dirichlet_alpha: float = Field(default=0.3, gt=0)
    dirichlet_epsilon: float = Field(default=0.25, ge=0, le=1.0)
    max_search_depth: int = Field(default=64, ge=1)

    @field_validator("temperature_final")
    @classmethod
    def check_temp_final_le_initial(cls, v: float, info) -> float:
        data = info.data if info.data else info.values
        initial_temp = data.get("temperature_initial")
        if initial_temp is not None and v > initial_temp:
            raise ValueError(
                "temperature_final cannot be greater than temperature_initial"
            )
        return v


MCTSConfig.model_rebuild(force=True)
