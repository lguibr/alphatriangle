# File: alphatriangle/config/mcts_config.py
from pydantic import BaseModel, Field, field_validator


class MCTSConfig(BaseModel):
    """
    Configuration for Monte Carlo Tree Search (Pydantic model).
    --- Tuned for Enhanced Search ---
    """

    # Increased simulations for better policy evaluation
    num_simulations: int = Field(default=256, ge=1)  # CHANGED
    # Slightly higher PUCT coefficient for more exploration bias
    puct_coefficient: float = Field(default=1.8, gt=0)  # CHANGED
    # Temperature controls exploration in action selection
    temperature_initial: float = Field(default=1.0, ge=0)
    temperature_final: float = Field(default=0.1, ge=0)
    # Anneal temperature over first 30k game steps (proportional to 100k total)
    temperature_anneal_steps: int = Field(default=30_000, ge=0)  # CHANGED
    # Dirichlet noise for root exploration
    dirichlet_alpha: float = Field(default=0.3, gt=0)
    dirichlet_epsilon: float = Field(default=0.25, ge=0, le=1.0)
    # Keep max search depth reasonable
    max_search_depth: int = Field(default=32, ge=1)

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
