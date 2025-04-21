# File: alphatriangle/config/mcts_config.py
"""
Configuration for MCTS parameters specific to AlphaTriangle,
mirroring trimcts.SearchConfiguration for easy control.
"""

from pydantic import BaseModel, ConfigDict, Field

# Temporarily reduce simulations for faster profiling
DEFAULT_MAX_SIMULATIONS = 64  # CHANGED TEMPORARILY
DEFAULT_MAX_DEPTH = 16  # Keep depth lower for faster profiling too
DEFAULT_CPUCT = 1.5


class AlphaTriangleMCTSConfig(BaseModel):
    """MCTS Search Configuration managed within AlphaTriangle."""

    # Core Search Parameters
    max_simulations: int = Field(
        default=DEFAULT_MAX_SIMULATIONS,
        description="Maximum number of MCTS simulations per move.",
        gt=0,
    )
    max_depth: int = Field(
        default=DEFAULT_MAX_DEPTH,
        description="Maximum depth for tree traversal during simulation.",
        gt=0,
    )

    # UCT Parameters (AlphaZero style)
    cpuct: float = Field(
        default=DEFAULT_CPUCT,
        description="Constant determining the level of exploration (PUCT).",
    )

    # Dirichlet Noise (for root node exploration)
    dirichlet_alpha: float = Field(
        default=0.3, description="Alpha parameter for Dirichlet noise.", ge=0
    )
    dirichlet_epsilon: float = Field(
        default=0.25,
        description="Weight of Dirichlet noise in root prior probabilities.",
        ge=0,
        le=1.0,
    )

    # Discount Factor (Primarily for MuZero/Value Propagation)
    discount: float = Field(
        default=1.0,
        description="Discount factor (gamma) for future rewards/values.",
        ge=0.0,
        le=1.0,
    )

    # Use ConfigDict for Pydantic V2
    model_config = ConfigDict(validate_assignment=True)
