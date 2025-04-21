# File: alphatriangle/config/mcts_config.py
"""
Configuration for MCTS parameters specific to AlphaTriangle,
mirroring trimcts.SearchConfiguration for easy control.
"""

from pydantic import BaseModel, ConfigDict, Field
from trimcts import SearchConfiguration  # Import base config for reference

# Restore default simulations to a higher value for better play quality
DEFAULT_MAX_SIMULATIONS = 1024  # Increased from 64
DEFAULT_MAX_DEPTH = 32  # Can increase depth slightly too
DEFAULT_CPUCT = 1.5
DEFAULT_MCTS_BATCH_SIZE = 64  # Default batch size for network evals within MCTS


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

    # Batching for Network Evaluations within MCTS
    mcts_batch_size: int = Field(
        default=DEFAULT_MCTS_BATCH_SIZE,
        description="Number of leaf nodes to collect in C++ MCTS before calling network evaluate_batch.",
        gt=0,
    )

    # Use ConfigDict for Pydantic V2
    model_config = ConfigDict(validate_assignment=True)

    def to_trimcts_config(self) -> SearchConfiguration:
        """Converts this config to the trimcts.SearchConfiguration."""
        return SearchConfiguration(
            max_simulations=self.max_simulations,
            max_depth=self.max_depth,
            cpuct=self.cpuct,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_epsilon=self.dirichlet_epsilon,
            discount=self.discount,
            mcts_batch_size=self.mcts_batch_size,  # Pass batch size
        )


# Ensure model is rebuilt after changes
AlphaTriangleMCTSConfig.model_rebuild(force=True)
