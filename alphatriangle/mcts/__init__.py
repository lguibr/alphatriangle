# File: src/mcts/__init__.py
"""
Monte Carlo Tree Search (MCTS) module.
Provides the core algorithm and components for game tree search.
"""

# Core MCTS components
# Change: Import MCTSConfig from the central config location
from alphatriangle.config import MCTSConfig

from .core.node import Node
from .core.search import (
    MCTSExecutionError,
    run_mcts_simulations,
)  # Import the exception
from .core.types import ActionPolicyMapping, ActionPolicyValueEvaluator

# Action selection and policy generation strategies
from .strategy.policy import get_policy_target, select_action_based_on_visits

__all__ = [
    # Core
    "Node",
    "run_mcts_simulations",
    "MCTSConfig",  # Export Pydantic MCTSConfig
    "ActionPolicyValueEvaluator",
    "ActionPolicyMapping",
    "MCTSExecutionError",  # Export the exception
    # Strategy
    "select_action_based_on_visits",
    "get_policy_target",
]
