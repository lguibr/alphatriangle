# File: src/mcts/__init__.py
"""
Monte Carlo Tree Search (MCTS) module.
Provides the core algorithm and components for game tree search.
"""

# Core MCTS components
from .core.node import Node
from .core.search import (
    run_mcts_simulations,
    MCTSExecutionError,
)  # Import the exception

# Change: Import MCTSConfig from the central config location
from src.config import MCTSConfig
from .core.types import ActionPolicyValueEvaluator, ActionPolicyMapping

# Action selection and policy generation strategies
from .strategy.policy import select_action_based_on_visits, get_policy_target

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
