"""
Monte Carlo Tree Search (MCTS) module.
Provides the core algorithm and components for game tree search.
"""

from alphatriangle.config import MCTSConfig

from .core.node import Node
from .core.search import (
    MCTSExecutionError,
    run_mcts_simulations,
)
from .core.types import ActionPolicyMapping, ActionPolicyValueEvaluator
from .strategy.policy import get_policy_target, select_action_based_on_visits

__all__ = [
    # Core
    "Node",
    "run_mcts_simulations",
    "MCTSConfig",
    "ActionPolicyValueEvaluator",
    "ActionPolicyMapping",
    "MCTSExecutionError",
    # Strategy
    "select_action_based_on_visits",
    "get_policy_target",
]
