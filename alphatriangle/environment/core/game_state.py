import logging
import random
from typing import TYPE_CHECKING

from ...config import EnvConfig
from ...utils.types import ActionType
from .. import shapes
from ..grid.grid_data import GridData
from ..logic.actions import get_valid_actions
from ..logic.step import execute_placement
from .action_codec import decode_action

if TYPE_CHECKING:
    from ...structs import Shape


logger = logging.getLogger(__name__)


class GameState:
    """
    Represents the mutable state of the game. Does not handle NN feature extraction
    or visualization/interaction-specific state.
    """

    def __init__(
        self, config: EnvConfig | None = None, initial_seed: int | None = None
    ):
        self.env_config = config if config else EnvConfig()  # type: ignore[call-arg]
        self._rng = (
            random.Random(initial_seed) if initial_seed is not None else random.Random()
        )

        self.grid_data: GridData = None  # type: ignore
        self.shapes: list[Shape | None] = []
        self.game_score: float = 0.0
        self.game_over: bool = False
        self.triangles_cleared_this_episode: int = 0
        self.pieces_placed_this_episode: int = 0
        self.current_step: int = 0

        self.reset()

    def reset(self):
        """Resets the game to the initial state."""
        self.grid_data = GridData(self.env_config)
        self.shapes = [None] * self.env_config.NUM_SHAPE_SLOTS
        self.game_score = 0.0
        self.triangles_cleared_this_episode = 0
        self.pieces_placed_this_episode = 0
        self.game_over = False
        self.current_step = 0

        # Call refill_shape_slots with the updated signature (no index)
        shapes.refill_shape_slots(self, self._rng)

        if not self.valid_actions():
            logger.warning(
                "Game is over immediately after reset (no valid initial moves)."
            )
            self.game_over = True

    def step(self, action_index: ActionType) -> tuple[float, bool]:
        """
        Performs one game step.
        Returns:
            Tuple[float, bool]: (reward, done)
        """
        if self.is_over():
            logger.warning("Attempted to step in a game that is already over.")
            return 0.0, True

        shape_idx, r, c = decode_action(action_index, self.env_config)
        reward = execute_placement(self, shape_idx, r, c, self._rng)
        self.current_step += 1

        if not self.game_over and not self.valid_actions():
            self.game_over = True
            logger.info(f"Game over detected after step {self.current_step}.")

        return reward, self.game_over

    def valid_actions(self) -> list[ActionType]:
        """Returns a list of valid encoded action indices."""
        return get_valid_actions(self)

    def is_over(self) -> bool:
        """Checks if the game is over."""
        return self.game_over

    def get_outcome(self) -> float:
        """Returns the terminal outcome value (e.g., final score). Used by MCTS."""
        if not self.is_over():
            logger.warning("get_outcome() called on a non-terminal state.")
            # Consider returning a default value or raising an error?
            # Returning current score might be misleading for MCTS if not terminal.
            # Let's return 0.0 as a neutral value if not over.
            return 0.0
        return self.game_score

    def copy(self) -> "GameState":
        """Creates a deep copy for simulations (e.g., MCTS)."""
        new_state = GameState.__new__(GameState)
        new_state.env_config = self.env_config
        new_state._rng = random.Random()
        new_state._rng.setstate(self._rng.getstate())
        new_state.grid_data = self.grid_data.deepcopy()
        new_state.shapes = [s.copy() if s else None for s in self.shapes]
        new_state.game_score = self.game_score
        new_state.game_over = self.game_over
        new_state.triangles_cleared_this_episode = self.triangles_cleared_this_episode
        new_state.pieces_placed_this_episode = self.pieces_placed_this_episode
        new_state.current_step = self.current_step
        return new_state

    def __str__(self) -> str:
        shape_strs = [str(s) if s else "None" for s in self.shapes]
        return f"GameState(Step:{self.current_step}, Score:{self.game_score:.1f}, Over:{self.is_over()}, Shapes:[{', '.join(shape_strs)}])"
