import logging
from typing import TYPE_CHECKING

from ..core.action_codec import encode_action
from ..grid import logic as GridLogic

if TYPE_CHECKING:
    from ...utils.types import ActionType
    from ..core.game_state import GameState

logger = logging.getLogger(__name__)


def get_valid_actions(state: "GameState") -> list["ActionType"]:
    """
    Calculates and returns a list of all valid encoded action indices
    for the current game state.
    """
    valid_actions: list[ActionType] = []
    for shape_idx, shape in enumerate(state.shapes):
        if shape is None:
            continue

        for r in range(state.env_config.ROWS):
            for c in range(state.env_config.COLS):
                if GridLogic.can_place(state.grid_data, shape, r, c):
                    action_index = encode_action(shape_idx, r, c, state.env_config)
                    valid_actions.append(action_index)

    return valid_actions
