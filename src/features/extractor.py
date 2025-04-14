# File: src/features/extractor.py
import logging
from typing import TYPE_CHECKING, cast

import numpy as np

from src.config import ModelConfig

# Import GameState only for type checking
if TYPE_CHECKING:
    from src.environment import GameState

from src.utils.types import StateType

from . import grid_features

logger = logging.getLogger(__name__)


class GameStateFeatures:
    """Extracts features from GameState for NN input."""

    def __init__(self, game_state: "GameState", model_config: ModelConfig):
        self.gs = game_state
        self.env_config = game_state.env_config
        self.model_config = model_config

    def _get_grid_state(self) -> np.ndarray:
        """
        Returns grid state as a single channel numpy array.
        Values: 1.0 (occupied), 0.0 (empty playable), -1.0 (death cell).
        Orientation is implicit.
        Shape: (C, H, W) where C is GRID_INPUT_CHANNELS
        """
        rows, cols = self.env_config.ROWS, self.env_config.COLS
        # Initialize with correct number of channels
        grid_state = np.zeros(
            (self.model_config.GRID_INPUT_CHANNELS, rows, cols), dtype=np.float32
        )

        # Populate the first channel (or the only channel if C=1)
        for r in range(rows):
            for c in range(cols):
                tri = self.gs.grid_data.triangles[r][c]
                if tri.is_death:
                    grid_state[0, r, c] = -1.0
                elif tri.is_occupied:
                    grid_state[0, r, c] = 1.0
                else:
                    grid_state[0, r, c] = 0.0

        # Add more channels here if GRID_INPUT_CHANNELS > 1
        # Example: grid_state[1, :, :] = ... (e.g., color features)

        if not np.all(np.isfinite(grid_state)):
            logger.error(
                f"Non-finite values detected in extracted grid state! Min: {np.nanmin(grid_state)}, Max: {np.nanmax(grid_state)}, Mean: {np.nanmean(grid_state)}"
            )
            grid_state = np.nan_to_num(grid_state, nan=0.0, posinf=0.0, neginf=0.0)

        return grid_state

    def _get_shape_features(self) -> np.ndarray:
        """Extracts features for each shape slot."""
        num_slots = self.env_config.NUM_SHAPE_SLOTS

        FEATURES_PER_SHAPE_HERE = 7
        shape_feature_matrix = np.zeros(
            (num_slots, FEATURES_PER_SHAPE_HERE), dtype=np.float32
        )

        for i, shape in enumerate(self.gs.shapes):
            if shape and shape.triangles:
                n_tris = len(shape.triangles)
                ups = sum(1 for _, _, is_up in shape.triangles if is_up)
                downs = n_tris - ups
                min_r, min_c, max_r, max_c = shape.bbox()
                height = max_r - min_r + 1
                width_eff = (max_c - min_c + 1) * 0.75 + 0.25 if n_tris > 0 else 0

                # Populate features
                shape_feature_matrix[i, 0] = np.clip(n_tris / 5.0, 0, 1)
                shape_feature_matrix[i, 1] = ups / n_tris if n_tris > 0 else 0
                shape_feature_matrix[i, 2] = downs / n_tris if n_tris > 0 else 0
                shape_feature_matrix[i, 3] = np.clip(
                    height / self.env_config.ROWS, 0, 1
                )
                shape_feature_matrix[i, 4] = np.clip(
                    width_eff / self.env_config.COLS, 0, 1
                )
                shape_feature_matrix[i, 5] = np.clip(
                    ((min_r + max_r) / 2.0) / self.env_config.ROWS, 0, 1
                )
                shape_feature_matrix[i, 6] = np.clip(
                    ((min_c + max_c) / 2.0) / self.env_config.COLS, 0, 1
                )
        # Flatten the matrix to get a 1D array
        return shape_feature_matrix.flatten()

    def _get_shape_availability(self) -> np.ndarray:
        """Returns a binary vector indicating which shape slots are filled."""
        return np.array([1.0 if s else 0.0 for s in self.gs.shapes], dtype=np.float32)

    def _get_explicit_features(self) -> np.ndarray:
        """Extracts scalar features like score, heights, holes, etc."""
        EXPLICIT_FEATURES_DIM_HERE = 6
        features = np.zeros(EXPLICIT_FEATURES_DIM_HERE, dtype=np.float32)
        occupied = self.gs.grid_data._occupied_np
        death = self.gs.grid_data._death_np
        rows, cols = self.env_config.ROWS, self.env_config.COLS

        heights = grid_features.get_column_heights(occupied, death, rows, cols)
        holes = grid_features.count_holes(occupied, death, heights, rows, cols)
        bump = grid_features.get_bumpiness(heights)
        total_playable_cells = np.sum(~death)

        # Populate features
        features[0] = np.clip(self.gs.game_score / 100.0, -5.0, 5.0)
        features[1] = np.mean(heights) / rows if rows > 0 else 0
        features[2] = np.max(heights) / rows if rows > 0 else 0
        features[3] = holes / total_playable_cells if total_playable_cells > 0 else 0
        features[4] = (bump / (cols - 1)) / rows if cols > 1 and rows > 0 else 0
        features[5] = np.clip(self.gs.pieces_placed_this_episode / 100.0, 0, 1)

        # Ensure return type is ndarray and handle potential NaNs
        # Explicitly cast to satisfy mypy's no-any-return
        return cast(
            np.ndarray, np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        )

    def get_combined_other_features(self) -> np.ndarray:
        """Combines all non-grid features into a single flat vector."""
        shape_feats = self._get_shape_features()
        avail_feats = self._get_shape_availability()
        explicit_feats = self._get_explicit_features()
        combined = np.concatenate([shape_feats, avail_feats, explicit_feats])

        expected_dim = self.model_config.OTHER_NN_INPUT_FEATURES_DIM
        if combined.shape[0] != expected_dim:
            raise ValueError(
                f"Combined other_features dimension mismatch! Extracted {combined.shape[0]}, but ModelConfig expects {expected_dim}"
            )

        if not np.all(np.isfinite(combined)):
            logger.error(
                f"Non-finite values detected in combined other_features! Min: {np.nanmin(combined)}, Max: {np.nanmax(combined)}, Mean: {np.nanmean(combined)}"
            )
            combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)

        return combined.astype(np.float32)


def extract_state_features(
    game_state: "GameState", model_config: ModelConfig
) -> StateType:
    """
    Extracts and returns the state dictionary {grid, other_features} for NN input.
    Requires ModelConfig to ensure dimensions match the network's expectations.
    Includes validation for non-finite values.
    """
    extractor = GameStateFeatures(game_state, model_config)
    state_dict: StateType = {
        "grid": extractor._get_grid_state(),
        "other_features": extractor.get_combined_other_features(),
    }
    grid_feat = state_dict["grid"]
    other_feat = state_dict["other_features"]
    logger.debug(
        f"Extracted Features (State {game_state.current_step}): Grid(shape={grid_feat.shape}, min={grid_feat.min():.2f}, max={grid_feat.max():.2f}, mean={grid_feat.mean():.2f}), Other(shape={other_feat.shape}, min={other_feat.min():.2f}, max={other_feat.max():.2f}, mean={other_feat.mean():.2f})"
    )
    return state_dict
