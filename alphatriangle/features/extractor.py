# File: alphatriangle/features/extractor.py
import logging
from typing import TYPE_CHECKING, cast

import numpy as np

from ..config import ModelConfig
from ..utils.types import StateType
from . import grid_features  # Keep this import

if TYPE_CHECKING:
    from ..environment import GameState


logger = logging.getLogger(__name__)


class GameStateFeatures:
    """Extracts features from GameState for NN input. Reads from GridData NumPy arrays."""

    def __init__(self, game_state: "GameState", model_config: ModelConfig):
        self.gs = game_state
        self.env_config = game_state.env_config
        self.model_config = model_config
        # Get direct references to NumPy arrays for efficiency
        self.occupied_np = game_state.grid_data._occupied_np
        self.death_np = game_state.grid_data._death_np
        # self.color_id_np = game_state.grid_data._color_id_np # Not used in current features

    def _get_grid_state(self) -> np.ndarray:
        """
        Returns grid state as a single channel numpy array based on NumPy arrays.
        Values: 1.0 (occupied playable), 0.0 (empty playable), -1.0 (death cell).
        Shape: (C, H, W) where C is GRID_INPUT_CHANNELS
        """
        rows, cols = self.env_config.ROWS, self.env_config.COLS
        # Initialize with 0.0 (empty playable)
        grid_state: np.ndarray = np.zeros(
            (self.model_config.GRID_INPUT_CHANNELS, rows, cols), dtype=np.float32
        )

        # Mark occupied playable cells as 1.0
        playable_occupied_mask = self.occupied_np & ~self.death_np
        grid_state[0, playable_occupied_mask] = 1.0

        # Mark death cells as -1.0
        grid_state[0, self.death_np] = -1.0

        # No need for the loop or isfinite check here if input arrays are guaranteed finite

        return grid_state

    def _get_shape_features(self) -> np.ndarray:
        """Extracts features for each shape slot. (No change needed here)"""
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
        """Returns a binary vector indicating which shape slots are filled. (No change needed)"""
        return np.array([1.0 if s else 0.0 for s in self.gs.shapes], dtype=np.float32)

    def _get_explicit_features(self) -> np.ndarray:
        """
        Extracts scalar features like score, heights, holes, etc.
        Uses GridData NumPy arrays directly.
        """
        EXPLICIT_FEATURES_DIM_HERE = 6
        features = np.zeros(EXPLICIT_FEATURES_DIM_HERE, dtype=np.float32)
        # Use the direct references stored in self
        occupied = self.occupied_np
        death = self.death_np
        rows, cols = self.env_config.ROWS, self.env_config.COLS

        # Pass NumPy arrays directly to grid_features functions
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

        return cast(
            "np.ndarray", np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        )

    def get_combined_other_features(self) -> np.ndarray:
        """Combines all non-grid features into a single flat vector."""
        shape_feats = self._get_shape_features()
        avail_feats = self._get_shape_availability()
        explicit_feats = self._get_explicit_features()
        combined = np.concatenate([shape_feats, avail_feats, explicit_feats])

        expected_dim = self.model_config.OTHER_NN_INPUT_FEATURES_DIM
        if combined.shape[0] != expected_dim:
            # Log error instead of raising ValueError immediately during feature extraction
            logger.error(
                f"Combined other_features dimension mismatch! Extracted {combined.shape[0]}, but ModelConfig expects {expected_dim}. Padding/truncating."
            )
            # Pad or truncate to match expected dimension
            if combined.shape[0] < expected_dim:
                padding = np.zeros(
                    expected_dim - combined.shape[0], dtype=combined.dtype
                )
                combined = np.concatenate([combined, padding])
            else:
                combined = combined[:expected_dim]

        if not np.all(np.isfinite(combined)):
            logger.error(
                f"Non-finite values detected in combined other_features! Min: {np.nanmin(combined)}, Max: {np.nanmax(combined)}, Mean: {np.nanmean(combined)}"
            )
            combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)

        return cast("np.ndarray", combined.astype(np.float32))


def extract_state_features(
    game_state: "GameState", model_config: ModelConfig
) -> StateType:
    """
    Extracts and returns the state dictionary {grid, other_features} for NN input.
    Requires ModelConfig to ensure dimensions match the network's expectations.
    Includes validation for non-finite values. Now reads from GridData NumPy arrays.
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
