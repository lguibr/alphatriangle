import logging

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..utils.types import Experience

logger = logging.getLogger(__name__)

arbitrary_types_config = ConfigDict(arbitrary_types_allowed=True)


class SelfPlayResult(BaseModel):
    """Pydantic model for structuring results from a self-play worker."""

    model_config = arbitrary_types_config

    episode_experiences: list[Experience]
    final_score: float
    episode_steps: int

    total_simulations: int = Field(..., ge=0)
    avg_root_visits: float = Field(..., ge=0)
    avg_tree_depth: float = Field(..., ge=0)

    @model_validator(mode="after")
    def check_experience_structure(self) -> "SelfPlayResult":
        """Basic structural validation for experiences."""
        invalid_count = 0
        valid_experiences = []
        # Rename unused loop variable 'i' to '_i'
        for _i, exp in enumerate(self.episode_experiences):
            is_valid = False
            if isinstance(exp, tuple) and len(exp) == 3:
                state_type, policy_map, value = exp
                # Combine nested if statements
                if (
                    isinstance(state_type, dict)
                    and "grid" in state_type
                    and "other_features" in state_type
                    and isinstance(state_type["grid"], np.ndarray)
                    and isinstance(state_type["other_features"], np.ndarray)
                    and isinstance(policy_map, dict)
                    # Use isinstance with | for multiple types
                    and isinstance(value, float | int)
                    # Basic check for NaN/inf in features
                    and np.all(np.isfinite(state_type["grid"]))
                    and np.all(np.isfinite(state_type["other_features"]))
                ):
                    is_valid = True

            if is_valid:
                valid_experiences.append(exp)
            else:
                invalid_count += 1
                # Log only once per validation failure type if needed
                # logger.warning(f"SelfPlayResult validation: Invalid experience structure at index {i}: {type(exp)}")

        if invalid_count > 0:
            logger.warning(
                f"SelfPlayResult validation: Found {invalid_count} invalid experience structures. Keeping only valid ones."
            )
            # Note: Modifying self within validator is generally discouraged,
            # but here we filter invalid data before it propagates.
            # A cleaner approach might be a separate validation function called after creation.
            # However, for immediate use, this ensures the validated object has valid experiences.
            object.__setattr__(
                self, "episode_experiences", valid_experiences
            )  # Use object.__setattr__ to bypass Pydantic's immutability during validation

        return self


SelfPlayResult.model_rebuild(force=True)
