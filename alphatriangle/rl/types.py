# File: alphatriangle/rl/types.py
import logging

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..utils.types import Experience, StateType  # Import StateType

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
        """Basic structural validation for experiences, including StateType."""
        invalid_count = 0
        valid_experiences = []
        # Rename unused loop variable 'i' to '_i'
        for i, exp in enumerate(self.episode_experiences):
            is_valid = False
            reason = "Unknown structure"
            try:
                if isinstance(exp, tuple) and len(exp) == 3:
                    state_type: StateType = exp[0]
                    policy_map = exp[1]
                    value = exp[2]
                    # Combine nested if statements
                    if (
                        isinstance(state_type, dict)
                        and "grid" in state_type
                        and "other_features" in state_type
                        and "available_shapes_geometry" in state_type  # Check new key
                        and isinstance(state_type["grid"], np.ndarray)
                        and isinstance(state_type["other_features"], np.ndarray)
                        and isinstance(
                            state_type["available_shapes_geometry"], list
                        )  # Check list type
                        and isinstance(policy_map, dict)
                        # Use isinstance with | for multiple types
                        and isinstance(value, float | int)
                    ):
                        # Basic check for NaN/inf in features
                        if np.all(np.isfinite(state_type["grid"])) and np.all(
                            np.isfinite(state_type["other_features"])
                        ):
                            # Basic check for geometry structure (list of Optional[Tuple[List, int]])
                            geom_valid = True
                            for item in state_type["available_shapes_geometry"]:
                                # Combine nested if using 'and'
                                if item is not None and not (
                                    isinstance(item, tuple)
                                    and len(item) == 2
                                    and isinstance(item[0], list)
                                    and isinstance(item[1], int)
                                ):
                                    geom_valid = False
                                    reason = "Invalid geometry item structure"
                                    break
                            if geom_valid:
                                is_valid = True
                            else:
                                reason = (
                                    reason
                                    or "Invalid available_shapes_geometry structure"
                                )
                        else:
                            reason = "Non-finite features"
                    else:
                        reason = f"Incorrect types or missing keys: state_keys={list(state_type.keys()) if isinstance(state_type, dict) else type(state_type)}, policy={type(policy_map)}, value={type(value)}"
                else:
                    reason = f"Not a tuple of length 3: type={type(exp)}, len={len(exp) if isinstance(exp, tuple) else 'N/A'}"
            except Exception as e:
                reason = f"Validation exception: {e}"
                logger.error(
                    f"SelfPlayResult validation: Exception validating experience {i}: {e}",
                    exc_info=True,
                )

            if is_valid:
                valid_experiences.append(exp)
            else:
                invalid_count += 1
                logger.warning(
                    f"SelfPlayResult validation: Invalid experience structure at index {i}. Reason: {reason}. Data: {exp}"
                )

        if invalid_count > 0:
            logger.warning(
                f"SelfPlayResult validation: Found {invalid_count} invalid experience structures out of {len(self.episode_experiences)}. Keeping only valid ones."
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
