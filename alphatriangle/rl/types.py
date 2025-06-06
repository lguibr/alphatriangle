import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..utils.types import Experience, StateType

logger = logging.getLogger(__name__)

arbitrary_types_config = ConfigDict(arbitrary_types_allowed=True)


class SelfPlayResult(BaseModel):
    """Pydantic model for structuring results from a self-play worker."""

    model_config = arbitrary_types_config

    episode_experiences: list[Experience]
    final_score: float
    episode_steps: int
    trainer_step_at_episode_start: int

    total_simulations: int = Field(..., ge=0)
    avg_root_visits: float = Field(..., ge=0)
    avg_tree_depth: float = Field(..., ge=0)
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context from the episode (e.g., triangles_cleared).",
    )

    @model_validator(mode="after")
    def check_experience_structure(self) -> "SelfPlayResult":
        """Basic structural validation for experiences, including StateType."""
        invalid_count = 0
        valid_experiences = []
        for i, exp in enumerate(self.episode_experiences):
            is_valid = False
            reason = "Unknown structure"
            try:
                if isinstance(exp, tuple) and len(exp) == 3:
                    state_type: StateType = exp[0]
                    policy_map = exp[1]
                    value = exp[2]
                    if (
                        isinstance(state_type, dict)
                        and "grid" in state_type
                        and "other_features" in state_type
                        and isinstance(state_type["grid"], np.ndarray)
                        and isinstance(state_type["other_features"], np.ndarray)
                        and isinstance(policy_map, dict)
                        and isinstance(value, float | int)
                    ):
                        if np.all(np.isfinite(state_type["grid"])) and np.all(
                            np.isfinite(state_type["other_features"])
                        ):
                            is_valid = True
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
            # Use object.__setattr__ to modify the field within the validator
            object.__setattr__(self, "episode_experiences", valid_experiences)

        return self


SelfPlayResult.model_rebuild(force=True)
