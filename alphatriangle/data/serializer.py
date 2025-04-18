# File: alphatriangle/data/serializer.py
import json
import logging
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cloudpickle
import numpy as np
import torch
from pydantic import ValidationError

from ..utils.sumtree import SumTree
from .schemas import BufferData, CheckpointData

if TYPE_CHECKING:
    from torch.optim import Optimizer

    from ..rl.core.buffer import ExperienceBuffer

logger = logging.getLogger(__name__)


class Serializer:
    """Handles serialization and deserialization of training data."""

    def load_checkpoint(self, path: Path) -> CheckpointData | None:
        """Loads and validates checkpoint data from a file."""
        try:
            with path.open("rb") as f:
                loaded_data = cloudpickle.load(f)
            if isinstance(loaded_data, CheckpointData):
                # Pydantic automatically validates on load if type matches
                return loaded_data
            else:
                logger.error(
                    f"Loaded checkpoint file {path} did not contain a CheckpointData object (type: {type(loaded_data)})."
                )
                return None
        except ValidationError as e:
            logger.error(
                f"Pydantic validation failed for checkpoint {path}: {e}", exc_info=True
            )
            return None
        except FileNotFoundError:
            logger.warning(f"Checkpoint file not found: {path}")
            return None
        except Exception as e:
            logger.error(
                f"Error loading/deserializing checkpoint from {path}: {e}",
                exc_info=True,
            )
            return None

    def save_checkpoint(self, data: CheckpointData, path: Path):
        """Saves checkpoint data to a file using cloudpickle."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as f:
                cloudpickle.dump(data, f)
            logger.info(f"Checkpoint data saved to {path}")
        except Exception as e:
            logger.error(
                f"Failed to save checkpoint file to {path}: {e}", exc_info=True
            )
            raise  # Re-raise the exception

    def load_buffer(self, path: Path) -> BufferData | None:
        """Loads and validates buffer data from a file."""
        try:
            with path.open("rb") as f:
                loaded_data = cloudpickle.load(f)
            if isinstance(loaded_data, BufferData):
                # Perform basic validation on loaded experiences
                valid_experiences = []
                invalid_count = 0
                for i, exp in enumerate(loaded_data.buffer_list):
                    if (
                        isinstance(exp, tuple)
                        and len(exp) == 3
                        and isinstance(exp[0], dict)
                        and "grid" in exp[0]
                        and "other_features" in exp[0]
                        and isinstance(exp[0]["grid"], np.ndarray)
                        and isinstance(exp[0]["other_features"], np.ndarray)
                        and isinstance(exp[1], dict)
                        and isinstance(exp[2], float | int)
                    ):
                        valid_experiences.append(exp)
                    else:
                        invalid_count += 1
                        logger.warning(
                            f"Skipping invalid experience structure at index {i} in loaded buffer: {type(exp)}"
                        )
                if invalid_count > 0:
                    logger.warning(
                        f"Found {invalid_count} invalid experience structures in loaded buffer."
                    )
                loaded_data.buffer_list = valid_experiences
                return loaded_data
            else:
                logger.error(
                    f"Loaded buffer file {path} did not contain a BufferData object (type: {type(loaded_data)})."
                )
                return None
        except ValidationError as e:
            logger.error(
                f"Pydantic validation failed for buffer {path}: {e}", exc_info=True
            )
            return None
        except FileNotFoundError:
            logger.warning(f"Buffer file not found: {path}")
            return None
        except Exception as e:
            logger.error(
                f"Failed to load/deserialize experience buffer from {path}: {e}",
                exc_info=True,
            )
            return None

    def save_buffer(self, data: BufferData, path: Path):
        """Saves buffer data to a file using cloudpickle."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as f:
                cloudpickle.dump(data, f)
            logger.info(f"Buffer data saved to {path}")
        except Exception as e:
            logger.error(
                f"Error saving experience buffer to {path}: {e}", exc_info=True
            )
            raise  # Re-raise the exception

    def prepare_optimizer_state(self, optimizer: "Optimizer") -> dict[str, Any]:
        """Prepares optimizer state dictionary, moving tensors to CPU."""
        optimizer_state_cpu = {}
        try:
            optimizer_state_dict = optimizer.state_dict()

            def move_to_cpu(item):
                if isinstance(item, torch.Tensor):
                    return item.cpu()
                elif isinstance(item, dict):
                    return {k: move_to_cpu(v) for k, v in item.items()}
                elif isinstance(item, list):
                    return [move_to_cpu(elem) for elem in item]
                else:
                    return item

            optimizer_state_cpu = move_to_cpu(optimizer_state_dict)
        except Exception as e:
            logger.error(f"Could not prepare optimizer state for saving: {e}")
        return optimizer_state_cpu

    def prepare_buffer_data(self, buffer: "ExperienceBuffer") -> BufferData | None:
        """Prepares buffer data for saving, extracting experiences."""
        try:
            if buffer.use_per:
                if hasattr(buffer, "tree") and isinstance(buffer.tree, SumTree):
                    buffer_list = [
                        buffer.tree.data[i]
                        for i in range(buffer.tree.n_entries)
                        if buffer.tree.data[i] != 0
                    ]
                else:
                    logger.error("PER buffer tree is missing or invalid during save.")
                    return None
            else:
                buffer_list = list(buffer.buffer)

            # Basic validation before creating BufferData
            valid_experiences = []
            invalid_count = 0
            for i, exp in enumerate(buffer_list):
                if (
                    isinstance(exp, tuple)
                    and len(exp) == 3
                    and isinstance(exp[0], dict)
                    and "grid" in exp[0]
                    and "other_features" in exp[0]
                    and isinstance(exp[0]["grid"], np.ndarray)
                    and isinstance(exp[0]["other_features"], np.ndarray)
                    and isinstance(exp[1], dict)
                    and isinstance(exp[2], float | int)
                ):
                    valid_experiences.append(exp)
                else:
                    invalid_count += 1
                    logger.warning(
                        f"Skipping invalid experience structure at index {i} during save prep: {type(exp)}"
                    )
            if invalid_count > 0:
                logger.warning(
                    f"Found {invalid_count} invalid experience structures before saving buffer."
                )

            return BufferData(buffer_list=valid_experiences)
        except Exception as e:
            logger.error(f"Error preparing buffer data for saving: {e}")
            return None

    def save_config_json(self, configs: dict[str, Any], path: Path):
        """Saves the configuration dictionary as JSON."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w") as f:

                def default_serializer(obj):
                    if isinstance(obj, torch.Tensor | np.ndarray):
                        return "<tensor/array>"
                    if isinstance(obj, deque):
                        return list(obj)
                    try:
                        return obj.__dict__ if hasattr(obj, "__dict__") else str(obj)
                    except TypeError:
                        return f"<object of type {type(obj).__name__}>"

                json.dump(configs, f, indent=4, default=default_serializer)
            logger.info(f"Run config saved to {path}")
        except Exception as e:
            logger.error(
                f"Failed to save run config JSON to {path}: {e}", exc_info=True
            )
            raise
