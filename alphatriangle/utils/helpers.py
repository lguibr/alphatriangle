# File: alphatriangle/utils/helpers.py
import logging
import random
from typing import cast

import numpy as np
import torch

logger = logging.getLogger(__name__)


def get_device(device_preference: str = "auto") -> torch.device:
    """
    Gets the appropriate torch device based on preference and availability.
    Prioritizes MPS on Mac if 'auto' is selected.
    """
    if device_preference == "cuda" and torch.cuda.is_available():
        logger.info("Using CUDA device.")
        return torch.device("cuda")
    # --- CHANGED: Prioritize MPS if available and preferred/auto ---
    if (
        device_preference in ["auto", "mps"]
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()  # Check if MPS is built
    ):
        logger.info(f"Using MPS device (Preference: {device_preference}).")
        return torch.device("mps")
    # --- END CHANGED ---
    if device_preference == "cpu":
        logger.info("Using CPU device.")
        return torch.device("cpu")

    # Auto-detection fallback (after MPS check)
    if torch.cuda.is_available():
        logger.info("Auto-selected CUDA device.")
        return torch.device("cuda")
    # Check MPS again in fallback (should have been caught above if available)
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        logger.info("Auto-selected MPS device.")
        return torch.device("mps")

    logger.info("Auto-selected CPU device.")
    return torch.device("cpu")


def set_random_seeds(seed: int = 42):
    """Sets random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    # Use NumPy's recommended way to seed the global RNG state if needed,
    # or preferably use a Generator instance. For simplicity here, we seed global.
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Optional: Set deterministic algorithms for CuDNN
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    # --- ADDED: Seed MPS if available ---
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            # Use torch.mps.manual_seed if available (newer PyTorch versions)
            if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
                torch.mps.manual_seed(seed)  # type: ignore
            else:
                # Fallback for older versions if needed, though less common
                # torch.manual_seed(seed) might cover MPS indirectly in some versions
                pass
        except Exception as e:
            logger.warning(f"Could not set MPS seed: {e}")
    # --- END ADDED ---
    logger.info(f"Set random seeds to {seed}")


def format_eta(seconds: float | None) -> str:
    """Formats seconds into a human-readable HH:MM:SS or MM:SS string."""
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "N/A"
    if seconds > 3600 * 24 * 30:
        return ">1 month"

    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)

    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def normalize_color_for_matplotlib(
    color_tuple_0_255: tuple[int, int, int],
) -> tuple[float, float, float]:
    """Converts RGB tuple (0-255) to Matplotlib format (0.0-1.0)."""
    if isinstance(color_tuple_0_255, tuple) and len(color_tuple_0_255) == 3:
        valid_color = tuple(max(0, min(255, c)) for c in color_tuple_0_255)
        return cast("tuple[float, float, float]", tuple(c / 255.0 for c in valid_color))
    logger.warning(
        f"Invalid color format for normalization: {color_tuple_0_255}, returning black."
    )
    return (0.0, 0.0, 0.0)
