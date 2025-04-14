import logging
from typing import cast

logger = logging.getLogger(__name__)


def normalize_color_for_matplotlib(
    color_tuple_0_255: tuple[int, int, int],
) -> tuple[float, float, float]:
    """Converts RGB tuple (0-255) to Matplotlib format (0.0-1.0)."""
    if isinstance(color_tuple_0_255, tuple) and len(color_tuple_0_255) == 3:
        # Ensure values are within 0-255 before dividing
        valid_color = tuple(max(0, min(255, c)) for c in color_tuple_0_255)
        # Cast the result to the expected precise tuple type
        return cast("tuple[float, float, float]", tuple(c / 255.0 for c in valid_color))
    logger.warning(
        f"Invalid color format for normalization: {color_tuple_0_255}, returning black."
    )
    return (0.0, 0.0, 0.0)
