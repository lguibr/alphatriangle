import pygame
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_FONT_NAME = None
FALLBACK_FONT_NAME = "arial,freesans"


def load_single_font(name: Optional[str], size: int) -> Optional[pygame.font.Font]:
    """Loads a single font, handling potential errors."""
    try:
        font = pygame.font.SysFont(name, size)
        return font
    except Exception as e:
        logger.error(f"Error loading font '{name}' size {size}: {e}")
        if name != FALLBACK_FONT_NAME:
            logger.warning(f"Attempting fallback font: {FALLBACK_FONT_NAME}")
            try:
                font = pygame.font.SysFont(FALLBACK_FONT_NAME, size)
                logger.info(f"Loaded fallback font: {FALLBACK_FONT_NAME} size {size}")
                return font
            except Exception as e_fallback:
                logger.error(f"Fallback font failed: {e_fallback}")
                return None
        return None


def load_fonts(
    font_sizes: Optional[Dict[str, int]] = None,
) -> Dict[str, Optional[pygame.font.Font]]:
    """Loads standard game fonts."""
    if font_sizes is None:
        font_sizes = {
            "ui": 24,
            "score": 30,
            "help": 18,
            "title": 48,
        }

    fonts: Dict[str, Optional[pygame.font.Font]] = {}
    required_fonts = ["score", "help"]

    logger.info("Loading fonts...")
    for name, size in font_sizes.items():
        fonts[name] = load_single_font(DEFAULT_FONT_NAME, size)

    for name in required_fonts:
        if fonts.get(name) is None:
            logger.critical(
                f"Essential font '{name}' failed to load. Text rendering will be affected."
            )

    return fonts
