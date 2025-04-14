import logging
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import pygame

if TYPE_CHECKING:
    from ..visualization.core.visualizer import Visualizer

logger = logging.getLogger(__name__)


def process_pygame_events(
    visualizer: "Visualizer",
) -> Generator[pygame.event.Event, Any, bool]:
    """
    Processes basic Pygame events like QUIT, ESCAPE, VIDEORESIZE.
    Yields other events for mode-specific handlers.
    Returns False via StopIteration value if the application should quit, True otherwise.
    """
    should_quit = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            logger.info("Received QUIT event.")
            should_quit = True
            break
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            logger.info("Received ESCAPE key press.")
            should_quit = True
            break
        if event.type == pygame.VIDEORESIZE:
            try:
                w, h = max(320, event.w), max(240, event.h)
                visualizer.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
                visualizer.layout_rects = None
                logger.info(f"Window resized to {w}x{h}")
            except pygame.error as e:
                logger.error(f"Error resizing window: {e}")
            yield event
        else:
            yield event
    return not should_quit
