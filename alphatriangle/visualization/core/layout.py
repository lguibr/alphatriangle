# File: alphatriangle/visualization/core/layout.py
# Changes:
# - Reduced top_area_h ratio from 0.35 to 0.2 for training layout.

import logging

import pygame

from ...config import VisConfig

logger = logging.getLogger(__name__)


def calculate_interactive_layout(
    screen_width: int, screen_height: int, vis_config: VisConfig
) -> dict[str, pygame.Rect]:
    """
    Calculates layout rectangles for interactive modes (play/debug).
    Places grid on the left and preview on the right.
    """
    sw, sh = screen_width, screen_height
    pad = vis_config.PADDING
    hud_h = vis_config.HUD_HEIGHT
    preview_w = vis_config.PREVIEW_AREA_WIDTH

    available_h = max(0, sh - hud_h - 2 * pad)
    available_w = max(0, sw - 3 * pad)  # 3 pads: left, between, right

    grid_w = max(0, available_w - preview_w)
    grid_h = available_h

    grid_rect = pygame.Rect(pad, pad, grid_w, grid_h)
    preview_rect = pygame.Rect(grid_rect.right + pad, pad, preview_w, grid_h)

    screen_rect = pygame.Rect(0, 0, sw, sh)
    grid_rect = grid_rect.clip(screen_rect)
    preview_rect = preview_rect.clip(screen_rect)

    logger.debug(
        f"Interactive Layout calculated: Grid={grid_rect}, Preview={preview_rect}"
    )

    return {
        "grid": grid_rect,
        "preview": preview_rect,
    }


def calculate_training_layout(
    screen_width: int,
    screen_height: int,
    vis_config: VisConfig,
    bottom_margin: int = 0,
) -> dict[str, pygame.Rect]:
    """
    Calculates layout rectangles for training visualization mode.
    Splits screen between worker grid (top) and stats area (bottom).
    Gives more space to the stats/plots area.
    """
    sw, sh = screen_width, screen_height
    pad = vis_config.PADDING
    plot_internal_padding = 15

    hud_h = vis_config.HUD_HEIGHT
    total_available_h = max(0, sh - hud_h - 2 * pad)

    # --- CHANGED: Reduced height ratio for worker grid ---
    top_area_h = int(total_available_h * 0.20)  # Reduced from 0.35
    # --- END CHANGED ---
    top_area_w = sw - 2 * pad

    worker_grid_rect = pygame.Rect(pad, pad, top_area_w, top_area_h)

    stats_area_y = worker_grid_rect.bottom + pad
    stats_area_w = sw - 2 * pad

    stats_area_h = max(0, sh - stats_area_y - pad - hud_h)
    stats_area_rect = pygame.Rect(pad, stats_area_y, stats_area_w, stats_area_h)

    plot_area_x = stats_area_rect.left + plot_internal_padding
    plot_area_y = stats_area_rect.top + plot_internal_padding
    plot_area_w = max(0, stats_area_rect.width - 2 * plot_internal_padding)
    plot_area_h = max(
        0, stats_area_rect.height - bottom_margin - pad - 2 * plot_internal_padding
    )

    plot_rect = pygame.Rect(plot_area_x, plot_area_y, plot_area_w, plot_area_h)

    screen_rect = pygame.Rect(0, 0, sw, sh)
    worker_grid_rect = worker_grid_rect.clip(screen_rect)
    stats_area_rect = stats_area_rect.clip(screen_rect)
    plot_rect = plot_rect.clip(screen_rect)

    logger.debug(
        f"Training Layout calculated: WorkerGrid={worker_grid_rect}, StatsArea={stats_area_rect}, PlotRect={plot_rect}"
    )

    return {
        "worker_grid": worker_grid_rect,
        "stats_area": stats_area_rect,
        "plots": plot_rect,
    }


calculate_layout = calculate_training_layout
