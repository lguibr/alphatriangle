# File: alphatriangle/visualization/drawing/hud.py
from typing import Any

import pygame

from ..core import colors
from ..ui import ProgressBar


def render_hud(
    surface: pygame.Surface,
    mode: str,
    fonts: dict[str, pygame.font.Font | None],
    display_stats: dict[str, Any] | None = None,
) -> None:
    """
    Renders global information (like step count, worker status) at the bottom.
    Individual game scores are not shown here anymore.
    Help text is skipped in training_visual mode.
    """
    screen_w, screen_h = surface.get_size()
    help_font = fonts.get("help")
    stats_font = fonts.get("help")  # Use same font for stats line
    step_font = fonts.get("ui") or help_font  # Use UI font for step, fallback to help

    bottom_y = screen_h - 10  # Position from bottom

    stats_rect = None
    step_rect = None

    # --- Render Global Step as "Weight Updates" ---
    if step_font and display_stats:
        train_progress = display_stats.get("train_progress")
        global_step = (
            train_progress.current_steps
            if isinstance(train_progress, ProgressBar)  # Check type
            else display_stats.get("global_step", "?")
        )
        step_text = f"Weight Updates: {global_step}"
        step_surf = step_font.render(step_text, True, colors.YELLOW)
        step_rect = step_surf.get_rect(bottomleft=(15, bottom_y))
        surface.blit(step_surf, step_rect)

    # Render other global training stats if available, positioned after the step count
    if stats_font and display_stats and step_rect:
        stats_items = []
        episodes = display_stats.get("total_episodes", "?")
        sims = display_stats.get("total_simulations", "?")
        num_workers = display_stats.get("num_workers", "?")
        pending_tasks = display_stats.get("pending_tasks", "?")

        stats_items.append(f"Episodes: {episodes}")
        if isinstance(sims, int | float):
            sims_str = (
                f"{sims / 1e6:.2f}M"
                if sims >= 1e6
                else (f"{sims / 1e3:.1f}k" if sims >= 1000 else str(int(sims)))
            )
            stats_items.append(f"Sims: {sims_str}")
        else:
            stats_items.append(f"Sims: {sims}")

        stats_items.append(f"Workers: {pending_tasks}/{num_workers} busy")

        stats_text = " | ".join(stats_items)
        stats_surf = stats_font.render(stats_text, True, colors.CYAN)
        stats_rect = stats_surf.get_rect(bottomleft=(step_rect.right + 20, bottom_y))
        surface.blit(stats_surf, stats_rect)

    # --- CHANGED: Skip help text in training visual mode ---
    if help_font and mode != "training_visual":
        help_text = "[ESC] Quit"
        if mode == "play":
            help_text += " | [Click] Select/Place Shape"
        elif mode == "debug":
            help_text += " | [Click] Toggle Cell"
        # No need for training_visual case here anymore

        help_surf = help_font.render(help_text, True, colors.LIGHT_GRAY)
        help_rect = help_surf.get_rect(bottomright=(screen_w - 15, bottom_y))

        combined_left_width = (
            stats_rect.right if stats_rect else (step_rect.right if step_rect else 0)
        )
        if combined_left_width > help_rect.left - 20:
            help_rect.bottom = (
                stats_rect.top
                if stats_rect
                else (step_rect.top if step_rect else bottom_y)
            ) - 5
            help_rect.right = screen_w - 15

        surface.blit(help_surf, help_rect)
    # --- END CHANGED ---
