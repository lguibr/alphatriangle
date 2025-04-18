# File: alphatriangle/visualization/ui/progress_bar.py
# Changes:
# - Modify render text logic: If info_line is provided, prepend default progress info.
# - Cast return type of _get_pulsing_color to satisfy mypy.

import math
import random
import time
from typing import Any, cast  # Added cast

import pygame

from ...utils import format_eta
from ..core import colors


class ProgressBar:
    """
    A reusable progress bar component for visualization.
    Handles overflow by cycling colors and displaying actual progress count.
    Includes rounded corners and subtle pulsing effect.
    Can display a custom info line alongside default progress text.
    """

    def __init__(
        self,
        entity_title: str,
        total_steps: int,
        start_time: float | None = None,
        initial_steps: int = 0,
        initial_color: tuple[int, int, int] = colors.BLUE,
    ):
        self.entity_title = entity_title
        self._original_total_steps = max(
            1, total_steps if total_steps is not None else 1
        )
        self.initial_steps = max(0, initial_steps)
        self.current_steps = self.initial_steps
        self.start_time = start_time if start_time is not None else time.time()
        self._last_step_time = self.start_time
        self._step_times: list[float] = []
        self.extra_data: dict[str, Any] = {}
        self._current_bar_color = initial_color
        self._last_cycle = -1
        self._rng = random.Random()
        self._pulse_phase = random.uniform(0, 2 * math.pi)

    def add_steps(self, steps_added: int):
        """Adds steps to the progress bar's current count."""
        if steps_added <= 0:
            return
        self.current_steps += steps_added
        self._check_color_cycle()

    def set_current_steps(self, steps: int):
        """Directly sets the current step count."""
        self.current_steps = max(0, steps)
        self._check_color_cycle()

    def _check_color_cycle(self):
        """Updates the bar color if a new cycle is reached."""
        current_cycle = self.current_steps // self._original_total_steps
        if current_cycle > self._last_cycle:
            self._last_cycle = current_cycle
            if current_cycle > 0:
                available_colors = [
                    c
                    for c in colors.PROGRESS_BAR_CYCLE_COLORS
                    if c != self._current_bar_color
                ]
                if not available_colors:
                    available_colors = colors.PROGRESS_BAR_CYCLE_COLORS
                if available_colors:
                    self._current_bar_color = self._rng.choice(available_colors)

    def update_extra_data(self, data: dict[str, Any]):
        """Updates or adds key-value pairs to display."""
        self.extra_data.update(data)

    def reset_time(self):
        """Resets the start time to now, keeping current steps."""
        self.start_time = time.time()
        self._last_step_time = self.start_time
        self._step_times = []
        self.initial_steps = self.current_steps

    def reset_all(self, new_total_steps: int | None = None):
        """Resets steps to 0 and start time to now. Optionally updates total steps."""
        self.current_steps = 0
        self.initial_steps = 0
        if new_total_steps is not None:
            self._original_total_steps = max(1, new_total_steps)
        self.start_time = time.time()
        self._last_step_time = self.start_time
        self._step_times = []
        self.extra_data = {}
        self._last_cycle = -1
        self._current_bar_color = (
            colors.PROGRESS_BAR_CYCLE_COLORS[0]
            if colors.PROGRESS_BAR_CYCLE_COLORS
            else colors.BLUE
        )

    def get_progress_fraction(self) -> float:
        """Returns progress within the current cycle as a fraction (0.0 to 1.0)."""
        if self._original_total_steps <= 1:
            return 1.0
        if self.current_steps == 0:
            return 0.0
        progress_in_cycle = self.current_steps % self._original_total_steps
        if progress_in_cycle == 0 and self.current_steps > 0:
            return 1.0
        else:
            return progress_in_cycle / self._original_total_steps

    def get_elapsed_time(self) -> float:
        """Returns the time elapsed since the start time."""
        return time.time() - self.start_time

    def get_eta_seconds(self) -> float | None:
        """Calculates the estimated time remaining in seconds."""
        if (
            self._original_total_steps <= 1
            or self.current_steps >= self._original_total_steps
        ):
            return None
        steps_processed = self.current_steps - self.initial_steps
        if steps_processed <= 0:
            return None
        elapsed = self.get_elapsed_time()
        if elapsed < 1.0:
            return None
        speed = steps_processed / elapsed
        if speed < 1e-6:
            return None
        remaining_steps = self._original_total_steps - self.current_steps
        if remaining_steps <= 0:
            return 0.0
        eta = remaining_steps / speed
        return eta

    def _get_pulsing_color(self) -> tuple[int, int, int]:
        """Applies a subtle brightness pulse to the current bar color."""
        base_color = self._current_bar_color
        pulse_amplitude = 15
        brightness_offset = int(
            pulse_amplitude * math.sin(time.time() * 4 + self._pulse_phase)
        )
        # --- CHANGED: Explicitly cast to tuple[int, int, int] ---
        pulsed_color = cast(
            "tuple[int, int, int]",
            tuple(max(0, min(255, c + brightness_offset)) for c in base_color),
        )
        # --- END CHANGED ---
        return pulsed_color

    def render(
        self,
        surface: pygame.Surface,
        position: tuple[int, int],
        width: int,
        height: int,
        font: pygame.font.Font,
        bg_color: tuple[int, int, int] = colors.DARK_GRAY,
        text_color: tuple[int, int, int] = colors.WHITE,
        border_width: int = 1,
        border_color: tuple[int, int, int] = colors.GRAY,
        border_radius: int = 3,
        info_line: str | None = None,  # Keep optional info_line
    ):
        """Draws the progress bar onto the given surface."""
        x, y = position
        progress_fraction = self.get_progress_fraction()

        # Background
        bg_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(surface, bg_color, bg_rect, border_radius=border_radius)

        # Pulsing Bar Fill
        fill_width = int(width * progress_fraction)
        if fill_width > 0:
            fill_width = min(width, fill_width)
            fill_rect = pygame.Rect(x, y, fill_width, height)
            pulsing_bar_color = self._get_pulsing_color()
            pygame.draw.rect(
                surface, pulsing_bar_color, fill_rect, border_radius=border_radius
            )

        # Border
        if border_width > 0:
            pygame.draw.rect(
                surface,
                border_color,
                bg_rect,
                border_width,
                border_radius=border_radius,
            )

        # --- Text Rendering (Modified) ---
        line_height = font.get_height()
        if height >= line_height + 4:
            # Always generate default progress text parts
            elapsed_time_str = format_eta(self.get_elapsed_time())
            eta_seconds = self.get_eta_seconds()
            eta_str = format_eta(eta_seconds) if eta_seconds is not None else "--"
            processed_steps = self.current_steps
            expected_steps = self._original_total_steps
            processed_str = (
                f"{processed_steps / 1e6:.1f}M"
                if processed_steps >= 1e6
                else (
                    f"{processed_steps / 1e3:.0f}k"
                    if processed_steps >= 1000
                    else f"{processed_steps:,}"
                )
            )
            expected_str = (
                f"{expected_steps / 1e6:.1f}M"
                if expected_steps >= 1e6
                else (
                    f"{expected_steps / 1e3:.0f}k"
                    if expected_steps >= 1000
                    else f"{expected_steps:,}"
                )
            )
            progress_text = f"{processed_str}/{expected_str}"
            if self._original_total_steps <= 1:
                progress_text = f"{processed_str}"
            extra_text = ""
            if self.extra_data:
                extra_items = [f"{k}:{v}" for k, v in self.extra_data.items()]
                extra_text = " | " + " ".join(extra_items)

            # Construct the display text
            default_progress_info = f"{self.entity_title}: {progress_text} (T:{elapsed_time_str} ETA:{eta_str}){extra_text}"

            # --- CHANGED: Prepend default info if info_line is given ---
            if info_line is not None:
                display_text = (
                    f"{default_progress_info} || {info_line}"  # Combine with separator
                )
            else:
                display_text = default_progress_info  # Use only default
            # --- END CHANGED ---

            # Truncate if necessary
            max_text_width = width - 10
            if font.size(display_text)[0] > max_text_width:
                while (
                    font.size(display_text + "...")[0] > max_text_width
                    and len(display_text) > 20
                ):
                    display_text = display_text[:-1]
                display_text += "..."

            # Render and blit centered vertically
            text_surf = font.render(display_text, True, text_color)
            text_rect = text_surf.get_rect(center=(x + width // 2, y + height // 2))
            surface.blit(text_surf, text_rect)
        # --- End Text Rendering ---
