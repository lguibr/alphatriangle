# File: alphatriangle/visualization/ui/progress_bar.py
import random  # Added import
import time
from typing import Any

import pygame

from ...utils import format_eta
from ..core import colors


class ProgressBar:
    """
    A reusable progress bar component for visualization.
    Handles overflow by cycling colors and displaying actual progress count.
    """

    def __init__(
        self,
        entity_title: str,
        total_steps: int,
        start_time: float | None = None,
        initial_steps: int = 0,
        initial_color: tuple[int, int, int] = colors.GREEN,  # Added initial color
    ):
        self.entity_title = entity_title
        # --- CHANGED: Store original total, ensure >= 1 ---
        self._original_total_steps = max(
            1, total_steps if total_steps is not None else 1
        )
        # --- END CHANGED ---
        self.initial_steps = max(0, initial_steps)
        self.current_steps = self.initial_steps
        self.start_time = start_time if start_time is not None else time.time()
        self._last_step_time = self.start_time
        self._step_times: list[float] = []
        self.extra_data: dict[str, Any] = {}
        # --- ADDED: Color cycling attributes ---
        self._current_bar_color = initial_color
        self._last_cycle = -1  # Track the last completed cycle
        self._rng = random.Random()  # Local RNG for color choice
        # --- END ADDED ---

    def add_steps(self, steps_added: int):
        """Adds steps to the progress bar's current count."""
        if steps_added <= 0:
            return
        # --- REMOVED: Clamping based on total_steps ---
        # if self.total_steps > 1:
        #     self.current_steps = min(self.total_steps, self.current_steps + steps_added)
        # else:
        #     self.current_steps += steps_added
        # --- END REMOVED ---
        self.current_steps += steps_added  # Simply increment
        self._check_color_cycle()  # Check if color needs update

    def set_current_steps(self, steps: int):
        """Directly sets the current step count."""
        # --- REMOVED: Clamping based on total_steps ---
        # if self.total_steps > 1:
        #     self.current_steps = max(0, min(self.total_steps, steps))
        # else:
        #     self.current_steps = max(0, steps)
        # --- END REMOVED ---
        self.current_steps = max(0, steps)  # Set directly, ensure non-negative
        self._check_color_cycle()  # Check if color needs update

    # --- ADDED: Method to update color ---
    def _check_color_cycle(self):
        """Updates the bar color if a new cycle is reached."""
        current_cycle = self.current_steps // self._original_total_steps
        if current_cycle > self._last_cycle:
            self._last_cycle = current_cycle
            if current_cycle > 0:  # Don't change color on the very first cycle (0)
                available_colors = [
                    c
                    for c in colors.PROGRESS_BAR_CYCLE_COLORS
                    if c != self._current_bar_color
                ]
                if not available_colors:  # Fallback if only one color defined
                    available_colors = colors.PROGRESS_BAR_CYCLE_COLORS
                self._current_bar_color = self._rng.choice(available_colors)

    # --- END ADDED ---

    def update_extra_data(self, data: dict[str, Any]):
        """Updates or adds key-value pairs to display."""
        self.extra_data.update(data)

    def reset_time(self):
        """Resets the start time to now, keeping current steps."""
        self.start_time = time.time()
        self._last_step_time = self.start_time
        self._step_times = []
        self.initial_steps = self.current_steps  # Reset initial steps for ETA calc

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
        self._last_cycle = -1  # Reset cycle tracking
        # Optionally reset color? Or keep the last one? Let's reset to default.
        self._current_bar_color = (
            colors.PROGRESS_BAR_CYCLE_COLORS[0]
            if colors.PROGRESS_BAR_CYCLE_COLORS
            else colors.GREEN
        )

    # --- MODIFIED: Handle exact match for 100% display ---
    def get_progress_fraction(self) -> float:
        """Returns progress within the current cycle as a fraction (0.0 to 1.0)."""
        if self._original_total_steps <= 1:
            return 1.0  # Always full if total is 1 or less

        if self.current_steps == 0:
            return 0.0

        # Calculate progress within the current cycle
        progress_in_cycle = self.current_steps % self._original_total_steps

        if progress_in_cycle == 0:
            # If modulo is 0 and current_steps > 0, it means we hit the target exactly.
            # Show 100% in this case.
            return 1.0
        else:
            # Otherwise, show the fraction within the current cycle.
            return progress_in_cycle / self._original_total_steps

    # --- END MODIFIED ---

    def get_elapsed_time(self) -> float:
        """Returns the time elapsed since the start time."""
        return time.time() - self.start_time

    def get_eta_seconds(self) -> float | None:
        """
        Calculates the estimated time remaining in seconds.
        Returns None if ETA cannot be determined or if progress exceeds total.
        """
        # --- CHANGED: Disable ETA after first cycle ---
        if (
            self._original_total_steps <= 1
            or self.current_steps >= self._original_total_steps
        ):
            return None  # ETA not meaningful after overflow or for total=1
        # --- END CHANGED ---

        steps_processed = self.current_steps - self.initial_steps
        if steps_processed <= 0:
            return None

        elapsed = self.get_elapsed_time()
        if elapsed < 1.0:  # Avoid division by zero or unstable early estimates
            return None

        speed = steps_processed / elapsed
        if speed < 1e-6:
            return None

        remaining_steps = self._original_total_steps - self.current_steps
        if remaining_steps <= 0:  # Should be caught by the check above, but safe
            return 0.0

        eta = remaining_steps / speed
        return eta

    def render(
        self,
        surface: pygame.Surface,
        position: tuple[int, int],
        width: int,
        height: int,
        font: pygame.font.Font,
        # --- REMOVED: bar_color parameter ---
        # bar_color: tuple[int, int, int] = colors.BLUE,
        # --- END REMOVED ---
        bg_color: tuple[int, int, int] = colors.DARK_GRAY,
        text_color: tuple[int, int, int] = colors.WHITE,
        border_width: int = 1,
        border_color: tuple[int, int, int] = colors.GRAY,
    ):
        """Draws the progress bar onto the given surface."""
        x, y = position
        # --- CHANGED: Use progress fraction for fill ---
        progress_fraction = self.get_progress_fraction()
        # --- END CHANGED ---
        elapsed_time_str = format_eta(self.get_elapsed_time())
        eta_seconds = self.get_eta_seconds()
        eta_str = format_eta(eta_seconds) if eta_seconds is not None else "N/A"

        bg_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(surface, bg_color, bg_rect)

        # --- CHANGED: Use progress_fraction and _current_bar_color ---
        fill_width = int(width * progress_fraction)
        if fill_width > 0:
            # Clip fill_width to not exceed total width visually
            fill_width = min(width, fill_width)
            fill_rect = pygame.Rect(x, y, fill_width, height)
            pygame.draw.rect(
                surface, self._current_bar_color, fill_rect
            )  # Use cycled color
        # --- END CHANGED ---

        if border_width > 0:
            pygame.draw.rect(surface, border_color, bg_rect, border_width)

        text_y_offset = 2
        available_text_height = height - 2 * text_y_offset
        line_height = font.get_height()
        num_lines = 0
        if available_text_height >= line_height:
            num_lines += 1
        if available_text_height >= line_height * 2:
            num_lines += 1
        if available_text_height >= line_height * 3:
            num_lines += 1
        if self.extra_data and available_text_height >= line_height * 4:
            num_lines += 1

        total_text_height = num_lines * line_height + max(0, num_lines - 1) * 2
        current_y = (
            y + text_y_offset + max(0, (available_text_height - total_text_height) // 2)
        )
        center_x = x + width // 2

        if num_lines >= 1:
            title_surf = font.render(self.entity_title, True, text_color)
            title_rect = title_surf.get_rect(centerx=center_x, top=current_y)
            surface.blit(title_surf, title_rect)
            current_y += line_height + 2

        if num_lines >= 2:
            # --- CHANGED: Display current / original_total ---
            processed_steps = self.current_steps
            expected_steps = self._original_total_steps
            progress_text = f"{processed_steps:,} / {expected_steps:,}"
            # --- END CHANGED ---
            progress_surf = font.render(progress_text, True, text_color)
            progress_rect = progress_surf.get_rect(centerx=center_x, top=current_y)
            surface.blit(progress_surf, progress_rect)
            current_y += line_height + 2

        if num_lines >= 3:
            time_text = f"Elapsed: {elapsed_time_str} | ETA: {eta_str}"
            time_surf = font.render(time_text, True, text_color)
            time_rect = time_surf.get_rect(centerx=center_x, top=current_y)
            surface.blit(time_surf, time_rect)
            current_y += line_height + 2

        if num_lines >= 4 and self.extra_data:
            extra_texts = [f"{k}: {v}" for k, v in self.extra_data.items()]
            extra_text = " | ".join(extra_texts)
            extra_surf = font.render(extra_text, True, text_color)
            extra_rect = extra_surf.get_rect(centerx=center_x, top=current_y)
            surface.blit(extra_surf, extra_rect)
