# File: src/visualization/ui/progress_bar.py
import time
from typing import Any

import pygame

from src.utils import format_eta

from ..core import colors


class ProgressBar:
    """A reusable progress bar component for visualization."""

    def __init__(
        self,
        entity_title: str,
        total_steps: int,
        start_time: float | None = None,
        initial_steps: int = 0,
    ):
        self.entity_title = entity_title
        self.total_steps = max(1, total_steps if total_steps is not None else 1)
        self.initial_steps = max(0, initial_steps)
        self.current_steps = self.initial_steps
        self.start_time = start_time if start_time is not None else time.time()
        self._last_step_time = self.start_time
        self._step_times: list[float] = []  # Add type hint
        self.extra_data: dict[str, Any] = {}

    def add_steps(self, steps_added: int):
        """Adds steps to the progress bar's current count."""
        if steps_added <= 0:
            return
        if self.total_steps > 1:
            self.current_steps = min(self.total_steps, self.current_steps + steps_added)
        else:
            self.current_steps += steps_added

    def set_current_steps(self, steps: int):
        """Directly sets the current step count."""
        if self.total_steps > 1:
            self.current_steps = max(0, min(self.total_steps, steps))
        else:
            self.current_steps = max(0, steps)

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
            self.total_steps = max(1, new_total_steps)
        self.start_time = time.time()
        self._last_step_time = self.start_time
        self._step_times = []
        self.extra_data = {}

    def get_progress(self) -> float:
        """Returns progress as a fraction (0.0 to 1.0). Returns 1.0 if total_steps is 1."""
        if self.total_steps <= 1:
            return 0.0
        return min(1.0, self.current_steps / self.total_steps)

    def get_elapsed_time(self) -> float:
        """Returns the time elapsed since the start time."""
        return time.time() - self.start_time

    def get_eta_seconds(self) -> float | None:
        """Calculates the estimated time remaining in seconds."""
        if self.total_steps <= 1:
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

        remaining_steps = self.total_steps - self.current_steps
        if remaining_steps <= 0:
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
        bar_color: tuple[int, int, int] = colors.BLUE,
        bg_color: tuple[int, int, int] = colors.DARK_GRAY,
        text_color: tuple[int, int, int] = colors.WHITE,
        border_width: int = 1,
        border_color: tuple[int, int, int] = colors.GRAY,
    ):
        """Draws the progress bar onto the given surface."""
        x, y = position
        progress = self.get_progress()
        elapsed_time_str = format_eta(self.get_elapsed_time())
        eta_seconds = self.get_eta_seconds()
        eta_str = format_eta(eta_seconds) if eta_seconds is not None else "N/A"

        bg_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(surface, bg_color, bg_rect)

        fill_width = 0
        if self.total_steps > 1:
            fill_width = int(width * progress)

        if fill_width > 0:
            fill_rect = pygame.Rect(x, y, fill_width, height)
            pygame.draw.rect(surface, bar_color, fill_rect)

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
        if total_text_height < available_text_height:
            current_y = (
                y + text_y_offset + (available_text_height - total_text_height) // 2
            )
        else:
            current_y = y + text_y_offset

        center_x = x + width // 2

        if num_lines >= 1:
            title_surf = font.render(self.entity_title, True, text_color)
            title_rect = title_surf.get_rect(centerx=center_x, top=current_y)
            surface.blit(title_surf, title_rect)
            current_y += line_height + 2

        if num_lines >= 2:
            processed_steps = self.current_steps
            expected_steps = self.total_steps

            if expected_steps <= 1:
                progress_text = f"{processed_steps} steps"
            else:
                progress_text = f"{processed_steps} / {expected_steps} ({progress:.1%})"

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
