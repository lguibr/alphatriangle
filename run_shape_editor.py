# File: run_shape_editor.py
import pygame
import sys
import os
import logging
import random
from typing import List, Tuple, Optional, Dict

# Ensure the src directory is in the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Imports from your project structure
from src import config, utils, environment, visualization, structs

# Basic logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ShapeEditor")


class ShapeEditor:
    """Manages the state and rendering for the interactive shape editor."""

    def __init__(self):
        self.vis_config = config.VisConfig()
        self.env_config = config.EnvConfig()

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode(
            (self.vis_config.SCREEN_WIDTH, self.vis_config.SCREEN_HEIGHT),
            pygame.RESIZABLE,
        )
        pygame.display.set_caption(f"{config.APP_NAME} - Shape Editor")
        self.clock = pygame.time.Clock()
        self.fonts = visualization.load_fonts()

        # --- Editor State ---
        self.grid_data = environment.GridData(self.env_config)
        self.current_shape_triangles: List[Tuple[int, int, bool]] = []
        self.current_shape_color: Tuple[int, int, int] = random.choice(
            structs.SHAPE_COLORS
        )
        self.hover_coord: Optional[Tuple[int, int]] = None
        self.layout_rects: Optional[Dict[str, pygame.Rect]] = None
        self._layout_calculated_for_size: Tuple[int, int] = (0, 0)

        # --- Shape Collection ---
        # Stores tuples of (normalized_triangle_list, color_tuple)
        self.saved_shapes: List[
            Tuple[List[Tuple[int, int, bool]], Tuple[int, int, int]]
        ] = []

        self.running = True
        logger.info("Shape Editor Initialized.")
        self.ensure_layout()

    def ensure_layout(self) -> Dict[str, pygame.Rect]:
        """Calculates or retrieves the layout rectangles."""
        current_w, current_h = self.screen.get_size()
        current_size = (current_w, current_h)

        if (
            self.layout_rects is None
            or self._layout_calculated_for_size != current_size
        ):
            # Use interactive layout for grid/preview split
            self.layout_rects = visualization.calculate_interactive_layout(
                current_w, current_h, self.vis_config
            )
            self._layout_calculated_for_size = current_size
            logger.info(
                f"Recalculated layout for size {current_size}: {self.layout_rects}"
            )
        return self.layout_rects if self.layout_rects is not None else {}

    def normalize_shape_triangles(
        self,
    ) -> Optional[List[Tuple[int, int, bool]]]:
        """Normalizes triangle coordinates relative to the top-leftmost triangle."""
        if not self.current_shape_triangles:
            return None

        min_r = min(r for r, c, u in self.current_shape_triangles)
        # Find the minimum column index among triangles in the minimum row
        min_c_in_min_r = min(
            c for r, c, u in self.current_shape_triangles if r == min_r
        )

        normalized = sorted(
            [
                (r - min_r, c - min_c_in_min_r, u)
                for r, c, u in self.current_shape_triangles
            ]
        )
        return normalized

    def save_current_shape(self):
        """Normalizes and adds the current shape to the saved list, then clears."""
        normalized_triangles = self.normalize_shape_triangles()
        if not normalized_triangles:
            logger.warning("Cannot save empty shape.")
            return

        # Store both triangles and color internally
        shape_data = (normalized_triangles, self.current_shape_color)
        self.saved_shapes.append(shape_data)
        logger.info(f"Saved shape #{len(self.saved_shapes)} to internal list.")
        self.clear_shape()  # Clear for the next shape

    def print_all_saved_shapes(self):
        """Prints all collected shapes to the console in the desired list-of-lists format."""
        if not self.saved_shapes:
            logger.warning("No shapes saved yet to print.")
            return

        print("\n" + "=" * 40)
        print("Collected Shape Definitions (Triangle Lists Only):")
        print("=" * 40)
        print("[")  # Start of the outer list

        num_shapes = len(self.saved_shapes)
        for i, (triangles, color) in enumerate(self.saved_shapes):
            print(f"    [ # Shape {i+1}")  # Start of the inner list for this shape
            num_triangles = len(triangles)
            for j, (r, c, u) in enumerate(triangles):
                # Indentation for the tuple elements
                print("        (")
                print(f"            {r},")
                print(f"            {c},")
                print(f"            {u},")
                # Add comma after tuple unless it's the last one in the list
                if j < num_triangles - 1:
                    print("        ),")
                else:
                    print("        )")
            # Add comma after inner list unless it's the last shape in the outer list
            if i < num_shapes - 1:
                print("    ],")
            else:
                print("    ]")
        print("]")  # End of the outer list
        print("=" * 40 + "\n")
        logger.info(
            f"Printed {len(self.saved_shapes)} collected shapes to console (triangle lists only)."
        )

    def clear_shape(self):
        """Clears the current shape."""
        self.current_shape_triangles = []
        # Optionally cycle color when clearing
        # self.cycle_color()
        logger.info("Cleared current shape.")

    def cycle_color(self):
        """Cycles through the available shape colors."""
        current_index = -1
        try:
            current_index = structs.SHAPE_COLORS.index(self.current_shape_color)
        except ValueError:
            pass  # Color not found, start from beginning
        next_index = (current_index + 1) % len(structs.SHAPE_COLORS)
        self.current_shape_color = structs.SHAPE_COLORS[next_index]
        logger.info(f"Changed shape color to: {self.current_shape_color}")

    def handle_events(self):
        """Processes Pygame events."""
        self.hover_coord = None  # Reset hover each frame
        mouse_pos = pygame.mouse.get_pos()
        grid_rect = self.ensure_layout().get("grid")

        if grid_rect:
            hover_coords = visualization.get_grid_coords_from_screen(
                mouse_pos, grid_rect, self.env_config
            )
            if hover_coords:
                r, c = hover_coords
                if (
                    self.grid_data.valid(r, c)
                    and not self.grid_data.triangles[r][c].is_death
                ):
                    self.hover_coord = hover_coords

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_n:  # N for New/Clear
                    self.clear_shape()
                elif event.key == pygame.K_s:  # S for Save current shape to list
                    self.save_current_shape()
                elif event.key == pygame.K_p:  # P for Print all saved shapes
                    self.print_all_saved_shapes()
                elif event.key == pygame.K_c:  # C for Cycle color
                    self.cycle_color()
            if event.type == pygame.VIDEORESIZE:
                try:
                    w, h = max(320, event.w), max(240, event.h)
                    self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
                    self.layout_rects = None  # Force layout recalculation
                    logger.info(f"Window resized to {w}x{h}")
                except pygame.error as e:
                    logger.error(f"Error resizing window: {e}")
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if grid_rect and grid_rect.collidepoint(mouse_pos):
                    click_coords = visualization.get_grid_coords_from_screen(
                        mouse_pos, grid_rect, self.env_config
                    )
                    if click_coords:
                        r, c = click_coords
                        if (
                            self.grid_data.valid(r, c)
                            and not self.grid_data.triangles[r][c].is_death
                        ):
                            is_up = (r + c) % 2 != 0
                            triangle_tuple = (r, c, is_up)
                            if triangle_tuple in self.current_shape_triangles:
                                self.current_shape_triangles.remove(triangle_tuple)
                                logger.debug(f"Removed triangle: {triangle_tuple}")
                            else:
                                self.current_shape_triangles.append(triangle_tuple)
                                logger.debug(f"Added triangle: {triangle_tuple}")
                                # Sort to maintain consistent order (optional)
                                self.current_shape_triangles.sort()

    def render_current_shape_on_grid(self, grid_surf: pygame.Surface):
        """Draws the currently edited triangles onto the grid surface."""
        if not self.current_shape_triangles:
            return

        cw, ch, ox, oy = visualization.core.coord_mapper._calculate_render_params(
            grid_surf.get_width(), grid_surf.get_height(), self.env_config
        )
        if cw <= 0 or ch <= 0:
            return

        for r, c, is_up in self.current_shape_triangles:
            temp_tri = structs.Triangle(r, c, is_up)
            pts = temp_tri.get_points(ox, oy, cw, ch)
            pygame.draw.polygon(grid_surf, self.current_shape_color, pts)
            pygame.draw.polygon(grid_surf, visualization.colors.WHITE, pts, 1)  # Border

    def render_saved_shapes_list(self, preview_surf: pygame.Surface):
        """Renders the list of saved shapes in the preview area."""
        preview_surf.fill(visualization.colors.PREVIEW_BG)
        num_saved = len(self.saved_shapes)
        if num_saved == 0:
            font = self.fonts.get("help")
            if font:
                text_surf = font.render(
                    "Press 'S' to save a shape", True, visualization.colors.LIGHT_GRAY
                )
                text_rect = text_surf.get_rect(
                    center=(
                        preview_surf.get_width() // 2,
                        preview_surf.get_height() // 2,
                    )
                )
                preview_surf.blit(text_surf, text_rect)
            return

        # --- Calculate layout for saved shapes ---
        pad = self.vis_config.PREVIEW_PADDING
        inner_pad = self.vis_config.PREVIEW_INNER_PADDING
        border = self.vis_config.PREVIEW_BORDER_WIDTH
        title_font = self.fonts.get("help")
        title_h = title_font.get_height() + 5 if title_font else 0

        # Simple vertical layout
        available_h = preview_surf.get_height() - 2 * pad
        slot_h = max(
            30, (available_h / num_saved) if num_saved > 0 else available_h
        )  # Min height 30
        slot_w = preview_surf.get_width() - 2 * pad
        current_y = pad

        for i, (triangles, color) in enumerate(self.saved_shapes):
            if current_y + slot_h > preview_surf.get_height() - pad:
                logger.debug("Preview area full, stopping render of saved shapes.")
                break  # Stop rendering if we run out of space

            slot_rect_local = pygame.Rect(pad, current_y, slot_w, slot_h)
            pygame.draw.rect(
                preview_surf,
                visualization.colors.PREVIEW_BORDER,
                slot_rect_local,
                border,
            )

            # Draw shape number title
            if title_font:
                title_surf = title_font.render(
                    f"Shape #{i+1}", True, visualization.colors.LIGHT_GRAY
                )
                title_rect = title_surf.get_rect(
                    left=slot_rect_local.left + inner_pad,
                    top=slot_rect_local.top + inner_pad,
                )
                preview_surf.blit(title_surf, title_rect)

            # Calculate drawing area for the shape itself
            shape_area_top = slot_rect_local.top + title_h
            draw_area_w = slot_w - 2 * (border + inner_pad)
            draw_area_h = max(0, slot_h - title_h - 2 * (border + inner_pad))

            if draw_area_w > 0 and draw_area_h > 0:
                temp_shape = structs.Shape(triangles, color)
                min_r, min_c, max_r, max_c = temp_shape.bbox()
                shape_rows = max_r - min_r + 1
                shape_cols_eff = (
                    (max_c - min_c + 1) * 0.75 + 0.25 if temp_shape.triangles else 1
                )

                scale_w = (
                    draw_area_w / shape_cols_eff if shape_cols_eff > 0 else draw_area_w
                )
                scale_h = draw_area_h / shape_rows if shape_rows > 0 else draw_area_h
                cell_size = max(1.0, min(scale_w, scale_h))

                shape_render_w = shape_cols_eff * cell_size
                shape_render_h = shape_rows * cell_size
                draw_topleft_x = (
                    slot_rect_local.left
                    + border
                    + inner_pad
                    + (draw_area_w - shape_render_w) / 2
                )
                draw_topleft_y = (
                    shape_area_top
                    + border
                    + inner_pad
                    + (draw_area_h - shape_render_h) / 2
                )

                visualization.drawing.shapes.draw_shape(
                    preview_surf,
                    temp_shape,
                    (draw_topleft_x, draw_topleft_y),
                    cell_size,
                    is_selected=False,
                    origin_offset=(-min_r, -min_c),
                )

            current_y += slot_h  # Move to next slot position (no vertical padding here)

    def render_hud(self):
        """Renders help text and saved shape count at the bottom."""
        screen_w, screen_h = self.screen.get_size()
        help_font = self.fonts.get("help")
        if not help_font:
            return

        # Display saved shape count
        count_text = f"Saved: {len(self.saved_shapes)}"
        count_surf = help_font.render(count_text, True, visualization.colors.YELLOW)
        count_rect = count_surf.get_rect(left=15, bottom=screen_h - 10)
        self.screen.blit(count_surf, count_rect)

        # Display instructions
        instructions = [
            "[Click Grid] Add/Remove Triangle",
            "[N] New Shape",
            "[S] Save Shape to List",
            "[P] Print All Saved",
            "[C] Cycle Color",
            "[ESC] Quit",
        ]
        text = " | ".join(instructions)
        help_surf = help_font.render(text, True, visualization.colors.LIGHT_GRAY)
        # Position instructions to the right of the count text
        help_rect = help_surf.get_rect(left=count_rect.right + 20, bottom=screen_h - 10)
        # If it overflows, maybe center it instead or wrap text (simpler: just let it clip)
        if help_rect.right > screen_w - 15:
            help_rect.centerx = screen_w // 2
            help_rect.bottom = screen_h - 10

        self.screen.blit(help_surf, help_rect)

    def render(self):
        """Renders the entire editor screen."""
        self.screen.fill(visualization.colors.DARK_GRAY)
        layout_rects = self.ensure_layout()
        grid_rect = layout_rects.get("grid")
        preview_rect = layout_rects.get("preview")

        # Render Grid Area
        if grid_rect and grid_rect.width > 0 and grid_rect.height > 0:
            try:
                grid_surf = self.screen.subsurface(grid_rect)
                # Draw background and base grid
                visualization.drawing.grid.draw_grid_background(
                    grid_surf, visualization.colors.GRID_BG_DEFAULT
                )
                visualization.drawing.grid.draw_grid_triangles(
                    grid_surf, self.grid_data, self.env_config
                )
                # Draw the shape being edited
                self.render_current_shape_on_grid(grid_surf)
                # Draw hover highlight
                if self.hover_coord:
                    visualization.drawing.highlight.draw_debug_highlight(
                        grid_surf,
                        self.hover_coord[0],
                        self.hover_coord[1],
                        self.env_config,
                    )
            except ValueError as e:
                logger.error(f"Error creating grid subsurface ({grid_rect}): {e}")
                pygame.draw.rect(self.screen, visualization.colors.RED, grid_rect, 1)

        # Render Preview Area (now shows saved shapes)
        if preview_rect and preview_rect.width > 0 and preview_rect.height > 0:
            try:
                preview_surf = self.screen.subsurface(preview_rect)
                self.render_saved_shapes_list(
                    preview_surf
                )  # Call the updated render function
            except ValueError as e:
                logger.error(f"Error creating preview subsurface ({preview_rect}): {e}")
                pygame.draw.rect(self.screen, visualization.colors.RED, preview_rect, 1)

        # Render HUD
        self.render_hud()

        pygame.display.flip()

    def run(self):
        """Main application loop."""
        while self.running:
            dt = self.clock.tick(self.vis_config.FPS) / 1000.0
            self.handle_events()
            self.render()

        logger.info("Shape Editor loop finished.")
        # Automatically print any collected shapes on exit if not already printed
        if self.saved_shapes:
            logger.info("Printing collected shapes on exit...")
            self.print_all_saved_shapes()

        pygame.quit()


if __name__ == "__main__":
    editor = ShapeEditor()
    editor.run()
    logger.info("Exiting Shape Editor.")
    sys.exit()
