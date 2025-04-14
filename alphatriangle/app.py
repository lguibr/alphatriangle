import logging

import pygame

from . import (
    config,
    environment,
    interaction,
    visualization,
)

logger = logging.getLogger(__name__)


class Application:
    """Main application integrating visualization and interaction."""

    def __init__(self, mode: str = "play"):
        self.vis_config = config.VisConfig()
        self.env_config = config.EnvConfig()
        self.mode = mode

        pygame.init()
        pygame.font.init()
        self.screen = self._setup_screen()
        self.clock = pygame.time.Clock()
        self.fonts = visualization.load_fonts()

        if self.mode in ["play", "debug"]:
            # Create GameState first
            self.game_state = environment.GameState(self.env_config)
            # Create Visualizer
            self.visualizer = visualization.Visualizer(
                self.screen, self.vis_config, self.env_config, self.fonts
            )
            # Create InputHandler, passing GameState and Visualizer
            self.input_handler = interaction.InputHandler(
                self.game_state, self.visualizer, self.mode, self.env_config
            )
        else:
            # Handle other modes or raise error if necessary
            logger.error(f"Unsupported application mode: {self.mode}")
            raise ValueError(f"Unsupported application mode: {self.mode}")

        self.running = True

    def _setup_screen(self) -> pygame.Surface:
        """Initializes the Pygame screen."""
        screen = pygame.display.set_mode(
            (self.vis_config.SCREEN_WIDTH, self.vis_config.SCREEN_HEIGHT),
            pygame.RESIZABLE,
        )
        pygame.display.set_caption(f"{config.APP_NAME} - {self.mode.capitalize()} Mode")
        return screen

    def run(self):
        """Main application loop."""
        logger.info(f"Starting application in {self.mode} mode.")
        while self.running:
            # dt = ( # Unused variable
            #     self.clock.tick(self.vis_config.FPS) / 1000.0
            # )  # Delta time (unused currently)
            self.clock.tick(self.vis_config.FPS)  # Still tick the clock

            # Handle Input using InputHandler
            if self.input_handler:
                self.running = self.input_handler.handle_input()
                if not self.running:
                    break  # Exit loop if handle_input returns False
            else:
                # Fallback event handling if input_handler is not initialized (should not happen in play/debug)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        self.running = False
                    # Basic resize handling needed even without input handler
                    # Combine nested if statements
                    if event.type == pygame.VIDEORESIZE and self.visualizer:
                        try:
                            w, h = max(320, event.w), max(240, event.h)
                            # Update visualizer's screen reference
                            self.visualizer.screen = pygame.display.set_mode(
                                (w, h), pygame.RESIZABLE
                            )
                            # Invalidate visualizer's layout cache
                            self.visualizer.layout_rects = None
                        except pygame.error as e:
                            logger.error(f"Error resizing window: {e}")
                if not self.running:
                    break

            # Render using Visualizer
            if (
                self.mode in ["play", "debug"]
                and self.visualizer
                and self.game_state
                and self.input_handler
            ):
                # Get interaction state needed for rendering from InputHandler
                interaction_render_state = (
                    self.input_handler.get_render_interaction_state()
                )
                # Pass game state, mode, and interaction state to visualizer
                self.visualizer.render(
                    self.game_state,
                    self.mode,
                    **interaction_render_state,  # Unpack the dict as keyword arguments
                )
                pygame.display.flip()  # Update the full display

        logger.info("Application loop finished.")
        pygame.quit()
