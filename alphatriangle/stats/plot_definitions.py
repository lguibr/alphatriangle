# File: alphatriangle/stats/plot_definitions.py
from typing import Literal, NamedTuple

# Define type for x-axis data source
PlotXAxisType = Literal["index", "global_step", "buffer_size"]

# Define metric key constant for weight updates
WEIGHT_UPDATE_METRIC_KEY = "Internal/Weight_Update_Step"


class PlotDefinition(NamedTuple):
    """Configuration for a single subplot."""

    metric_key: str  # Key in the StatsCollectorData dictionary
    label: str  # Title displayed on the plot
    y_log_scale: bool  # Use logarithmic scale for y-axis
    x_axis_type: PlotXAxisType  # What the x-axis represents


class PlotDefinitions:
    """Holds the definitions for all plots in the dashboard."""

    def __init__(self, colors: dict[str, tuple[float, float, float]]):
        self.colors = colors  # Store colors if needed for default lookups
        self.nrows: int = 4
        self.ncols: int = 3
        # Key used to get weight update steps for vertical lines
        self.weight_update_key = WEIGHT_UPDATE_METRIC_KEY  # Use the constant

        # Define the layout and properties of each plot
        self._definitions: list[PlotDefinition] = [
            # Row 1
            # --- CHANGED: x_axis_type to "index" ---
            PlotDefinition("RL/Current_Score", "Score", False, "index"),
            PlotDefinition(
                "Rate/Episodes_Per_Sec", "Episodes/sec", False, "buffer_size"
            ),
            PlotDefinition("Loss/Total", "Total Loss", True, "global_step"),
            # Row 2
            PlotDefinition("RL/Step_Reward", "Step Reward", False, "index"),
            PlotDefinition(
                "Rate/Simulations_Per_Sec", "Sims/sec", False, "buffer_size"
            ),
            PlotDefinition("Loss/Policy", "Policy Loss", True, "global_step"),
            # Row 3
            PlotDefinition("MCTS/Step_Visits", "MCTS Visits", False, "index"),
            PlotDefinition("Buffer/Size", "Buffer Size", False, "buffer_size"),
            PlotDefinition("Loss/Value", "Value Loss", True, "global_step"),
            # Row 4
            PlotDefinition("MCTS/Step_Depth", "MCTS Depth", False, "index"),
            # --- END CHANGED ---
            PlotDefinition("Rate/Steps_Per_Sec", "Steps/sec", False, "global_step"),
            PlotDefinition("LearningRate", "Learn Rate", True, "global_step"),
        ]

        # Validate grid size
        if len(self._definitions) > self.nrows * self.ncols:
            raise ValueError(
                f"Number of plot definitions ({len(self._definitions)}) exceeds grid size ({self.nrows}x{self.ncols})"
            )

    def get_definitions(self) -> list[PlotDefinition]:
        """Returns the list of plot definitions."""
        return self._definitions


# Define PlotType for potential external use, though PlotDefinition is more specific
PlotType = PlotDefinition
