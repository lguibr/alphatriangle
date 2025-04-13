from pydantic import BaseModel, Field


class VisConfig(BaseModel):
    """Configuration for visualization (Pydantic model)."""

    FPS: int = Field(30, gt=0)
    SCREEN_WIDTH: int = Field(1000, gt=0)
    SCREEN_HEIGHT: int = Field(800, gt=0)

    # Layout
    GRID_AREA_RATIO: float = Field(0.7, gt=0, le=1.0)
    PREVIEW_AREA_WIDTH: int = Field(150, gt=0)
    PADDING: int = Field(10, ge=0)
    HUD_HEIGHT: int = Field(40, ge=0)

    # Fonts (sizes)
    FONT_UI_SIZE: int = Field(24, gt=0)
    FONT_SCORE_SIZE: int = Field(30, gt=0)
    FONT_HELP_SIZE: int = Field(18, gt=0)

    # Preview Area
    PREVIEW_PADDING: int = Field(5, ge=0)
    PREVIEW_BORDER_WIDTH: int = Field(1, ge=0)
    PREVIEW_SELECTED_BORDER_WIDTH: int = Field(3, ge=0)
    PREVIEW_INNER_PADDING: int = Field(2, ge=0)
