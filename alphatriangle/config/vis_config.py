from pydantic import BaseModel, Field


class VisConfig(BaseModel):
    """Configuration for visualization (Pydantic model)."""

    FPS: int = Field(default=30, gt=0)
    SCREEN_WIDTH: int = Field(default=1000, gt=0)
    SCREEN_HEIGHT: int = Field(default=800, gt=0)

    # Layout
    GRID_AREA_RATIO: float = Field(default=0.7, gt=0, le=1.0)
    PREVIEW_AREA_WIDTH: int = Field(default=150, gt=0)
    PADDING: int = Field(default=10, ge=0)
    HUD_HEIGHT: int = Field(default=40, ge=0)

    # Fonts (sizes)
    FONT_UI_SIZE: int = Field(default=24, gt=0)
    FONT_SCORE_SIZE: int = Field(default=30, gt=0)
    FONT_HELP_SIZE: int = Field(default=18, gt=0)

    # Preview Area
    PREVIEW_PADDING: int = Field(default=5, ge=0)
    PREVIEW_BORDER_WIDTH: int = Field(default=1, ge=0)
    PREVIEW_SELECTED_BORDER_WIDTH: int = Field(default=3, ge=0)
    PREVIEW_INNER_PADDING: int = Field(default=2, ge=0)


VisConfig.model_rebuild(force=True)
