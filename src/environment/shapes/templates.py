# File: src/environment/shapes/templates.py
# ==============================================================================
# ==                    PREDEFINED SHAPE TEMPLATES                          ==
# ==                                                                        ==
# ==    DO NOT MODIFY THIS LIST MANUALLY unless you are absolutely sure!    ==
# == These shapes are fundamental to the game's design and balance.         ==
# == Modifying them can have unintended consequences on gameplay and agent  ==
# == training.                                                              ==
# ==============================================================================

# List of predefined shape templates. Each template is a list of relative triangle coordinates (dr, dc, is_up).
# Coordinates are relative to the shape's origin (typically the top-leftmost triangle).
# is_up = True for upward-pointing triangle, False for downward-pointing.
PREDEFINED_SHAPE_TEMPLATES: list[list[tuple[int, int, bool]]] = [
    [  # Shape 1
        (
            0,
            0,
            True,
        )
    ],
    [  # Shape 1
        (
            0,
            0,
            True,
        )
    ],
    [  # Shape 2
        (
            0,
            0,
            True,
        ),
        (
            1,
            0,
            False,
        ),
    ],
    [  # Shape 2
        (
            0,
            0,
            True,
        ),
        (
            1,
            0,
            False,
        ),
    ],
    [  # Shape 3
        (
            0,
            0,
            False,
        )
    ],
    [  # Shape 4
        (
            0,
            0,
            True,
        ),
        (
            0,
            1,
            False,
        ),
    ],
    [  # Shape 4
        (
            0,
            0,
            True,
        ),
        (
            0,
            1,
            False,
        ),
    ],
    [  # Shape 5
        (
            0,
            0,
            False,
        ),
        (
            0,
            1,
            True,
        ),
    ],
    [  # Shape 5
        (
            0,
            0,
            False,
        ),
        (
            0,
            1,
            True,
        ),
    ],
    [  # Shape 6
        (
            0,
            0,
            True,
        ),
        (
            0,
            1,
            False,
        ),
        (
            0,
            2,
            True,
        ),
    ],
    [  # Shape 7
        (
            0,
            0,
            False,
        ),
        (
            0,
            1,
            True,
        ),
        (
            0,
            2,
            False,
        ),
    ],
    [  # Shape 8
        (
            0,
            0,
            True,
        ),
        (
            0,
            1,
            False,
        ),
        (
            0,
            2,
            True,
        ),
        (
            1,
            0,
            False,
        ),
    ],
    [  # Shape 9
        (
            0,
            0,
            True,
        ),
        (
            0,
            1,
            False,
        ),
        (
            0,
            2,
            True,
        ),
        (
            1,
            2,
            False,
        ),
    ],
    [  # Shape 10
        (
            0,
            0,
            False,
        ),
        (
            0,
            1,
            True,
        ),
        (
            1,
            0,
            True,
        ),
        (
            1,
            1,
            False,
        ),
    ],
    [  # Shape 11
        (
            0,
            0,
            True,
        ),
        (
            0,
            2,
            True,
        ),
        (
            1,
            0,
            False,
        ),
        (
            1,
            1,
            True,
        ),
        (
            1,
            2,
            False,
        ),
    ],
    [  # Shape 12
        (
            0,
            0,
            True,
        ),
        (
            1,
            -2,
            False,
        ),
        (
            1,
            -1,
            True,
        ),
        (
            1,
            0,
            False,
        ),
    ],
    [  # Shape 13
        (
            0,
            0,
            True,
        ),
        (
            0,
            1,
            False,
        ),
        (
            1,
            0,
            False,
        ),
        (
            1,
            1,
            True,
        ),
    ],
    [  # Shape 14
        (
            0,
            0,
            True,
        ),
        (
            0,
            1,
            False,
        ),
        (
            1,
            0,
            False,
        ),
        (
            1,
            1,
            True,
        ),
        (
            1,
            2,
            False,
        ),
    ],
    [  # Shape 15
        (
            0,
            0,
            True,
        ),
        (
            0,
            1,
            False,
        ),
        (
            0,
            2,
            True,
        ),
        (
            1,
            0,
            False,
        ),
        (
            1,
            1,
            True,
        ),
    ],
    [  # Shape 16
        (
            0,
            0,
            True,
        ),
        (
            0,
            1,
            False,
        ),
        (
            0,
            2,
            True,
        ),
        (
            1,
            0,
            False,
        ),
        (
            1,
            2,
            False,
        ),
    ],
    [  # Shape 17
        (
            0,
            0,
            True,
        ),
        (
            0,
            1,
            False,
        ),
        (
            0,
            2,
            True,
        ),
        (
            1,
            1,
            True,
        ),
        (
            1,
            2,
            False,
        ),
    ],
    [  # Shape 18
        (
            0,
            0,
            True,
        ),
        (
            0,
            2,
            True,
        ),
        (
            1,
            0,
            False,
        ),
        (
            1,
            1,
            True,
        ),
        (
            1,
            2,
            False,
        ),
    ],
    [  # Shape 19
        (
            0,
            0,
            True,
        ),
        (
            0,
            1,
            False,
        ),
        (
            1,
            0,
            False,
        ),
        (
            1,
            1,
            True,
        ),
        (
            1,
            2,
            False,
        ),
    ],
    [  # Shape 20
        (
            0,
            0,
            False,
        ),
        (
            0,
            1,
            True,
        ),
        (
            1,
            1,
            False,
        ),
    ],
    [  # Shape 21
        (
            0,
            0,
            True,
        ),
        (
            1,
            -1,
            True,
        ),
        (
            1,
            0,
            False,
        ),
    ],
    [  # Shape 22
        (
            0,
            0,
            True,
        ),
        (
            1,
            0,
            False,
        ),
        (
            1,
            1,
            True,
        ),
    ],
    [  # Shape 23
        (
            0,
            0,
            True,
        ),
        (
            1,
            -1,
            True,
        ),
        (
            1,
            0,
            False,
        ),
        (
            1,
            1,
            True,
        ),
    ],
    [  # Shape 24
        (
            0,
            0,
            True,
        ),
        (
            1,
            -1,
            True,
        ),
        (
            1,
            0,
            False,
        ),
    ],
    [  # Shape 25
        (
            0,
            0,
            False,
        ),
        (
            0,
            1,
            True,
        ),
        (
            0,
            2,
            False,
        ),
        (
            1,
            1,
            False,
        ),
    ],
    [  # Shape 26
        (
            0,
            0,
            False,
        ),
        (
            0,
            1,
            True,
        ),
        (
            1,
            1,
            False,
        ),
    ],
    [  # Shape 27
        (
            0,
            0,
            True,
        ),
        (
            0,
            1,
            False,
        ),
        (
            1,
            0,
            False,
        ),
    ],
]
