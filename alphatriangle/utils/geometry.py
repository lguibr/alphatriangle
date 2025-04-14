def is_point_in_polygon(
    point: tuple[float, float], polygon: list[tuple[float, float]]
) -> bool:
    """
    Checks if a point is inside a polygon using the ray casting algorithm.

    Args:
        point: Tuple (x, y) representing the point coordinates.
        polygon: List of tuples [(x1, y1), (x2, y2), ...] representing polygon vertices in order.

    Returns:
        True if the point is inside the polygon, False otherwise.
    """
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        # Combine nested if statements
        if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
            # Use ternary operator for xinters calculation
            xinters = ((y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x) if p1y != p2y else x

            # Check if point is on the segment boundary or crosses the ray
            if abs(p1x - p2x) < 1e-9:  # Vertical line segment
                if abs(x - p1x) < 1e-9:
                    return True  # Point is on the vertical segment
            elif abs(x - xinters) < 1e-9:  # Point is exactly on the intersection
                return True  # Point is on the boundary
            elif (
                p1x == p2x or x <= xinters
            ):  # Point is to the left or on a non-horizontal segment
                inside = not inside

        p1x, p1y = p2x, p2y

    # Check if the point is exactly one of the vertices
    for px, py in polygon:
        if abs(x - px) < 1e-9 and abs(y - py) < 1e-9:
            return True

    return inside
