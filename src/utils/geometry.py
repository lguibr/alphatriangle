from typing import List, Tuple


def is_point_in_polygon(
    point: Tuple[float, float], polygon: List[Tuple[float, float]]
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
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if abs(p1x - p2x) < 1e-9:
                        if abs(x - p1x) < 1e-9:
                            return True
                    elif abs(x - xinters) < 1e-9:
                        return True
                    elif p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    for px, py in polygon:
        if abs(x - px) < 1e-9 and abs(y - py) < 1e-9:
            return True

    return inside
