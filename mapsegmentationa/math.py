import numpy as np

from sc2.position import Point2


def direction(line_start: np.ndarray, line_end: np.ndarray) -> np.ndarray:
    """Calculate the direction vector of the line."""
    direction = line_end - line_start
    return direction / np.linalg.norm(direction)


def normal(direction: np.ndarray) -> np.ndarray:
    """Calculate the normal vector of the line."""
    return np.array([-direction[1], direction[0]])


def filter_points(
    points: np.ndarray, midpoint: np.ndarray, normal: np.ndarray, side: str
) -> np.ndarray:
    """
    Filters the given points based on their position relative to a plane defined by a midpoint and normal vector.

    Args:
        points (np.ndarray): The array of points to be filtered.
        midpoint (np.ndarray): The midpoint of the plane.
        normal (np.ndarray): The normal vector of the plane.
        side (str): The side of the plane to filter the points. Can be "left", "right", "middle", or any other value.

    Returns:
        np.ndarray: The filtered array of points.

    """
    dot_product = np.dot(points - midpoint, normal).round(3)
    if side == "left":
        return points[dot_product < 0]
    elif side == "right":
        return points[dot_product > 0]
    elif side == "middle":
        return points[dot_product == 0]
    else:
        return points


def mirror_points_across_line(
    points_to_mirror: list[Point2],
    line_start: Point2,
    line_end: Point2,
    by_midpoint: bool = False,
    side: str = "left",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function mirrors points across a line defined by two points (line_start and line_end).
    The side of the line to consider for mirroring is determined by side parameter.

    Args:
        points_to_mirror: The points to mirror.
        line_start: The start point of the line.
        line_end: The end point of the line.
        by_midpoint: Whether to mirror the points by the midpoint of the line.
        side: The side of the line to consider for mirroring.

    Returns:
        The original points, the mirrored points and the points that lie on the line.
    """
    points_to_mirror = np.array(points_to_mirror, dtype=float)
    line_start = np.array(line_start, dtype=float)
    line_end = np.array(line_end, dtype=float)

    line_direction = direction(line_start, line_end)
    line_normal = normal(line_direction).round(3)
    line_midpoint = (line_start + line_end) / 2

    middle_points = filter_points(
        points_to_mirror, line_midpoint, line_normal, "middle"
    )
    points_to_mirror = filter_points(points_to_mirror, line_midpoint, line_normal, side)

    if by_midpoint:
        mirrored_points = 2 * line_midpoint - points_to_mirror
    else:
        mirrored_points = points_to_mirror - 2 * np.outer(
            np.dot(points_to_mirror - line_midpoint, line_normal), line_normal
        )

    return points_to_mirror, mirrored_points.round(1), middle_points.round(1)


def perpendicular_bisector(
    line_start: Point2,
    line_end: Point2,
) -> tuple[Point2, Point2]:
    """
    This function calculates the perpendicular bisector of a line defined by two points (line_start and line_end).

    Args:
        line_start: The start point of the line.
        line_end: The end point of the line.

    Returns:
        The perpendicular bisector of the line.
    """
    return_type = type(line_start)
    line_start = np.array(line_start, dtype=float)
    line_end = np.array(line_end, dtype=float)

    line_direction = direction(line_start, line_end)
    line_normal = normal(line_direction)
    line_midpoint = (line_start + line_end) / 2

    # Calculate the points of the perpendicular bisecting line
    bisector_start = line_midpoint - line_normal
    bisector_end = line_midpoint + line_normal

    # Return the perpendicular bisecting line
    return return_type(bisector_start), return_type(bisector_end)
