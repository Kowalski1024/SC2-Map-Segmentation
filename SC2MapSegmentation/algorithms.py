from collections import deque
from itertools import chain
from typing import Callable, Iterable, Sequence

import numpy as np

from sc2.position import Point2

from .utils.data_structures import FindUnion, Point
from .utils.misc_utils import get_neighbors8


def pathable_height_grid(terrain_height: np.ndarray, placement_grid: np.ndarray) -> np.ndarray:
    """
    Returns a grid of the height of the pathable tiles

    Args:
        terrain_height (np.ndarray): terrain height grid
        placement_grid (np.ndarray): placement grid

    Returns:
        np.ndarray: height grid
    """
    height_grid = terrain_height.copy()
    pathing_grid = placement_grid

    group_of_tiles = flood_fill_all(
        pathing_grid.shape,
        lambda point: pathing_grid[point] == 1,
    )

    for group in group_of_tiles:
        point = next(iter(group))
        height = height_grid[point]
        x, y = np.array(list(zip(*group)))
        height_grid[x, y] = height

    np.putmask(height_grid, np.logical_not(pathing_grid), 0)

    return height_grid


def flood_fill(
    start: Point,
    is_accessible: Callable[[Point], bool],
    get_neighbors: Callable[[Point], list[Point]] = get_neighbors8,
) -> set[Point]:
    """
    Finds all accessible areas in a grid starting at a point

    Args:
        start (Point): The point to start the search at
        is_accessible (Callable[[Point], bool]): The function to determine if a point is accessible
        get_neighbors (Callable[[Point], list[Point]], optional): The function to get the neighbors of a point

    Returns:
        set[Point]: The accessible areas
    """
    seen: set[Point] = set()
    queue: deque[Point] = deque([start])

    while queue:
        point = queue.pop()

        if point in seen:
            continue

        if not is_accessible(point):
            continue

        seen.add(point)
        queue.extend(get_neighbors(point))

    return seen


def flood_fill_all(
    grid_shape: tuple[int, int],
    is_accessible: Callable[[Point], bool],
    get_neighbors: Callable[[Point], list[Point]] = get_neighbors8,
) -> list[set[Point]]:
    """
    Finds all accessible areas in a grid

    Args:
        grid_shape (tuple[int, int]): The shape of the grid
        is_accessible (Callable[[Point], bool]): The function to determine if a point is accessible
        get_neighbors (Callable[[Point], list[Point]], optional): The function to get the neighbors of a point

    Returns:
        list[set[Point]]: The accessible areas
    """
    groups: list[set[Point]] = []
    seen: set[Point] = set()
    width, height = grid_shape

    for x in range(width):
        for y in range(height):
            point = Point(x, y)

            if point in seen or not is_accessible(point):
                continue

            group = flood_fill(point, is_accessible, get_neighbors)
            groups.append(group)
            seen.update(group)

    return groups


def find_surrounding(
    group: Iterable[Point],
    is_accessible: Callable[[Point], bool],
    get_neighbors: Callable[[Point], list[Point]] = get_neighbors8,
) -> set[Point]:
    """
    Finds the accessible surrounding of a group of points

    Args:
        group (Iterable[Point]): The group of points
        grid (np.ndarray): The grid to search
        is_accessible (Callable[[Point], bool]): The function to determine if a point is accessible
        get_neighbors (Callable[[Point], list[Point]], optional): The function to get the neighbors of a point

    Returns:
        set[Point]: The accessible surrounding of the group
    """
    surrounding: set[Point] = set()
    group_set: set[Point] = set(group)

    for current in group:
        for point in get_neighbors(current):
            if point in group_set:
                continue

            if not is_accessible(point):
                continue

            surrounding.add(point)

    return surrounding


def group_connected_points(
    points: Iterable[Point],
    get_neighbors: Callable[[Point], list[Point]] = get_neighbors8,
) -> list[set[Point]]:
    """
    Groups points together by finding the connected components of the graph.

    Args:
        points (Iterable[Point]): The points to group
        get_neighbors (Callable[[Point], list[Point]]): The function to get the neighbors of a point

    Returns:
        list[set[Point]]: list of groups of points
    """
    find_union = FindUnion(points)
    points_set = set(points)

    for point in points:
        for p in get_neighbors(point):
            if p in points_set:
                find_union.union(point, p)

    return find_union.groups()


def filter_obtuse_points(
    points: Sequence[Point2], location: Point2, angle: float = np.pi * 0.6
) -> list[Point2]:
    """
    Filters points to form an convex hull-like shape around the location

    Args:
        points (list[Point2]): list of points
        location (Point2): location
        angle (float, optional): maximum angle between two points

    Returns:
        list[Point2]: list of points without non-convex points
    """

    def angle_between_points(center: Point2, point1: Point2, point2: Point2) -> float:
        """Returns the angle between two points"""
        vector1 = tuple(point1 - center)
        vector2 = tuple(point2 - center)
        unit_vector1 = vector1 / np.linalg.norm(vector1)
        unit_vector2 = vector2 / np.linalg.norm(vector2)
        return np.arccos(np.clip(np.dot(unit_vector1, unit_vector2), -1.0, 1.0))

    def filter_points(points: Sequence[Point2]) -> dict[Point2, None]:
        """Filters points to form an convex hull-like shape around the location"""
        new_points = {point: None for point in points}
        points_iter = chain(points, points)
        point1 = next(points_iter)
        offset = Point2((0.5, 0.5))

        try:
            for point2 in points_iter:
                # if the distance between two points is too far and the angle between them is obtuse
                # then skip the second point till the angle between them is desired
                if point1.manhattan_distance(point2) > 2:
                    while angle_between_points(point1 + offset, location, point2 + offset) > angle:
                        new_points.pop(point2)
                        point2 = next(points_iter)

                point1 = point2
        except (StopIteration, KeyError):
            pass

        return new_points

    dict_a = filter_points(points[::-1])
    dict_b = filter_points(points)
    return [key for key in dict_a.keys() if key in dict_b]


def scan_unbuildable_points(
    location: Point2,
    grid: np.ndarray,
    map_center: Point2,
    step: int = 1,
    max_distance: int = 25,
    counterclockwise: bool = True,
) -> list[Point2]:
    """
    Scans the map in a circle around the location and returns the first point that is not buildable

    Args:
        location (Point2): location to start the scan
        grid (np.ndarray): grid to scan, 0 value is not buildable
        map_center (Point2): center of the map
        step (int, optional): step size in degrees
        max_distance (int, optional): maximum distance to scan
        counterclockwise (bool, optional): direction to scan

    Returns:
        list[Point2]: list of points that are not buildable
    """

    def rotation_matrix(degrees):
        """Returns a rotation matrix for counterclockwise direction in degrees"""
        theta = np.radians(degrees)
        return np.array(
            ((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta)))
        )

    # get the direction vector from the location to the map center
    try:
        ray = (location - map_center).normalized
    except AssertionError:
        ray = Point2((0, 1))

    ray = np.array([[ray.x], [ray.y]])

    point_list = {}

    if counterclockwise:
        degrees = range(0, 360, step)
    else:
        degrees = range(0, -360, -step)

    # scan in a circle around the location
    for degree in degrees:
        rotated_ray = np.matmul(rotation_matrix(degree), ray)
        point, _ = scan_unbuildable_direction(location, rotated_ray, grid, max_distance)

        if point is not None:
            point_list[point] = None

    return list(point_list.keys())


def scan_unbuildable_direction(
    location: Point2, direction: np.ndarray, grid: np.ndarray, max_distance: int
) -> tuple[Point2, int]:
    """
    Scans the map in a certain direction until it finds a non-buildable point

    Args:
        location (Point2): location to start the scan
        direction (np.ndarray): direction to scan
        grid (np.ndarray): grid to scan, 0 value is not buildable
        max_distance (int): maximum distance to scan

    Returns:
        tuple[Point2, int]: first non-buildable point and the distance to it
    """

    for distance in range(1, max_distance):
        # get the point at the distance
        point = location + Point2(
            (direction[0][0] * distance, direction[1][0] * distance)
        )
        point = point.rounded

        # if the point is not buildable then return it
        if grid[point] == 0:
            return point, distance

    return None, max_distance
