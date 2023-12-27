from .dataclasses.map import Map
from sc2.bot_ai_internal import BotAIInternal
from typing import Iterable, Sequence
from itertools import chain, cycle

import numpy as np

from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units
from sc2.game_info import GameInfo

from .utils import group_points
from .dataclasses.passage import Passage, Ramp, ChokePoint
from .passages import find_passages


def map_segmentation(bot: BotAIInternal, rich: bool = True) -> Map:
    placement_grid = bot.game_info.placement_grid.copy()
    pathing_grid = bot.game_info.pathing_grid.copy()

    passages = find_passages(bot.game_info, bot.destructables, bot.mineral_field)


def regions_grid() -> np.ndarray:
    pass


def create_region(location: Point2, grid: np.ndarray, game_info: GameInfo):
    pass


def region_location_from_passages(passages: Iterable[Passage], distance: int = 5) -> list[Point2]:
    """
    Returns a list of vectors that point from the center of the passages to the center of the surrounding tiles multiplied by the distance

    Args:
        passages (Iterable[Passage]): passages
        distance (int, optional): distance to multiply the vectors with

    Returns:
        list[Point2]: list of vectors
    """
    
    locations = []

    for passage in passages:
        groups = group_points(passage.surrounding)

        if len(groups) == 1:
            continue

        center = Point2.center(passage.titles)

        for group in groups:
            group_center = Point2.center(group)

            vector = group_center - center

            locations.append(center + vector * distance)

    return locations


def remove_non_convex_points(points: Sequence[Point2], location: Point2) -> list[Point2]:
    """
    Removes points from a list where two points are too far from each other and the angle between them is obtuse
    to form a convex hull-like shape around the location
    
    Args:
        points (list[Point2]): list of points
        location (Point2): location
        
    Returns:
        list[Point2]: list of points without non-convex points
    """
    def is_obtuse_angle(point1: Point2, point2: Point2, point3: Point2) -> bool:
        """Returns true if the angle between the three points is obtuse"""
        a = point1.distance_to(point2)
        b = point2.distance_to(point3)
        c = point3.distance_to(point1)
        return a**2 + b**2 < c**2

    n = len(points) - 1

    while n >= 0:
        if n == 0:
            break

        point1 = points[n]
        point2 = points[n - 1]

        if point1.manhattan_distance(point2) > 2 and point1.distance_to(location) < point2.distance_to(location):
            while is_obtuse_angle(location, point1, point2):
                points.pop(n - 1)
                n -= 1
                point2 = points[n - 1]

        n -= 1

    return points

def filter_convex_hull_points(points: Sequence[Point2], location: Point2) -> list[Point2]:
    """
    Filters points to form an convex hull-like shape around the location

    Args:
        points (list[Point2]): list of points
        location (Point2): location

    Returns:
        list[Point2]: list of points that form a convex hull around the location
    """
    points = points.copy()
    points = remove_non_convex_points(points, location)
    points = remove_non_convex_points(points[::-1], location)

    return points


def depth_points(location: Point2, grid: np.ndarray, game_info: GameInfo, step: int = 1, max_distance: int = 30) -> list[Point2]:
    """
    Scans the map in a circle around the location and returns the first point that is not buildable

    Args:
        location (Point2): location to start the scan
        grid (np.ndarray): grid to scan
        game_info (GameInfo): game info
        step (int, optional): step size in degrees
        max_distance (int, optional): maximum distance to scan

    Returns:
        list[Point2]: list of points that are not buildable
    """
    def rotation_matrix(degrees):
        """Returns a rotation matrix for counterclockwise direction in degrees"""
        theta = np.radians(degrees)
        return np.array(((np.cos(theta), -np.sin(theta)),
                        (np.sin(theta), np.cos(theta))))

    # get the direction vector from the location to the map center
    ray = location.direction_vector(game_info.map_center).normalized
    ray = np.array([[ray.x], [ray.y]])

    point_list = []

    # scan in a circle around the location
    for degree in range(0, 360, step):
        rotated_ray = np.matmul(rotation_matrix(degree), ray)
        point, _ = scan_direction(location, rotated_ray, grid, max_distance)

        if point is not None:
            point_list.append(point)

    # clean from duplicates
    point_list = list(dict.fromkeys(point_list))

    return point_list


def scan_direction(location: Point2, direction: np.ndarray, grid: np.ndarray, max_distance: int) -> tuple[Point2, int]:
    """
    Scans the map in a certain direction until it finds a non-buildable point

    Args:
        location (Point2): location to start the scan
        direction (np.ndarray): direction to scan
        grid (np.ndarray): grid to scan
        max_distance (int): maximum distance to scan

    Returns:
        tuple[Point2, int]: first non-buildable point and the distance to it
    """

    for distance in range(1, max_distance):
        point = location + Point2((direction[0][0] * distance, direction[1][0] * distance))
        point = point.rounded

        if grid[point.y, point.x] == 0:
            return point, distance

    return None, max_distance
