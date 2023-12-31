from .dataclasses.map import Map
from sc2.bot_ai_internal import BotAIInternal
from typing import Iterable, Sequence
from collections import Counter

import numpy as np
from skimage.draw import line
from skimage.morphology import flood_fill
import matplotlib.pyplot as plt
import scipy

from sc2.position import Point2
from sc2.game_info import GameInfo

from .utils import group_points, flood_fill_all, find_surrounding
from .dataclasses.passage import Passage, Ramp
from .passages import find_passages


def map_segmentation(bot: BotAIInternal, rich: bool = True):
    placement_grid = bot.game_info.placement_grid.copy()
    pathing_grid = bot.game_info.pathing_grid.copy()

    passages = find_passages(bot.game_info, bot.destructables, bot.mineral_field)
    region_locations = sorted(region_location_from_passages(passages), 
                              key=lambda location: location.distance_to(bot.game_info.map_center))
    region_locations = bot.expansion_locations_list + region_locations

    segmentation_grid = placement_grid.data_numpy * -1

    for passage in passages:
        for point in passage.titles:
            segmentation_grid[point.y, point.x] = 0

    i = 0
    for location in region_locations:
        x, y = location.rounded

        if segmentation_grid[y, x] == -1:
            create_region(location, i+1, segmentation_grid, bot.game_info)
            i += 1

    segmentation_grid[segmentation_grid == -2] = -1

    passages += clear_grid(segmentation_grid, bot.game_info)

    plt.imshow(segmentation_grid)
    plt.show()

    return region_locations, passages


def clear_grid(grid: np.ndarray, game_info: GameInfo, min_size: int = 50):
    max_value = np.max(grid)
    unlabbeled_regions = flood_fill_all(grid, lambda point: grid[point.y, point.x] == -1)

    for region in unlabbeled_regions:
        max_value += 1
        for point in region:
            grid[point.y, point.x] = max_value

    for val in range(1, max_value + 1):
        indices = np.where(grid == val)
        region = [Point2((x, y)) for x, y in zip(indices[1], indices[0])]

        region_size = len(region)
        surrounding = find_surrounding(region, grid, lambda val: val > 0, neighbors4=True)

        tiles_per_region = Counter((grid[p.y, p.x] for p in surrounding))
        if tiles_per_region and ((val := tiles_per_region.most_common(1)[0])[1] ** 1.8 >= region_size or region_size < min_size):
            for point in region:
                grid[point.y, point.x] = val[0]
        elif region_size < 10:
            for point in region:
                grid[point.y, point.x] = 0

    passages = []
    connections_set = set()

    for val in range(1, max_value + 1):
        indices = np.where(grid == val)
        region = [Point2((x, y)) for x, y in zip(indices[1], indices[0])]

        surrounding = find_surrounding(region, grid, lambda val: val > 0, neighbors4=True)
        connections_set.update(surrounding)

    for group in group_points(connections_set, True):
        tiles_per_region = Counter((grid[p.y, p.x] for p in group))
        if len(tiles_per_region) > 1:
            passages.append(Passage(
                game_info=game_info,
                vision_blockers=None,
                destructables=None,
                minerals=None,
                titles=frozenset(),
                surrounding=frozenset(group),
            ))

    return passages


def create_region(location: Point2, index: int, grid: np.ndarray, game_info: GameInfo, max_depth: int = 25):
    def add_choke(point_a: Point2, point_b: Point2):
        """Adds a choke line to the grid"""
        rr, cc = line(point_a.y, point_a.x, point_b.y, point_b.x)
        grid[rr, cc] = np.where(grid[rr, cc] == -1, -2, grid[rr, cc])

    def find_chokes(points: list[Point2]) -> list[tuple[Point2, Point2]]:
        """Finds the choke points in a list of points"""
        return [
            (point_a, point_b)
            for point_a, point_b in zip(points, points[1:] + points[:1])
            if point_a.manhattan_distance(point_b) > 2
        ]
        

    depth_points_raw = depth_points(location, grid, game_info, max_distance=max_depth)
    depth_points_list = filter_convex_hull_points(depth_points_raw, location)

    for choke in find_chokes(depth_points_list):
        add_choke(*choke)

    location = location.rounded

    region_points = flood_fill(
        grid,
        (location.y, location.x),
        index,
        connectivity=1,
        in_place=True,
    )

    return region_points


def region_location_from_passages(
    passages: Iterable[Passage], distance: float = 5
) -> list[Point2]:
    """
    Returns a list of vectors that point from the center of the passages to the center of the surrounding tiles multiplied by the distance

    Args:
        passages (Iterable[Passage]): passages
        distance (float, optional): distance to multiply the vectors with

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
            vector = (group_center - center).normalized
            locations.append(group_center + vector * distance)

    return locations


def remove_non_convex_points(
    points: Sequence[Point2], location: Point2, angle: float = np.pi * 0.6
) -> list[Point2]:
    """
    Removes points from a list where two points are too far from each other 
    and the angle between them is obtuse to form a convex hull-like shape around the location

    Args:
        points (list[Point2]): list of points
        location (Point2): location

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

    n = len(points) - 1

    while n >= 0:
        if n == 0:
            break

        point1 = points[n]
        point2 = points[n - 1]

        if point1.manhattan_distance(point2) > 2:
            if point1.distance_to(location) < point2.distance_to(location):
                while angle_between_points(point1, location, point2) > angle:
                    points.pop(n - 1)
                    n -= 1
                    point2 = points[n - 1]

        n -= 1

    return points


def filter_convex_hull_points(
    points: Sequence[Point2], location: Point2, angle: float = np.pi * 0.6
) -> list[Point2]:
    """
    Filters points to form an convex hull-like shape around the location

    Args:
        points (list[Point2]): list of points
        location (Point2): location

    Returns:
        list[Point2]: list of points that form a convex hull around the location
    """
    points = points.copy()
    points = remove_non_convex_points(points, location, angle)
    points = remove_non_convex_points(points[::-1], location, angle)

    return points


def depth_points(
    location: Point2,
    grid: np.ndarray,
    game_info: GameInfo,
    step: int = 1,
    max_distance: int = 25,
) -> list[Point2]:
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
        return np.array(
            ((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta)))
        )

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


def scan_direction(
    location: Point2, direction: np.ndarray, grid: np.ndarray, max_distance: int
) -> tuple[Point2, int]:
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
        point = location + Point2(
            (direction[0][0] * distance, direction[1][0] * distance)
        )
        point = point.rounded

        if grid[point.y, point.x] == 0:
            return point, distance

    return None, max_distance
