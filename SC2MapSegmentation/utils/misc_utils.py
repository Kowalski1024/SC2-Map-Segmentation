from itertools import chain
from typing import Iterable

import numpy as np

from sc2.game_info import GameInfo
from sc2.position import Point2
from sc2.unit import Unit

from .data_structures import Point
from .destructables import change_destructable_status_in_grid

GROUND_HEIGHT = (175, 191, 207)
FOUR_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
EIGHT_DIRECTIONS = FOUR_DIRECTIONS + [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def get_terrain_z_height(game_info: GameInfo, posistion: Point2) -> float:
    """
    Returns the terrain z height of a position

    Args:
        game_info (GameInfo): The game info of the map
        posistion (Point2): The position to get the terrain z height of

    Returns:
        float: The terrain z height of the position
    """
    return -16 + 32 * game_info.terrain_height[posistion.rounded] / 255


def valid_point(grid: np.ndarray, point: Point2) -> bool:
    """
    Returns whether a point is valid

    Args:
        grid (np.ndarray): The grid to check the point against
        point (Point2): The point to check

    Returns:
        bool: Whether the point is valid
    """
    return 0 <= point.x < grid.shape[0] and 0 <= point.y < grid.shape[1]


def get_neighbors4(point: Point) -> list[Point]:
    """
    Returns the 4 neighbors of a point in a grid

    Args:
        point (Point): The point to get the neighbors of

    Returns:
        list[Point]: The neighbors of the point
    """
    x, y = point
    return [Point(x + dx, y + dy) for dx, dy in FOUR_DIRECTIONS]


def get_neighbors8(point: Point) -> list[Point]:
    """
    Returns the 8 neighbors of a point in a grid

    Args:
        point (Point): The point to get the neighbors of

    Returns:
        list[Point]: The neighbors of the point
    """
    x, y = point
    return [Point(x + dx, y + dy) for dx, dy in EIGHT_DIRECTIONS]


def mark_unbuildable_tiles(
    pathing_grid: np.ndarray,
    placement_grid: np.ndarray,
    destructables: Iterable[Unit],
    minerals: Iterable[Unit],
    vision_blockers: Iterable[Point2],
) -> np.ndarray:
    """
    Creates a grid based on the pathing grid with unbuildable tiles marked as 1 and buildable tiles marked as 0.

    Args:
        pathing_grid (np.ndarray): the pathing grid of the map
        placement_grid (np.ndarray): the placement grid of the map
        destructables (Units): the destructables on the map
        minerals (Units): the minerals on the map
        vision_blockers (Iterable[Point2]): the vision blockers on the map

    Returns:
        np.ndarray: the unbuildable pathing grid
    """
    pathing_grid = pathing_grid.copy()
    placement_grid = placement_grid.copy()

    # Add placement grid to pathing grid
    pathing_grid[placement_grid == 1] = 1

    # Add destructables and minerals to pathing grid and remove from placement grid
    for unit in chain(destructables, minerals):
        change_destructable_status_in_grid(pathing_grid, unit, 1)
        change_destructable_status_in_grid(placement_grid, unit, 0)

    # Add vision blockers to pathing grid
    for x, y in vision_blockers:
        pathing_grid[x, y] = 1

    # Remove placement grid from pathing grid
    pathing_grid[placement_grid == 1] = 0

    return pathing_grid
