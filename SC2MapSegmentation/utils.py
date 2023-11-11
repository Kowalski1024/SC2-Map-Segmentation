from collections import deque
from typing import Iterable, Callable, TypeVar
from itertools import chain

import numpy as np

from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from .destructables import (destructable_2x2, destructable_2x4,
                            destructable_2x6, destructable_4x2,
                            destructable_4x4, destructable_4x12,
                            destructable_6x2, destructable_6x6,
                            destructable_12x4, destructable_BLUR,
                            destructable_ULBR)

T = TypeVar('T')
Point = tuple[int, int]


class FindUnion:
    """
    Find-Union data structure

    Args:
        items (Iterable[T]): items to be added to the data structure
    """
    def __init__(self, items: Iterable[T]) -> None:
        self.items = items
        self.parents = {item: item for item in items}

    def find(self, item: T) -> T:
        if self.parents[item] == item:
            return item
        else:
            return self.find(self.parents[item])

    def union(self, item1: T, item2: T) -> None:
        self.parents[self.find(item1)] = self.find(item2)

    def groups(self) -> list[set[T]]:
        """
        Returns the groups of items in the data structure

        Returns:
            list[set[T]]: groups of items in the data structure
        """
        groups = {}
        for item in self.items:
            parent = self.find(item)
            if parent in groups:
                groups[parent].add(item)
            else:
                groups[parent] = {item}
        return list(groups.values())


def neighbor4(x: int, y: int) -> Iterable[Point]:
    """
    Returns the 4 neighbors of (x, y)

    Args:
        x (int): x coordinate
        y (int): y coordinate

    Returns:
        Iterable[tuple[int, int]]: 4 neighbors of (x, y)
    """
    return (x, y - 1), (x - 1, y), (x + 1, y), (x, y + 1)


def neighbor8(x: int, y: int) -> Iterable[Point]:
    """
    Returns the 8 neighbors of (x, y)

    Args:
        x (int): x coordinate
        y (int): y coordinate

    Returns:
        Iterable[tuple[int, int]]: 8 neighbors of (x, y)
    """
    return ((x - 1, y - 1), (x, y - 1), (x + 1, y - 1),
            (x - 1, y), (x + 1, y),
            (x - 1, y + 1), (x, y + 1), (x + 1, y + 1))


def numpy_bfs(grid: np.ndarray, point: Point, neighbor_func: Callable = neighbor8) -> set[Point]:
    """
    Performs a BFS on the grid starting at (x, y) and returns the visited points.

    Warning: This function modifies the grid.

    Args:
        grid (np.ndarray): numpy grid
        point (Point): starting point
        neighbor_func (Callable, optional): function that returns the neighbors of a point, default neighbor8

    Returns:
        set[Point]: visited points
    """
    queue = deque([point])
    visited = set()
    rows, cols = grid.shape

    assert 0 <= point[0] < rows and 0 <= point[1] < cols, f"Starting position ({point}) is out of bounds"

    while queue:
        x, y = queue.popleft()
        grid[x, y] = 0

        for nx, ny in neighbor_func(x, y):
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] == 0:
                queue.append((nx, ny))
                visited.add((nx, ny))

    return visited


def find_surrounding(group: set[Point], grid: np.ndarray, neighbor_func: Callable = neighbor8) -> set[Point]:
    """
    Finds the surrounding tiles of a group of points.

    Args:
        group (set[Point]): group of points
        grid (np.ndarray): numpy grid
        neighbor_func (Callable, optional): function that returns the neighbors of a point, default neighbor8

    Returns:
        set[Point]: surrounding tiles
    """
    return {(nx, ny) for point in group for nx, ny in neighbor_func(*point) if grid[nx, ny] == 1}


def group_points(points: set[Point], neighbor_func: Callable = neighbor8) -> list[set[Point]]:
    """
    Groups points together by finding the connected components of the graph.

    Args:
        points (set[Point]): points to group
        neighbor_func (Callable, optional): function that returns the neighbors of a point, default neighbor8

    Returns:
        list[set[Point]]: list of groups
    """
    find_union = FindUnion(points)

    for point in points:
        for nx, ny in neighbor_func(*point):
            if (nx, ny) in points:
                find_union.union(point, (nx, ny))

    return find_union.groups()


def unbuildable_pathing_grid(
    pathing_grid: np.ndarray,
    placement_grid: np.ndarray,
    destructables: Units,
    minerals: Units,
    vision_blockers: Iterable[Point2]
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

    # Add destructables to pathing grid and remove from placement grid
    for unit in chain(destructables, minerals):
        change_destructable_status_in_grid(pathing_grid, unit, 1)
        change_destructable_status_in_grid(placement_grid, unit, 0)

    # Add vision blockers to placement grid
    for point in vision_blockers:
        pathing_grid[point.y, point.x] = 1

    # Remove placement grid from pathing grid
    pathing_grid[placement_grid == 1] = 0

    return pathing_grid


"""
https://github.com/spudde123/SC2MapAnalysis/blob/master/MapAnalyzer/utils.py
"""
def change_destructable_status_in_grid(grid: np.ndarray, unit: Unit, status: int) -> None:
    """
    Set destructable positions to status, modifies the grid in place

    Args:
        grid (np.ndarray): numpy grid
        unit (Unit): unit
        status (int): status to set

    Returns:
        None
    """
    type_id = unit.type_id
    pos = unit.position
    name = unit.name

    # this is checked with name because the id of the small mineral destructables
    # has changed over patches and may cause problems
    if name == "MineralField450":
        x = int(pos[0]) - 1
        y = int(pos[1])
        grid[x: (x + 2), y] = status
    elif type_id in destructable_2x2:
        w = 2
        h = 2
        x = int(pos[0] - w / 2)
        y = int(pos[1] - h / 2)
        grid[x: (x + w), y: (y + h)] = status
    elif type_id in destructable_2x4:
        w = 2
        h = 4
        x = int(pos[0] - w / 2)
        y = int(pos[1] - h / 2)
        grid[x: (x + w), y: (y + h)] = status
    elif type_id in destructable_2x6:
        w = 2
        h = 6
        x = int(pos[0] - w / 2)
        y = int(pos[1] - h / 2)
        grid[x: (x + w), y: (y + h)] = status
    elif type_id in destructable_4x2:
        w = 4
        h = 2
        x = int(pos[0] - w / 2)
        y = int(pos[1] - h / 2)
        grid[x: (x + w), y: (y + h)] = status
    elif type_id in destructable_4x4:
        w = 4
        h = 4
        x = int(pos[0] - w / 2)
        y = int(pos[1] - h / 2)
        grid[x: (x + w), (y + 1): (y + h - 1)] = status
        grid[(x + 1): (x + w - 1), y: (y + h)] = status
    elif type_id in destructable_6x2:
        w = 6
        h = 2
        x = int(pos[0] - w / 2)
        y = int(pos[1] - h / 2)
        grid[x: (x + w), y: (y + h)] = status
    elif type_id in destructable_6x6:
        w = 6
        h = 6
        x = int(pos[0] - w / 2)
        y = int(pos[1] - h / 2)
        grid[x: (x + w), (y + 1): (y + h - 1)] = status
        grid[(x + 1): (x + w - 1), y: (y + h)] = status
    elif type_id in destructable_12x4:
        w = 12
        h = 4
        x = int(pos[0] - w / 2)
        y = int(pos[1] - h / 2)
        grid[x: (x + w), y: (y + h)] = status
    elif type_id in destructable_4x12:
        w = 4
        h = 12
        x = int(pos[0] - w / 2)
        y = int(pos[1] - h / 2)
        grid[x: (x + w), y: (y + h)] = status
    elif type_id in destructable_BLUR:
        x_ref = int(pos[0] - 5)
        y_pos = int(pos[1])
        grid[(x_ref + 6): (x_ref + 6 + 2), y_pos + 4] = status
        grid[(x_ref + 5): (x_ref + 5 + 4), y_pos + 3] = status
        grid[(x_ref + 4): (x_ref + 4 + 6), y_pos + 2] = status
        grid[(x_ref + 3): (x_ref + 3 + 7), y_pos + 1] = status
        grid[(x_ref + 2): (x_ref + 2 + 7), y_pos] = status
        grid[(x_ref + 1): (x_ref + 1 + 7), y_pos - 1] = status
        grid[(x_ref + 0): (x_ref + 0 + 7), y_pos - 2] = status
        grid[(x_ref + 0): (x_ref + 0 + 6), y_pos - 3] = status
        grid[(x_ref + 1): (x_ref + 1 + 4), y_pos - 4] = status
        grid[(x_ref + 2): (x_ref + 2 + 2), y_pos - 5] = status

    elif type_id in destructable_ULBR:
        x_ref = int(pos[0] - 5)
        y_pos = int(pos[1])
        grid[(x_ref + 6): (x_ref + 6 + 2), y_pos - 5] = status
        grid[(x_ref + 5): (x_ref + 5 + 4), y_pos - 4] = status
        grid[(x_ref + 4): (x_ref + 4 + 6), y_pos - 3] = status
        grid[(x_ref + 3): (x_ref + 3 + 7), y_pos - 2] = status
        grid[(x_ref + 2): (x_ref + 2 + 7), y_pos - 1] = status
        grid[(x_ref + 1): (x_ref + 1 + 7), y_pos] = status
        grid[(x_ref + 0): (x_ref + 0 + 7), y_pos + 1] = status
        grid[(x_ref + 0): (x_ref + 0 + 6), y_pos + 2] = status
        grid[(x_ref + 1): (x_ref + 1 + 4), y_pos + 3] = status
        grid[(x_ref + 2): (x_ref + 2 + 2), y_pos + 4] = status
