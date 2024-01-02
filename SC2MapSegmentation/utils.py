from collections import deque
from typing import Iterable, Callable, TypeVar
from itertools import chain


import numpy as np

from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units
from sc2.game_info import GameInfo

from .utils.destructables import (destructable_2x2, destructable_2x4,
                            destructable_2x6, destructable_4x2,
                            destructable_4x4, destructable_4x12,
                            destructable_6x2, destructable_6x6,
                            destructable_12x4, destructable_BLUR,
                            destructable_ULBR)

T = TypeVar('T')


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


def flood_fill_points(start: Point2, points: Iterable) -> list[Point2]:
    """
    Flood fill algorithm for finding all points connected to a starting point

    Caution: This function is not optimized for speed

    Args:
        start (Point2): starting point
        points (Iterable[Point2]): points to search

    Returns:
        list[Point2]: points connected to the starting point
    """
    nodes: list[Point2] = []
    queue: deque[Point2] = deque([start])

    while queue:
        point = queue.pop()

        if point in nodes:
            continue

        if point not in points:
            continue

        nodes.append(point)
        x, y = point
        queue.extend(Point2((x + a, y + b)) for a in [-1, 0, 1] for b in [-1, 0, 1] if not (a == 0 and b == 0))

    return nodes


def flood_fill(start: Point2, grid: np.ndarray, pred: Callable[[Point2], bool]) -> set[Point2]:
    nodes: set[Point2] = set()
    queue: deque[Point2] = deque([start])
    width, height = grid.shape

    while queue:
        point = queue.pop()
        x, y = point

        if not (0 <= y < width and 0 <= x < height) or point in nodes:
            continue

        if pred(point):
            nodes.add(point)
            queue.extend(Point2((x + a, y + b)) for a in [-1, 0, 1] for b in [-1, 0, 1] if not (a == 0 and b == 0))

    return nodes


def flood_fill_all(grid: np.ndarray, pred: Callable[[Point2], bool]) -> list[set[Point2]]:
    groups: list[set[Point2]] = []
    width, height = grid.shape

    for y in range(width):
        for x in range(height):
            if any((x, y) in g for g in groups):
                continue

            point = Point2((x, y))

            if pred(point):
                groups.append(flood_fill(point, grid, pred))

    return groups


def find_surrounding(group: Iterable[Point2], grid: np.ndarray, pred: Callable[[int], bool], neighbors4=False) -> set[Point2]:
    width, height = grid.shape
    surrounding: set[Point2] = set()

    for point in group:
        if neighbors4:
            neighbors = point.neighbors4
        else:
            neighbors = point.neighbors8
            
        for x, y in neighbors:
            if 0 <= y < width and 0 <= x < height and pred(grid[y, x]):
                surrounding.add(Point2((x, y)))

    surrounding.difference_update(group)
    return surrounding


def group_points(points: Iterable[Point2], neighbors4=False) -> list[set[Point2]]:
    """
    Groups points together by finding the connected components of the graph.

    Args:
        points (set[Point2]): points to group

    Returns:
        list[set[Point2]]: list of groups
    """
    find_union = FindUnion(points)

    for point in points:
        if neighbors4:
            neighbors = point.neighbors4
        else:
            neighbors = point.neighbors8

        for p in neighbors:
            if p in points:
                find_union.union(point, p)

    return find_union.groups()


def unbuildable_pathing_grid(
    pathing_grid: np.ndarray,
    placement_grid: np.ndarray,
    destructables: Iterable[Unit],
    minerals: Iterable[Unit],
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
    grid = np.transpose(grid)
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

    
