from typing import Callable, Iterable
from collections import deque

import numpy as np


Point = tuple[int, int]


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


def passage_bfs(grid: np.ndarray, point: Point, neighbor_func: Callable = neighbor8,
                placement_grid: np.ndarray = None) -> tuple[set[Point], set[Point]]:
    """
    Performs a BFS on the grid starting at (x, y) and returns the visited points and border points.
    Border points are points that are buildable and are adjacent to found points.

    Args:
        grid (np.ndarray): numpy grid
        point (Point): starting point
        neighbor_func (Callable, optional): function that returns the neighbors of a point, defaults to neighbor8
        placement_grid (np.ndarray, optional): placement grid, defaults to None (no border points)

    Returns:
        tuple[set[Point], set[Point]]: visited points and border points
    """
    queue = deque([point])
    visited = set()
    border_points = set()
    rows, cols = grid.shape

    assert grid[point] == 0, f"Starting position ({point}) is not buildable"
    assert 0 <= point[0] < rows and 0 <= point[1] < cols, f"Starting position ({point}) is out of bounds"

    while queue:
        x, y = queue.popleft()
        grid[x, y] = 0

        for nx, ny in neighbor_func(x, y):
            if 0 <= nx < rows and 0 <= ny < cols:
                if grid[nx, ny] == 0:
                    queue.append((nx, ny))
                    visited.add((nx, ny))
                elif placement_grid is not None and placement_grid[nx, ny] == 1:
                    border_points.add((nx, ny))

    return visited, border_points
