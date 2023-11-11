from typing import Callable, Iterable
from collections import deque

import numpy as np

from sc2.position import Point2
from sc2.units import Units

from .utils import unbuildable_pathing_grid, numpy_bfs, find_surrounding, group_points
from .dataclasses.passage import Passage, Ramp, ChokePoint


Point = tuple[int, int]



def find_passages(
        pathing_grid: np.ndarray,
        placement_grid: np.ndarray,
        height_map: np.ndarray,
        destructables: Units,
        minerals: Units,
        vision_blockers: Iterable[Point2],
        threshold: int = 6
) -> list[Passage]:
    """
    Finds passages on the map as well as the surrounding tiles. Mostly for ramps and destructables.

    Args:
        pathing_grid (np.ndarray): pathing grid
        placement_grid (np.ndarray): placement grid
        height_map (np.ndarray): height map
        destructables (Units): destructables
        minerals (Units): minerals
        vision_blockers (Iterable[Point2]): vision blockers
        threshold (int, optional): minimum number of tiles to be considered as passage, default 6
    """
    grid = unbuildable_pathing_grid(pathing_grid, placement_grid, destructables, minerals, vision_blockers)

    # bfs the grid to find groups of unbuildable tiles
    groups = []
    for point in np.argwhere(grid == 1):
        visited = numpy_bfs(grid, tuple(point))

        if len(visited) >= threshold:
            surrounding = find_surrounding(visited, grid)
            grouped = group_points(surrounding)

            # check for ramps
            if len(grouped) == 1:
                continue

            point = next(iter(grouped[0]))
            is_ramp = any(height_map[nx, ny] != height_map[point] for points in grouped for nx, ny in points)



