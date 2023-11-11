from .dataclasses.map import Map
from sc2.bot_ai_internal import BotAIInternal
from typing import Iterable
from itertools import chain

import numpy as np

from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from .utils import change_destructable_status_in_grid, neighbor8, numpy_bfs
from .dataclasses.passage import Passage, Ramp, ChokePoint


def map_segmentation(bot: BotAIInternal, rich: bool = True) -> Map:
    placement_grid = bot.game_info.placement_grid.data_numpy.copy()
    pathing_grid = bot.game_info.pathing_grid.data_numpy.copy()


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


def regions_grid(
) -> np.ndarray:
    pass


def create_regions():
    pass


def bases_locations():
    pass


def _find_ramps_and_blockers(
        pathing_grid: np.ndarray,
        placement_grid: np.ndarray,
        destructables: Units,
        minerals: Units,
        vision_blockers: Iterable[Point2]
) -> list[Passage]:
    grid = unbuildable_pathing_grid(pathing_grid, placement_grid, destructables, minerals, vision_blockers)

    # bfs the grid to find groups of unbuildable tiles
    groups = []
    for x, y in np.nonzero(grid):
        if (x, y) not in groups:
            groups.append(numpy_bfs(grid, x, y, neighbor8))


