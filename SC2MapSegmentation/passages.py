from typing import Iterable

from sc2.game_info import GameInfo
from sc2.position import Point2
from sc2.unit import Unit
from .dataclasses.passage import Passage, Ramp, ChokePoint
from .utils import unbuildable_pathing_grid, flood_fill_all, find_surrounding


def passage_builder(
        game_info: GameInfo,
        points: Iterable[Point2],
        destructables: Iterable[Unit],
        minerals: Iterable[Unit]
) -> Passage:
    def check_height(a: Point2, b: Point2) -> bool:
        return game_info.terrain_height[a] == game_info.terrain_height[b]

    surrounding = find_surrounding(points, game_info.placement_grid.data_numpy, lambda x: x == 1)
    points = frozenset(points)
    surrounding = frozenset(surrounding)
    vision_blockers = frozenset(p for p in game_info.vision_blockers if p in points)

    test_point = next(iter(surrounding))
    minerals = {m.tag for m in minerals if m.position.rounded in points}
    destructables = {d.tag for d in destructables if d.position.rounded in points}

    # check if the passage is a ramp
    if any(not check_height(test_point, p) for p in surrounding):
        return Ramp(game_info, points, surrounding, vision_blockers, minerals, destructables)

    return ChokePoint(game_info, points, surrounding, vision_blockers, minerals, destructables)


def find_passages(
        game_info: GameInfo,
        destructables: Iterable[Unit],
        minerals: Iterable[Unit],
        threshold: int = 6
) -> list[Passage]:
    """
    Finds passages on the map as well as the surrounding tiles. Mostly for ramps and destructables.

    Args:
        game_info (GameInfo): game info
        destructables (Units): destructables
        minerals (Units): minerals
        threshold (int, optional): minimum number of tiles to be considered as passage, default 6
    """
    pathing_grid = game_info.pathing_grid.data_numpy.copy()
    placement_grid = game_info.placement_grid.data_numpy.copy()
    vision_blockers = game_info.vision_blockers

    grid = unbuildable_pathing_grid(pathing_grid, placement_grid, destructables, minerals,
                                    vision_blockers)

    # bfs the grid to find groups of unbuildable tiles
    groups = flood_fill_all(grid, lambda x: x == 1)

    return [
        passage_builder(game_info, group, destructables, minerals)
        for group in groups
        if len(group) >= threshold
    ]
