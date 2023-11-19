from typing import Iterable, Any, Mapping
import random
import numpy as np

from sc2.game_info import GameInfo
from sc2.position import Point2
from sc2.unit import Unit
from .dataclasses.passage import Passage, Ramp, ChokePoint
from .utils import unbuildable_pathing_grid, flood_fill_all, find_surrounding, group_points, flood_fill_points, flood_fill

import matplotlib.pyplot as plt


def create_edge(points: Iterable[Point2]) -> tuple[Point2, ...]:
    center = Point2.center(points)  # type: ignore
    furthest = max(points, key=lambda p: p.distance_to(center))
    return tuple(flood_fill_points(furthest, points))


def passage_info(
        game_info: GameInfo,
        titles: Iterable[Point2],
        destructables: Iterable[Unit],
        minerals: Iterable[Unit],
) -> Mapping[str, Any]:
    return {
        "game_info": game_info,
        "titles": frozenset(titles),
        "surrounding": frozenset(find_surrounding(titles, game_info.placement_grid.data_numpy, lambda x: x == 1)),
        "vision_blockers": frozenset(p for p in game_info.vision_blockers if p in titles),
        "minerals": {m.tag for m in minerals if m.position.rounded in titles},
        "destructables": {d.tag for d in destructables if d.position.rounded in titles}
    }


def create_ramp(
        game_info: GameInfo,
        titles: Iterable[Point2],
        destructables: Iterable[Unit],
        minerals: Iterable[Unit]
) -> Ramp:
    info = passage_info(game_info, titles, destructables, minerals)

    groups = group_points(info["surrounding"])

    if len(groups) != 2:
        raise ValueError(f"Expected 2 groups, got {len(groups)}.")

    low_titles = create_edge(groups[0])
    high_titles = create_edge(groups[1])

    # make sure the high ground is actually higher than the low ground
    low_center = Point2.center(low_titles).rounded  # type: ignore
    high_center = Point2.center(high_titles).rounded  # type: ignore
    if game_info.terrain_height[high_center] < game_info.terrain_height[low_center]:
        high_titles, low_titles = low_titles, high_titles

    return Ramp(
        **info,
        high_titles=high_titles,
        low_titles=low_titles
    )


def create_choke_point(
        game_info: GameInfo,
        titles: Iterable[Point2],
        destructables: Iterable[Unit],
        minerals: Iterable[Unit]
) -> ChokePoint:
    return ChokePoint(**passage_info(game_info, titles, destructables, minerals))


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
    height_grid = game_info.terrain_height.data_numpy.copy()
    vision_blockers = game_info.vision_blockers

    grid = unbuildable_pathing_grid(pathing_grid, placement_grid, destructables, minerals,
                                    vision_blockers)

    # find all ramps
    ramp_titles = flood_fill_all(grid, lambda p: grid[p.y, p.x] == 1 and height_grid[p.y, p.x] not in (175, 191, 207))
    ramp_titles = [titles for titles in ramp_titles if len(titles) >= threshold]
    for titles in ramp_titles:
        titles.update(find_surrounding(titles, grid, lambda x: x == 1))

    ramps: list[Passage] = [
        create_ramp(game_info, titles, destructables, minerals)
        for titles in ramp_titles
    ]

    # remove ramps from grid
    for ramp in ramps:
        for title in ramp.titles:
            grid[title.y, title.x] = 0

    # find all chokes
    choke_titles = flood_fill_all(grid, lambda p: grid[p.y, p.x] == 1 and height_grid[p.y, p.x] in (175, 191, 207))

    chokes: list[Passage] = [
        create_choke_point(game_info, titles, destructables, minerals)
        for titles in choke_titles
        if len(titles) >= threshold
    ]

    return ramps + chokes

