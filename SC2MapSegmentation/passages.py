from typing import Iterable, Any, Mapping, Type, Union
import random
import numpy as np
from collections import Counter, defaultdict
from itertools import chain

from sc2.game_info import GameInfo
from sc2.position import Point2
from sc2.unit import Unit
from .dataclasses.passage import Passage, Ramp, ChokePoint
from .utils import unbuildable_pathing_grid, flood_fill_all, find_surrounding, group_points, flood_fill_points, flood_fill

import matplotlib.pyplot as plt


GROUND_HEIGHT = (175, 191, 207)


def create_edge(points: Iterable[Point2]) -> tuple[Point2, ...]:
    """
    Creates an edge of a passage, e.g. the low ground tiles of a ramp.

    Args:
        points (Iterable[Point2]): points to create the edge from

    Returns:
        tuple[Point2, ...]: edge points
    """
    center = Point2.center(points)  # type: ignore
    furthest = max(points, key=lambda p: p.distance_to(center))
    return tuple(flood_fill_points(furthest, points))


def passage_info(
        game_info: GameInfo,
        titles: Iterable[Point2],
        destructables: Iterable[Unit],
        minerals: Iterable[Unit],
) -> Mapping[str, Any]:
    """
    Creates a dictionary with information about a passage.

    Args:
        game_info (GameInfo): game info
        titles (Iterable[Point2]): tiles of the passage
        destructables (Iterable[Unit]): destructables in the passage
        minerals (Iterable[Unit]): minerals in the passage

    Returns:
        Mapping[str, Any]: passage information
    """

    return {
        "game_info": game_info,
        "titles": frozenset(titles),
        "surrounding": frozenset(find_surrounding(titles, game_info.placement_grid.data_numpy, lambda x: x == 1)),
        "vision_blockers": frozenset(p for p in game_info.vision_blockers if p in titles),
        "minerals": {m.tag for m in minerals if m.position.rounded in titles},
        "destructables": {d.tag for d in destructables if d.position.rounded in titles}
    }


def create_passage(
        passage_type: Type[Passage],
        game_info: GameInfo,
        tiles: Iterable[Point2],
        destructables: Iterable[Unit],
        minerals: Iterable[Unit]
) -> Passage:
    """
    Creates a passage.

    Args:
        passage_type (Type[Passage]): type of the passage
        game_info (GameInfo): game info
        tiles (Iterable[Point2]): tiles of the passage
        destructables (Iterable[Unit]): destructables in the passage
        minerals (Iterable[Unit]): minerals in the passage

    Returns:
        Passage: created passage
    """

    info = passage_info(game_info, tiles, destructables, minerals)

    if passage_type == Ramp:
        groups = group_points(info["surrounding"])

        if len(groups) != 2:
            raise ValueError(f"Expected 2 groups, got {len(groups)}.")

        low_tiles = create_edge(groups[0])
        high_tiles = create_edge(groups[1])

        # make sure the high ground is actually higher than the low ground
        low_center = Point2.center(low_tiles).rounded  # type: ignore
        high_center = Point2.center(high_tiles).rounded  # type: ignore
        if game_info.terrain_height[high_center] < game_info.terrain_height[low_center]:
            high_tiles, low_tiles = low_tiles, high_tiles

        return Ramp(
            **info,
            high_tiles=high_tiles,
            low_tiles=low_tiles
        )
    else:
        return ChokePoint(**info)
    

def find_region_passages(
        game_info: GameInfo,
        region_grid: np.ndarray,
) -> list[Passage]:
    passages = []
    connections_set = set()
    max_value = np.max(region_grid)

    for val in range(1, max_value + 1):
        indices = np.where(region_grid == val)
        region = [Point2((x, y)) for x, y in zip(indices[1], indices[0])]

        surrounding = find_surrounding(region, region_grid, lambda val: val > 0, neighbors4=True)
        connections_set.update(surrounding)

    for group in group_points(connections_set, True):
        tiles_per_region = Counter((region_grid[p.y, p.x] for p in group))
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


def clean_and_update(passages: Iterable[Passage], grid: np.ndarray):
    results = []

    for passage in passages:
        if len(passage.surrounding) == 0:
            continue

        connections = defaultdict(list)
        for point in passage.surrounding:
            val = grid[point.y, point.x]
            if val == 0:
                continue
            connections[val].append(point)

        if len(connections) == 1:
            continue

        connections = {key: tuple(value) for key, value in connections.items()}
        passage.connections.update(connections)
        results.append(passage)

    return results


def find_passages(
        game_info: GameInfo,
        destructables: Iterable[Unit],
        minerals: Iterable[Unit],
        threshold: int = 4
) -> list[Passage]:
    """
    Finds all passages in the map.

    Args:
        game_info (GameInfo): game info
        destructables (Iterable[Unit]): destructables in the map
        minerals (Iterable[Unit]): minerals in the map
        threshold (int, optional): minimum size of passages

    Returns:
        list[Passage]: found passages
    """

    pathing_grid = game_info.pathing_grid.data_numpy
    placement_grid = game_info.placement_grid.data_numpy
    height_grid = game_info.terrain_height.data_numpy
    vision_blockers = game_info.vision_blockers

    grid = unbuildable_pathing_grid(pathing_grid, placement_grid, destructables, minerals,
                                    vision_blockers)

    # find all ramps
    ramp_tiles = flood_fill_all(grid, lambda p: grid[p.y, p.x] == 1 and height_grid[p.y, p.x] not in GROUND_HEIGHT)
    ramp_tiles = [tiles for tiles in ramp_tiles if len(tiles) >= threshold]
    for tiles in ramp_tiles:
        tiles.update(find_surrounding(tiles, grid, lambda x: x == 1))

    ramps = [
        create_passage(Ramp, game_info, tiles, destructables, minerals)
        for tiles in ramp_tiles
    ]

    # remove ramps from grid
    for ramp in ramps:
        for tile in ramp.titles:
            grid[tile.y, tile.x] = 0

    # find all chokes
    choke_tiles = flood_fill_all(grid, lambda p: grid[p.y, p.x] == 1 and height_grid[p.y, p.x] in GROUND_HEIGHT)

    chokes = [
        create_passage(ChokePoint, game_info, tiles, destructables, minerals)
        for tiles in choke_tiles
        if len(tiles) >= threshold
    ]

    return ramps + chokes
