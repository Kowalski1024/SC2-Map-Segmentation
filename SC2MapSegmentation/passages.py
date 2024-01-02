from typing import Iterable, Any, Mapping, Type, Union
import random
import numpy as np
from collections import Counter, defaultdict

from sc2.game_info import GameInfo
from sc2.position import Point2
from sc2.unit import Unit
from .dataclasses.passage import Passage, Ramp, ChokePoint

from .utils.misc_utils import GROUND_HEIGHT, mark_unbuildable_tiles
from .utils.algorithms import group_connected_points, flood_fill_all, find_surrounding, flood_fill_all
from .utils.data_structures import Point


def passage_info(
        game_info: GameInfo,
        tiles: Iterable[Point],
        surrounding_tiles: Iterable[Point],
        destructables: Iterable[Unit],
        minerals: Iterable[Unit],
) -> Mapping[str, Any]:
    surrounding_tiles = frozenset(map(Point2, surrounding_tiles))
    tiles = frozenset(map(Point2, tiles))

    return {
        "tiles": tiles,
        "surrounding_tiles": surrounding_tiles,
        "vision_blockers": frozenset(p for p in game_info.vision_blockers if p in tiles),
        "minerals": {m.tag for m in minerals if m.position.rounded in tiles},
        "destructables": {d.tag for d in destructables if d.position.rounded in tiles}
    }


def create_passage(
        passage_type: Type[Passage],
        game_info: GameInfo,
        tiles: Iterable[Point],
        destructables: Iterable[Unit],
        minerals: Iterable[Unit]
) -> Passage:
    grid = game_info.placement_grid.data_numpy.T
    surrounding_tiles = find_surrounding(tiles, lambda p: grid[p.x, p.y] == 1)
    info = passage_info(game_info, tiles, surrounding_tiles, destructables, minerals)

    if passage_type == Ramp:
        groups = group_connected_points(surrounding_tiles)

        if len(groups) != 2:
            raise ValueError(f"Expected 2 groups, got {len(groups)}.")

        return Ramp(
            **info,
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

    for group in group_connected_points(connections_set, True):
        tiles_per_region = Counter((region_grid[p.y, p.x] for p in group))
        if len(tiles_per_region) > 1:
            passages.append(Passage(
                game_info=game_info,
                vision_blockers=None,
                destructables=None,
                minerals=None,
                titles=frozenset(),
                surrounding_tiles=frozenset(group),
            ))

    return passages


def clean_and_update(passages: Iterable[Passage], grid: np.ndarray):
    results = []

    for passage in passages:
        if len(passage.surrounding_tiles) == 0:
            continue

        connections = defaultdict(list)
        for point in passage.surrounding_tiles:
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
    pathing_grid = game_info.pathing_grid.data_numpy.T
    placement_grid = game_info.placement_grid.data_numpy.T
    height_grid = game_info.terrain_height.data_numpy.T
    vision_blockers = game_info.vision_blockers

    grid = mark_unbuildable_tiles(pathing_grid, placement_grid, destructables, minerals, vision_blockers)

    # find all ramps
    ramp_tiles = flood_fill_all(grid.shape, lambda p: grid[p.x, p.y] == 1 and height_grid[p.x, p.y] not in GROUND_HEIGHT)
    ramp_tiles = [tiles for tiles in ramp_tiles if len(tiles) >= threshold]

    for tiles in ramp_tiles:
        tiles.update(find_surrounding(tiles, lambda p: grid[p.x, p.y] == 1))

    ramps = [
        create_passage(Ramp, game_info, tiles, destructables, minerals)
        for tiles in ramp_tiles
    ]

    # remove ramps from grid
    for ramp in ramps:
        grid[ramp.tiles_indices()] = 0

    # find all chokes
    choke_tiles = flood_fill_all(grid.shape, lambda p: grid[p.x, p.y] == 1 and height_grid[p.x, p.y] in GROUND_HEIGHT)

    chokes = [
        create_passage(ChokePoint, game_info, tiles, destructables, minerals)
        for tiles in choke_tiles
        if len(tiles) >= threshold
    ]

    return ramps + chokes
