from collections import defaultdict
from typing import Any, Iterable, Mapping, Type, Optional

import numpy as np

from sc2.game_info import GameInfo
from sc2.position import Point2
from sc2.unit import Unit

from .dataclasses.passage import ChokePoint, Passage, Ramp, Cliff
from .algorithms import find_surrounding, flood_fill_all, group_connected_points, pathable_height_grid
from .utils.data_structures import Point, FindUnion
from .utils.misc_utils import (
    GROUND_HEIGHT,
    get_neighbors4,
    get_neighbors5x5_outer,
    mark_unbuildable_tiles,
)


def find_passages(
    game_info: GameInfo,
    destructables: Iterable[Unit],
    minerals: Iterable[Unit],
    threshold: int = 4,
) -> list[Passage]:
    pathing_grid = game_info.pathing_grid.data_numpy.T
    placement_grid = game_info.placement_grid.data_numpy.T
    height_grid = game_info.terrain_height.data_numpy.T
    vision_blockers = game_info.vision_blockers

    grid = mark_unbuildable_tiles(
        pathing_grid, placement_grid, destructables, minerals, vision_blockers
    )

    # find all ramps
    ramp_tiles = flood_fill_all(
        grid.shape,
        lambda point: grid[point] == 1 and height_grid[point] not in GROUND_HEIGHT,
    )
    ramp_tiles = [tiles for tiles in ramp_tiles if len(tiles) >= threshold]

    for tiles in ramp_tiles:
        tiles.update(find_surrounding(tiles, lambda point: grid[point] == 1))

    ramps = [
        create_passage(Ramp, game_info, tiles, destructables, minerals)
        for tiles in ramp_tiles
    ]

    # remove ramps from grid
    for ramp in ramps:
        grid[ramp.tiles_indices()] = 0

    # find all chokes
    choke_tiles = flood_fill_all(
        grid.shape,
        lambda point: grid[point] == 1 and height_grid[point] in GROUND_HEIGHT,
    )

    chokes = [
        create_passage(ChokePoint, game_info, tiles, destructables, minerals)
        for tiles in choke_tiles
        if len(tiles) >= threshold
    ]

    return ramps + chokes


def find_cliff_passages(
    game_info: GameInfo,
    passages: Iterable[Passage],
) -> list[Passage]:
    height_grid = pathable_height_grid(
        game_info.terrain_height.data_numpy.T,
        game_info.placement_grid.data_numpy.T,
    )

    for passage in passages:
        height_grid[passage.tiles_indices()] = 0

    find_union = FindUnion([])
    non_zero = np.nonzero(height_grid)
    for x, y in zip(*non_zero):
        height = height_grid[x, y]

        if height == 0 or height not in GROUND_HEIGHT:
            continue

        point = Point2((x, y))
        neighbors = get_neighbors5x5_outer(point)

        for neighbor in neighbors:
            neighbor_height = height_grid[neighbor]
            if (
                neighbor_height == 0
                or neighbor_height == height
                or height not in GROUND_HEIGHT
            ):
                continue

            find_union.add(point)
            find_union.add(neighbor)
            find_union.union(point, neighbor)

    passages = []
    groups = find_union.groups()
    for group in groups:
        group = np.array(list(group))
        passages.append(
            create_passage(Cliff, game_info, [], [], [], surrounding_tiles=group)
        )

    return passages


def find_passages_between_regions(
    segmented_grid: np.ndarray, game_info: GameInfo
) -> list[Passage]:
    passages = []
    connections_set = set()
    max_value = np.max(segmented_grid)

    # find passages between regions
    for region_id in range(1, max_value + 1):
        indices = np.where(segmented_grid == region_id)
        region_points = [Point(x, y) for x, y in zip(*indices)]

        surrounding = find_surrounding(
            region_points,
            lambda point: segmented_grid[point] > 0,
            get_neighbors=get_neighbors4,
        )
        connections_set.update(surrounding)

    # create passages
    for group in group_connected_points(connections_set, get_neighbors=get_neighbors4):
        unique_tiles = set(segmented_grid[point] for point in group)
        if len(unique_tiles) > 1:
            passages.append(
                create_passage(
                    ChokePoint, game_info, [], [], [], surrounding_tiles=group
                )
            )

    return passages


def update_passage_connections(
    passages: Iterable[Passage], segmented_grid: np.ndarray
) -> list[Passage]:
    results = []

    for passage in passages:
        if len(passage.surrounding_tiles) == 0:
            continue

        connections = defaultdict(list)
        for point in passage.surrounding_tiles:
            grid_value = segmented_grid[point]

            if grid_value == 0:
                continue

            connections[grid_value].append(point)

        if len(connections) == 1:
            continue

        passage.connections.update(
            {key: tuple(value) for key, value in connections.items()}
        )
        results.append(passage)

    return results


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
        "vision_blockers": frozenset(
            p for p in game_info.vision_blockers if p in tiles
        ),
        "minerals": {m.tag for m in minerals if m.position.rounded in tiles},
        "destructables": {d.tag for d in destructables if d.position.rounded in tiles},
        "connections": {},
    }


def create_passage(
    passage_type: Type[Passage],
    game_info: GameInfo,
    tiles: Iterable[Point],
    destructables: Iterable[Unit],
    minerals: Iterable[Unit],
    surrounding_tiles: Optional[Iterable[Point]] = None,
) -> Passage:
    grid = game_info.placement_grid.data_numpy.T

    if surrounding_tiles is None:
        surrounding_tiles = find_surrounding(tiles, lambda point: grid[point] == 1)

    info = passage_info(game_info, tiles, surrounding_tiles, destructables, minerals)

    if passage_type == Ramp:
        groups = group_connected_points(surrounding_tiles)

        if len(groups) != 2:
            raise ValueError(f"Expected 2 groups, got {len(groups)}.")

        return Ramp(**info)
    elif passage_type == ChokePoint:
        return ChokePoint(**info)
    elif passage_type == Cliff:
        return Cliff(**info)
    else:
        raise ValueError(f"Unknown passage type {passage_type}.")
