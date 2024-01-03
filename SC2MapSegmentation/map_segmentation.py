from sc2.bot_ai_internal import BotAIInternal
from collections import Counter
from typing import Iterable, Mapping, Any

import numpy as np
from skimage.draw import line
from skimage.morphology import flood_fill
from loguru import logger

from sc2.position import Point2
from sc2.game_info import GameInfo

from .dataclasses.passage import Passage, ChokePoint, Ramp
from .dataclasses.region import Region
from .dataclasses.segmented_map import SegmentedMap
from .algorithms import (
    filter_obtuse_points,
    scan_unbuildable_points,
    flood_fill_all,
    find_surrounding,
)
from .utils.data_structures import Point
from .utils.misc_utils import get_neighbors4, get_config
from .passages import find_passages, find_passages_between_regions, update_passage_connections

import matplotlib.pyplot as plt


EMPTY_REGION_INDEX = 0


def map_segmentation(bot: BotAIInternal, configs_path: str = "SC2MapSegmentation\configs") -> SegmentedMap:
    def propagete(points: Iterable[Point2], region_config: str) -> None:
        max_value = np.max(segmentation_grid) + 1
        for i, location in enumerate(points):
            if segmentation_grid[location.rounded] == -1:
                propagate_region(
                    location=location,
                    region_id=i + max_value,
                    grid=segmentation_grid,
                    scaning_grid=scaning_grid,
                    game_info=bot.game_info,
                    max_distance=region_config["max_distance"],
                    filter_angle=region_config["filter_angle"],
                )

    config = get_config(bot.game_info.map_name, configs_path)
    map_center = bot.game_info.map_center

    placement_grid = bot.game_info.placement_grid.data_numpy.T
    pathing_grid = bot.game_info.pathing_grid.data_numpy.T

    # find ramps and basics choke points
    passages = find_passages(bot.game_info, bot.destructables, bot.mineral_field)

    # prepare the grids
    scaning_grid = pathing_grid.copy()
    segmentation_grid = placement_grid.copy() * -1

    for passage in passages:
        if isinstance(passage, ChokePoint):
            indices = passage.tiles_indices()
            segmentation_grid[indices] = 0
            scaning_grid[indices] = 0

    # create the regions from the base locations
    logger.info("Propagating base locations")
    propagete(bot.expansion_locations_list, config["bases"])

    # create the regions from the watch towers
    watch_towers_config = config["watch_towers"]
    if watch_towers_config["enabled"]:
        logger.info("Propagating watch towers")
        propagete((unit.position for unit in bot.watchtowers), watch_towers_config)

        # create the regions from the ramps
        logger.info("Propagating ramps")
        locations = []
        ramp_config = config["ramps"]
        for passage in passages:
            if isinstance(passage, Ramp):
                locations.extend(passage.calculate_side_points(ramp_config["distance_multiplier"]))
        
        locations.sort(key=lambda point: point.distance_to(map_center))
        propagete(locations, ramp_config)

    # create the region from choke points
    logger.info("Propagating choke points")
    locations = []
    choke_config = config["chokes"]
    for passage in passages:
        if isinstance(passage, ChokePoint):
            locations.extend(passage.calculate_side_points(choke_config["distance_multiplier"]))

    locations.sort(key=lambda point: point.distance_to(map_center))
    propagete(locations, choke_config)

    # clear and relabel the grid
    logger.info("Clearing and relabeling the grid")
    clear_and_relabel_grid(segmentation_grid)

    passages += find_passages_between_regions(segmentation_grid, bot.game_info)
    passages = update_passage_connections(passages, segmentation_grid)

    logger.info("Creating regions")
    regions = create_regions(bot, segmentation_grid, passages)

    segmented_map = SegmentedMap(
        name=bot.game_info.map_name,
        regions_grid=segmentation_grid,
        regions=regions,
        passages=tuple(passages),
        base_locations=tuple(bot.expansion_locations_list),
        game_info=bot.game_info,
    )

    plt.imshow(segmentation_grid)
    plt.show()

    return segmented_map


def create_regions(
    bot: BotAIInternal, segmented_grid: np.ndarray, passages: list[Passage]
) -> dict[int, Region]:
    """
    Creates regions from a segmented grid

    Args:
        bot (BotAIInternal): bot
        segmented_grid (np.ndarray): segmented grid
        passages (list[Passage]): passages

    Returns:
        dict[int, Region]: regions
    """

    def get_ids_map(
        point_types: dict[str, Iterable[Point2]],
    ) -> dict[str, dict[int, list[int]]]:
        ids_map = {}

        for point_type, points in point_types.items():
            ids_map[point_type] = {}

            for point in points:
                ids_map[point_type].setdefault(
                    segmented_grid[point.rounded], []
                ).append(point)

        return ids_map

    regions = {}
    max_value = np.max(segmented_grid)

    ids_map = get_ids_map(
        {
            "bases": bot.expansion_locations_list,
            "watch_towers": (unit.position for unit in bot.watchtowers),
            "vision_blockers": bot.game_info.vision_blockers,
        }
    )

    for region_id in range(1, max_value + 1):
        base = ids_map["bases"].get(region_id, [None])[0]
        watch_tower = ids_map["watch_towers"].get(region_id, [None])[0]
        region_vision_blockers = ids_map["vision_blockers"].get(region_id, ())

        region_indices = np.where(segmented_grid == region_id)
        region_passages = [
            passage for passage in passages if region_id in passage.connections
        ]

        region = Region(
            id=region_id,
            indices=region_indices,
            passages=tuple(region_passages),
            base_location=base,
            watch_tower=watch_tower,
            vision_blockers=region_vision_blockers,
        )

        regions[region_id] = region
    print("-" * 10)
    return regions


def clear_and_relabel_grid(
    segmented_grid: np.ndarray, min_size: int = 50, region_scaling: float = 1.8
) -> None:
    max_value = np.max(segmented_grid)
    # find the unlabbeled regions
    unlabbeled_regions = flood_fill_all(
        segmented_grid.shape, lambda point: segmented_grid[point] == -1
    )

    # add the unlabbeled regions to the grid as new regions
    for region in unlabbeled_regions:
        max_value += 1
        x, y = zip(*region)
        segmented_grid[x, y] = max_value

    # remove regions that are too small or are surrounded by a larger region
    for region_id in range(1, max_value + 1):
        indices = np.where(segmented_grid == region_id)
        region_points = [Point(x, y) for x, y in zip(*indices)]

        region_size = len(region_points)
        if region_size == 0:
            continue

        surrounding = find_surrounding(
            region_points, lambda point: segmented_grid[point] > 0
        )

        tiles_per_region = Counter((segmented_grid[point] for point in surrounding))
        if tiles_per_region and (
            (val := tiles_per_region.most_common(1)[0])[1] ** region_scaling
            >= region_size
            or region_size < min_size
        ):
            x, y = zip(*region_points)
            segmented_grid[x, y] = val[0]
        elif region_size < 10:
            x, y = zip(*region_points)
            segmented_grid[x, y] = 0

    # relabel the regions
    values = np.unique(segmented_grid)
    for i, value in enumerate(values):
        segmented_grid[segmented_grid == value] = i


def propagate_region(
    location: Point2,
    region_id: int,
    grid: np.ndarray,
    scaning_grid: np.ndarray,
    game_info: GameInfo,
    max_distance: int = 25,
    filter_angle: float = np.pi * 0.6,
) -> None:
    def find_chokes(points: list[Point2]) -> list[tuple[Point2, Point2]]:
        """Finds the choke points in a list of points"""
        return [
            (point_a, point_b)
            for point_a, point_b in zip(points, points[1:] + points[:1])
            if point_a.manhattan_distance(point_b) > 2
        ]

    def add_chokes(choke_points: list[tuple[Point2, Point2]], from_value: int, to_value: int) -> None:
        """Adds choke lines to the grid"""
        for point_a, point_b in choke_points:
            rr, cc = line(point_a.x, point_a.y, point_b.x, point_b.y)
            grid[rr, cc] = np.where(grid[rr, cc] == from_value, to_value, grid[rr, cc])

    map_center = game_info.map_center
    depth_points_raw = scan_unbuildable_points(
        location, scaning_grid, map_center, max_distance=max_distance
    )
    depth_points_list = filter_obtuse_points(depth_points_raw, location, filter_angle)

    chokes = find_chokes(depth_points_list)
    add_chokes(chokes, from_value=-1, to_value=-2)

    location = location.rounded

    flood_fill(
        grid,
        location,
        region_id,
        connectivity=1,
        in_place=True,
    )

    add_chokes(chokes, from_value=-2, to_value=-1)
