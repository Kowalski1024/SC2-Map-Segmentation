import time
from collections import Counter
from typing import Any, Iterable

import numpy as np
from loguru import logger
from skimage.draw import line
from skimage.morphology import flood_fill

from sc2.bot_ai_internal import BotAIInternal
from sc2.game_info import GameInfo
from sc2.position import Point2

from .algorithms import (
    filter_obtuse_points,
    find_surrounding,
    flood_fill_all,
    scan_unbuildable_points,
)
from .dataclasses.passage import ChokePoint, Passage, Ramp
from .dataclasses.region import Region
from .dataclasses.segmented_map import SegmentedMap
from .math import mirror_points_across_line, perpendicular_bisector
from .passages import (
    find_cliff_passages,
    find_passages,
    find_passages_between_regions,
    update_passage_connections,
)
from .utils.misc_utils import get_config, get_neighbors4

EMPTY_REGION_INDEX = 0


def map_segmentation(
    bot: BotAIInternal, configs_path: str = "MapSegmentation/configs"
) -> SegmentedMap:
    """
    Performs map segmentation based on the given bot and configuration path.

    Args:
        bot (BotAIInternal): The bot object containing game information.
        configs_path (str, optional): The path to the configuration files. Defaults to "MapSegmentation/configs".

    Returns:
        SegmentedMap: The segmented map object containing regions, passages, and other information.
    """

    def propagete(
        points: Iterable[Point2],
        region_config: dict[str, Any],
    ) -> None:
        """Propagates a region from a list of points"""
        max_value = np.max(segmentation_grid) + 1
        for i, location in enumerate(points):
            if segmentation_grid[location.rounded] == -1:
                propagate_region(
                    location=location,
                    region_id=i + max_value,
                    grid=segmentation_grid,
                    scaning_grid=scaning_grid,
                    game_info=bot.game_info,
                    max_distance=region_config.get("max_distance", 25),
                    filter_angle=region_config.get("filter_angle", np.pi * 0.6),
                )

    # ---------------------------- #

    logger.info("Starting map segmentation")
    start = time.time()

    config = get_config(bot.game_info.map_name, configs_path)
    map_center = bot.game_info.map_center

    placement_grid = bot.game_info.placement_grid.data_numpy.T
    pathing_grid = bot.game_info.pathing_grid.data_numpy.T

    # calculate the perpendicular bisector of the line
    # between the bot's start location and the enemy's start location
    line_start = min(bot.start_location, bot.enemy_start_locations[0])
    line_end = max(bot.start_location, bot.enemy_start_locations[0])
    segment = perpendicular_bisector(line_start, line_end)

    # find ramps and basics choke points
    passages = find_passages(bot.game_info, bot.destructables, bot.mineral_field)
    ramps = [passage for passage in passages if isinstance(passage, Ramp)]
    chokes = [passage for passage in passages if isinstance(passage, ChokePoint)]

    # prepare the grids
    scaning_grid = pathing_grid.copy()
    segmentation_grid = placement_grid.copy() * -1

    for passage in passages:
        if isinstance(passage, ChokePoint):
            indices = passage.tiles_indices
            segmentation_grid[indices] = 0
            scaning_grid[indices] = 0

    # locations
    locations = {
        "bases": bot.expansion_locations_list,
        "watch_towers": (unit.position for unit in bot.watchtowers),
        "ramps": get_propagation_locations(
            ramps, map_center, config["propagations"]["ramps"]
        ),
        "chokes": get_propagation_locations(
            chokes, map_center, config["propagations"]["chokes"]
        ),
    }

    # propagate the regions
    for name, points in locations.items():
        region_config = config["propagations"][name]

        if not region_config["enabled"]:
            logger.debug(f"Skipping {name}")
            continue

        logger.debug(f"Propagating {name}")

        if region_config.get("mirror", False):
            points = mirror_points(
                points,
                map_center,
                segment,
                side=region_config.get("mirror_side", "left"),
            )

        propagete(points, region_config)

    logger.debug("Clearing and relabeling the grid")
    clear_and_relabel_grid(segmentation_grid)

    logger.debug("Finding additional passages")
    passages += find_cliff_passages(bot.game_info, passages)
    passages += find_passages_between_regions(segmentation_grid, bot.game_info)
    passages = update_passage_connections(passages, segmentation_grid)

    logger.debug("Creating regions")
    regions = create_regions(bot, segmentation_grid, passages)

    # create the segmented map
    segmented_map = SegmentedMap(
        name=bot.game_info.map_name,
        regions_grid=segmentation_grid,
        regions=regions,
        passages=tuple(passages),
        base_locations=tuple(bot.expansion_locations_list),
        game_info=bot.game_info,
        config=config,
    )

    logger.debug("Updating passability")
    segmented_map.update_passability(bot.destructables, bot.mineral_field)

    logger.info(f"Map segmentation finished in {time.time() - start:.2f}s")
    return segmented_map


def mirror_points(
    points: Iterable[Point2],
    map_center: Point2,
    segment: tuple[Point2, Point2],
    side="left",
) -> list[Point2]:
    """
    Mirrors the given points across a line segment.

    Args:
        points (Iterable[Point2]): The points to be mirrored.
        map_center (Point2): The center of the map.
        segment (tuple[Point2, Point2]): The line segment across which the points will be mirrored.
        side (str, optional): The side of the line segment to mirror the points. Defaults to "left".

    Returns:
        list[Point2]: The mirrored points, sorted by their distance to the map center.
    """
    line_start, line_end = segment

    # center reflection
    if (line_start + line_end) / 2 == map_center:
        points, mirrored, middle = mirror_points_across_line(
            points, line_start, line_end, by_midpoint=True, side=side
        )
    else:
        points, mirrored, middle = mirror_points_across_line(
            points, line_start, line_end, side=side
        )

    locations = map(Point2, np.concatenate([middle, points, mirrored]))
    return sorted(locations, key=lambda point: point.distance_to(map_center))


def get_propagation_locations(
    passages: list[Passage], map_center: Point2, config: dict[str, int]
) -> list[Point2]:
    """
    Get the propagation locations for map segmentation.

    Args:
        passages (list[Passage]): List of passages.
        map_center (Point2): Center point of the map.
        config (dict[str, int]): Configuration dictionary.

    Returns:
        list[Point2]: List of propagation locations.
    """
    locations = [map_center]
    for passage in passages:
        if isinstance(passage, Ramp):
            locations.extend(
                passage.calculate_side_points(config["distance_multiplier"])
            )

    locations.sort(key=lambda point: point.distance_to(map_center))
    return locations


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
        """Creates a map of region ids to points"""
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
        # get the region data
        base = ids_map["bases"].get(region_id, [None])[0]
        watch_tower = ids_map["watch_towers"].get(region_id, [None])[0]
        region_vision_blockers = ids_map["vision_blockers"].get(region_id, ())

        # get the region indices and passages
        region_indices = np.where(segmented_grid == region_id)
        region_passages = [
            passage for passage in passages if region_id in passage.connections
        ]

        # create the region
        region = Region(
            id=region_id,
            indices=region_indices,
            passages=tuple(region_passages),
            base_location=base,
            watch_tower=watch_tower,
            vision_blockers=region_vision_blockers,
        )

        regions[region_id] = region

    return regions


def clear_and_relabel_grid(
    segmented_grid: np.ndarray, min_size: int = 50, region_scaling: float = 1.8
) -> None:
    """
    Clear and relabel the grid by removing small regions and relabeling unlabelled regions.

    Args:
        segmented_grid (np.ndarray): The segmented grid.
        min_size (int, optional): The minimum size of a region to be considered. Defaults to 50.
        region_scaling (float, optional): The scaling factor for region size comparison. Defaults to 1.8.
    """
    max_value = np.max(segmented_grid)
    # find the unlabbeled regions
    unlabbeled_regions = flood_fill_all(
        segmented_grid.shape,
        lambda point: segmented_grid[point] == -1,
        get_neighbors=get_neighbors4,
    )

    # add the unlabbeled regions to the grid as new regions
    for region in unlabbeled_regions:
        max_value += 1
        x, y = zip(*region)
        segmented_grid[x, y] = max_value

    # remove regions that are too small or are surrounded by a larger region
    for region_id in range(1, max_value + 1):
        indices = np.where(segmented_grid == region_id)
        region_points = [Point2((x, y)) for x, y in zip(*indices)]

        region_size = len(region_points)
        if region_size == 0:
            continue

        surrounding = find_surrounding(
            region_points,
            lambda point: segmented_grid[point] > 0,
            get_neighbors=get_neighbors4,
        )
        x, y = zip(*region_points)
        tiles_per_region = Counter((segmented_grid[point] for point in surrounding))

        # if the region is too small or is surrounded by a larger region
        if tiles_per_region and (
            (val := tiles_per_region.most_common(1)[0])[1] ** region_scaling
            >= region_size
            or region_size < min_size
        ):
            segmented_grid[x, y] = val[0]

        # if the region is too small and is not surrounded by a larger region
        elif region_size < 10:
            # find the closest region
            surrounding = find_surrounding(
                region_points, lambda point: segmented_grid[point] > 0
            )
            tiles_per_region = Counter((segmented_grid[point] for point in surrounding))

            # if there is a closest region then add the region to it otherwise remove the region
            if tiles_per_region:
                val = tiles_per_region.most_common(1)[0]
                segmented_grid[x, y] = val[0]
            else:
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
    """Propagates a region on the grid starting from the given location.

    Args:
        location (Point2): The starting location for region propagation.
        region_id (int): The ID of the region to be propagated.
        grid (np.ndarray): The grid representing the map.
        scaning_grid (np.ndarray): The grid used for scanning unbuildable points.
        game_info (GameInfo): Information about the game.
        max_distance (int, optional): The maximum distance to scan for unbuildable points. Defaults to 25.
        filter_angle (float, optional): The angle used to filter obtuse points. Defaults to np.pi * 0.6.
    """

    def find_chokes(points: list[Point2]) -> list[tuple[np.ndarray, np.ndarray]]:
        """Finds the choke points in a list of points"""
        return [
            line(point_a.x, point_a.y, point_b.x, point_b.y)
            for point_a, point_b in zip(points, points[1:] + points[:1])
            if point_a.manhattan_distance(point_b) > 2
        ]

    def add_chokes(
        choke_points: list[tuple[np.ndarray, np.ndarray]],
        from_value: int,
        to_value: int,
    ) -> None:
        """Adds choke lines to the grid"""
        if not choke_points:
            return

        rows, columns = zip(*choke_points)
        rows = np.concatenate(rows)
        columns = np.concatenate(columns)
        grid[rows, columns] = np.where(
            grid[rows, columns] == from_value, to_value, grid[rows, columns]
        )

    # get the map center
    map_center = game_info.map_center

    if location.y > map_center.y:
        counterclockwise = True
    else:
        counterclockwise = False

    # scan the map for unbuildable points
    depth_points_raw = scan_unbuildable_points(
        location,
        scaning_grid,
        map_center,
        max_distance=max_distance,
        counterclockwise=counterclockwise,
    )
    depth_points_list = filter_obtuse_points(depth_points_raw, location, filter_angle)

    # find the choke points and add them to the grid
    chokes = find_chokes(depth_points_list)
    add_chokes(chokes, from_value=-1, to_value=-2)

    location = location.rounded

    # add the region to the grid
    flood_fill(
        grid,
        location,
        region_id,
        connectivity=1,
        in_place=True,
    )

    # remove the added choke points
    add_chokes(chokes, from_value=-2, to_value=-1)
