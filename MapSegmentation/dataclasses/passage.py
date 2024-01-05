from dataclasses import dataclass
from functools import cached_property
from typing import Iterable, Optional

import numpy as np

import mapsegmentation.utils.debug_utils as debug
from mapsegmentation.algorithms import group_connected_points
from mapsegmentation.utils.misc_utils import get_terrain_z_height
from sc2.client import Client
from sc2.game_info import GameInfo
from sc2.position import Point2, Point3


@dataclass
class Passage:
    """
    Represents a passage in the map.

    Attributes:
        connections (dict[int, tuple[Point2, ...]]): A dictionary mapping region IDs to a tuple of connected points.

        surrounding_tiles (frozenset[Point2]): A set of points representing the surrounding tiles of the passage.
        tiles (Optional[frozenset[Point2]]): A set of points representing the tiles within the passage.

        vision_blockers (Optional[frozenset[Point2]]): A set of points representing vision blockers within the passage.
        destructables (Optional[set[Point2]]): A set of points representing destructible objects within the passage.
        minerals (Optional[set[Point2]]): A set of points representing mineral fields within the passage.
        passable (bool): Indicates whether the passage is passable or not.
    """

    connections: dict[int, tuple[Point2, ...]]

    surrounding_tiles: frozenset[Point2]
    tiles: Optional[frozenset[Point2]] = None

    vision_blockers: Optional[frozenset[Point2]] = None
    destructables: Optional[set[Point2]] = None
    minerals: Optional[set[Point2]] = None

    passable: bool = False

    @cached_property
    def center(self) -> Point2:
        """
        Calculates the center point of the passage.

        Returns:
            The center point of the passage.
        """
        points = self.tiles if self.tiles else self.surrounding_tiles
        return Point2.center(points)

    @cached_property
    def tiles_indices(self) -> tuple[np.array, np.array]:
        """
        Returns the x and y indices of the tiles in the passage.

        Returns:
            A tuple containing two numpy arrays: the x indices and the y indices of the tiles.
        """
        x, y = zip(*self.tiles)
        return np.array(x), np.array(y)

    @cached_property
    def surrounding_tiles_indices(self) -> tuple[np.array, np.array]:
        """
        Returns the indices of the surrounding tiles as numpy arrays.

        Returns:
            A tuple containing two numpy arrays: the x-coordinates and y-coordinates of the surrounding tiles.
        """
        x, y = zip(*self.surrounding_tiles)
        return np.array(x), np.array(y)

    def calculate_side_points(self, distance_multiplier: int = 5) -> list[Point2]:
        """
        Calculates the side points of the passage.

        Args:
            distance_multiplier (int): Multiplier for the distance between the side points and the center point.
                Defaults to 5.

        Returns:
            list[Point2]: List of side points calculated based on the passage's connections or surrounding tiles.
        """

        def calculate_vector(point_group: Iterable[Point2], center: Point2) -> Point2:
            """Calculates a vector based on the given point group and center point."""
            side_center = Point2.center(point_group)
            vector = (side_center - center).normalized
            return side_center + vector * distance_multiplier

        locations = []
        center = self.center

        if self.connections and len(self.connections) > 1:
            for points in self.connections.values():
                locations.append(calculate_vector(points, center))
        else:
            groups = group_connected_points(self.surrounding_tiles)
            for group in groups:
                locations.append(calculate_vector(group, center))

        return locations

    def __hash__(self) -> int:
        return hash(self.tiles)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"center={self.center}, "
            f"connections_keys={self.connections.keys()}, "
            f"passable={self.passable})"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"center={self.center}, "
            f"connections_keys={self.connections.keys()}, "
            f"surrounding_length={len(self.surrounding_tiles)}, "
            f"tiles_length={len(self.tiles)}, "
            f"vision_blockers_length={len(self.vision_blockers)}, "
            f"destructables_length={len(self.destructables)}, "
            f"minerals_length={len(self.minerals)}, "
            f"passable={self.passable})"
        )

    def draw_boxes(
        self, game_info: GameInfo, client: Client, height_offset: float = -0.15
    ) -> None:
        """
        Draws boxes around the tiles of the passage.

        Args:
            game_info (GameInfo): Information about the game.
            client (Client): The client object used for drawing.
            height_offset (float, optional): The height offset for drawing the boxes. Defaults to -0.15.
        """
        for point in self.tiles:
            debug.draw_point(game_info, client, point, height_offset + 0.1, debug.RED)

        self.draw_surrounding_regions(game_info, client, height_offset)

        center = self.center
        center = Point3((center.x, center.y, get_terrain_z_height(game_info, center)))
        client.debug_sphere_out(center, r=1, color=debug.WHITE)
        client.debug_text_world(
            f"{self.__class__.__name__}", center, size=16, color=debug.WHITE
        )

    def draw_surrounding_regions(
        self, game_info: GameInfo, client: Client, height_offset: float = -0.15
    ) -> None:
        """
        Draws the surrounding regions on the game map.

        Args:
            game_info (GameInfo): Information about the game map.
            client (Client): The client object used for drawing.
            height_offset (float, optional): The height offset for drawing the boxes. Defaults to -0.15.
        """
        if self.connections:
            for region, points in self.connections.items():
                for point in points:
                    debug.draw_point(
                        game_info,
                        client,
                        point,
                        height_offset,
                        debug.GREEN,
                        f"{region}",
                    )
        else:
            for point in self.surrounding_tiles:
                debug.draw_point(game_info, client, point, height_offset, debug.GREEN)


@dataclass
class Ramp(Passage):
    pass


@dataclass
class ChokePoint(Passage):
    pass


@dataclass
class Cliff(Passage):
    pass
