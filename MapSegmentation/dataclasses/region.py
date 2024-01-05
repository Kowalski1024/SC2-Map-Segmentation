from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np

from MapSegmentation.utils.debug_utils import GREEN, WHITE
from MapSegmentation.utils.misc_utils import get_terrain_z_height
from sc2.client import Client
from sc2.game_info import GameInfo
from sc2.position import Point2, Point3

from .passage import Passage


@dataclass
class Region:
    """
    Represents a region in the map.

    Attributes:
        id (int): The unique identifier of the region.
        indices (tuple[np.ndarray, np.ndarray]): The indices of the tiles that belong to the region.
        passages (tuple[Passage]): The passages that connect the region to other regions.

        base_location (Optional[Point2]): The base location associated with the region (if any).
        watch_tower (Optional[Point2]): The watch tower location associated with the region (if any).
        vision_blockers (tuple[Point2, ...]): The vision blockers within the region.
    """

    id: int
    indices: tuple[np.ndarray, np.ndarray]
    passages: tuple[Passage]

    base_location: Optional[Point2] = None
    watch_tower: Optional[Point2] = None
    vision_blockers: tuple[Point2, ...] = ()

    @cached_property
    def center(self) -> Point2:
        """
        Calculates the center point of the region.

        Returns:
            Point2: The center point of the region.
        """
        x, y = self.indices[0].mean(), self.indices[1].mean()
        return Point2((x, y))

    @cached_property
    def tiles(self) -> set[Point2]:
        """
        Returns a set of Point2 objects representing the tiles in the region.
        """
        return {Point2((x, y)) for x, y in zip(*self.indices)}

    def draw_center(
        self, game_info: GameInfo, client: Client, height_offset: float = 0
    ) -> None:
        """
        Draws the center of the region on the game map.

        Args:
            game_info (GameInfo): Information about the game map.
            client (Client): The client used to interact with the game.
            height_offset (float, optional): Offset to adjust the height of the center. Defaults to 0.
        """
        center = self.center
        height = get_terrain_z_height(game_info, center)
        client.debug_sphere_out(
            Point3((center.x + 0.5, center.y + 0.5, height + height_offset)),
            0.5,
            color=GREEN,
        )
        client.debug_text_world(
            f"id: {self.id}\nsize: {len(self.indices[0])}",
            Point3((center.x, center.y, height + height_offset + 0.5)),
            color=WHITE,
            size=10,
        )
