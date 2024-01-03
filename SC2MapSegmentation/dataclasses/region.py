from typing import Optional, NamedTuple
import numpy as np

from sc2.position import Point2, Point3
from sc2.client import Client
from sc2.game_info import GameInfo

from .passage import Passage
from SC2MapSegmentation.utils.misc_utils import get_terrain_z_height
from SC2MapSegmentation.utils.debug_utils import WHITE, GREEN


class Region(NamedTuple):
    id: int
    indices: tuple[np.ndarray, np.ndarray]
    passages: tuple[Passage]

    base_location: Optional[Point2] = None
    watch_tower: Optional[Point2] = None
    vision_blockers: tuple[Point2, ...] = tuple()

    def center(self) -> Point2:
        x, y = self.indices[0].mean(), self.indices[1].mean()
        return Point2((x, y))

    def draw_center(
        self, game_info: GameInfo, client: Client, height_offset: float = 0
    ):
        center = self.center()
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
