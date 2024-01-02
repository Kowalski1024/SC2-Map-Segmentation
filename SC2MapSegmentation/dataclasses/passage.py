from dataclasses import dataclass, field
from typing import Optional

from sc2.client import Client
from sc2.game_info import GameInfo
from sc2.position import Point2, Point3

from SC2MapSegmentation.utils import group_points


@dataclass(frozen=True)
class Passage:
    game_info: GameInfo
    vision_blockers: Optional[frozenset[Point2]]
    destructables: Optional[set[Point2]]
    minerals: Optional[set[Point2]]

    titles: frozenset[Point2]
    surrounding: frozenset[Point2]
    connections: dict[int, tuple[Point2, ...]] = field(default_factory=dict)

    def center(self) -> Point2:
        if self.titles:
            return Point2.center(self.titles)
        else:
            return Point2.center(self.surrounding)


    def draw_boxes(self, client: Client, height_offset: int = -0.1):
        for point in self.titles:
            height = self._get_terrain_z_height(point)

            client.debug_text_world(f"{height:.2f}", Point3((point.x + 0.5, point.y + 0.5, height)), size=10)
            point = Point3((point.x + 0.5, point.y + 0.5, height + height_offset))
            client.debug_box2_out(point, color=(0, 255, 0))

        self.draw_surrounding(client, height_offset)

        center = self.center()
        center = Point3((center.x, center.y, self._get_terrain_z_height(center) + height_offset))
        client.debug_sphere_out(center, r=1, color=(255, 255, 255))

    def draw_surrounding(self, client: Client, height_offset: int = -0.15):
        for region, points in self.connections.items():
            for point in points:
                height = self._get_terrain_z_height(point)

                client.debug_text_world(f"{region}", Point3((point.x + 0.5, point.y + 0.5, height)), size=10)
                point = Point3((point.x + 0.5, point.y + 0.5, height + height_offset))
                client.debug_box2_out(point, color=(255, 0, 0))

    def _get_terrain_z_height(self, pos: Point2) -> float:
        pos = pos.rounded
        return -16 + 32 * self.game_info.terrain_height[pos] / 255

    def __repr__(self):
        return f"{self.__class__.__name__}(points_len={len(self.titles)}, surrounding_len={len(self.surrounding)}, destructables={self.destructables}, minerals={self.minerals}), vision_blockers={self.vision_blockers})"


@dataclass(frozen=True)
class Ramp(Passage):
    low_tiles: tuple[Point2, ...] = field(default_factory=tuple)
    high_tiles: tuple[Point2, ...] = field(default_factory=tuple)

    def draw_surrounding(self, client: Client, height_offset: int = -0.1):
        for point in self.low_tiles:
            height = self._get_terrain_z_height(point)
            x = self.game_info.terrain_height[point]

            client.debug_text_world(f"{x:.2f}", Point3((point.x + 0.5, point.y + 0.5, height)), size=10)
            point = Point3((point.x + 0.5, point.y + 0.5, height + height_offset))
            client.debug_box2_out(point, color=(255, 0, 0))

        for point in self.high_tiles:
            height = self._get_terrain_z_height(point)
            x = self.game_info.terrain_height[point]

            client.debug_text_world(f"{x:.2f}", Point3((point.x + 0.5, point.y + 0.5, height)), size=10)
            point = Point3((point.x + 0.5, point.y + 0.5, height + height_offset))
            client.debug_box2_out(point, color=(0, 0, 255))

    def __repr__(self):
        return f"{self.__class__.__name__}(points_len={len(self.titles)}, surrounding_len={len(self.surrounding)}, destructables={self.destructables}, minerals={self.minerals}), vision_blockers={self.vision_blockers})"


@dataclass(frozen=True)
class ChokePoint(Passage):
    def __repr__(self):
        return f"{self.__class__.__name__}(points_len={len(self.titles)}, surrounding_len={len(self.surrounding)}, destructables={self.destructables}, minerals={self.minerals}), vision_blockers={self.vision_blockers})"
