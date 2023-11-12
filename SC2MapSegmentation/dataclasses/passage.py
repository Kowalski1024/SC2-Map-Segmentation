from dataclasses import dataclass

from sc2.client import Client
from sc2.game_info import GameInfo
from sc2.position import Point2, Point3


@dataclass(frozen=True)
class Passage:
    game_info: GameInfo
    points: frozenset[Point2]
    surrounding: frozenset[Point2]
    vision_blockers: frozenset[Point2] | None
    destructables: set[int] | None
    minerals: set[int] | None

    def draw_boxes(self, client: Client, height_offset: int = -0.15):
        for point in self.points:
            point = Point3((point.x, point.y, self._get_terrain_z_height(point) + height_offset))
            client.debug_box2_out(point, color=(0, 255, 0))

        for point in self.surrounding:
            point = Point3((point.x, point.y, self._get_terrain_z_height(point) + height_offset))
            client.debug_box2_out(point, color=(255, 0, 0))

    def _get_terrain_z_height(self, pos: Point2) -> float:
        return -16 + 32 * self.game_info.terrain_height[pos] / 255

    def __repr__(self):
        return f"{self.__class__.__name__}(points_len={len(self.points)}, surrounding_len={len(self.surrounding)}, destructables={self.destructables}, minerals={self.minerals})"


@dataclass(frozen=True)
class Ramp(Passage):
    def __repr__(self):
        return f"{self.__class__.__name__}(points_len={len(self.points)}, surrounding_len={len(self.surrounding)}, destructables={self.destructables}, minerals={self.minerals})"


@dataclass(frozen=True)
class ChokePoint(Passage):
    def __repr__(self):
        return f"{self.__class__.__name__}(points_len={len(self.points)}, surrounding_len={len(self.surrounding)}, destructables={self.destructables}, minerals={self.minerals})"
