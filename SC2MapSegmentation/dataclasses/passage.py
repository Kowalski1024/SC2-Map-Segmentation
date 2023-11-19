from dataclasses import dataclass

from sc2.client import Client
from sc2.game_info import GameInfo
from sc2.position import Point2, Point3


@dataclass(frozen=True)
class Passage:
    game_info: GameInfo
    vision_blockers: frozenset[Point2] | None
    destructables: set[int] | None
    minerals: set[int] | None

    titles: frozenset[Point2]
    surrounding: frozenset[Point2]

    def ramp_center(self) -> Point2:
        return Point2((sum(p.x for p in self.titles) / len(self.titles),
                       sum(p.y for p in self.titles) / len(self.titles)))

    def draw_boxes(self, client: Client, height_offset: int = -0.1):
        for point in self.titles:
            height = self._get_terrain_z_height(point)

            client.debug_text_world(f"{height:.2f}", Point3((point.x + 0.5, point.y + 0.5, height)), size=10)
            point = Point3((point.x + 0.5, point.y + 0.5, height + height_offset))
            client.debug_box2_out(point, color=(0, 255, 0))

        self.draw_surrounding(client, height_offset)

        center = self.ramp_center()
        center = Point3((center.x, center.y, self._get_terrain_z_height(center) + height_offset))
        client.debug_sphere_out(center, r=1, color=(255, 255, 255))

    def draw_surrounding(self, client: Client, height_offset: int = -0.15):
        for point in self.surrounding:
            height = self._get_terrain_z_height(point)

            client.debug_text_world(f"{height:.2f}", Point3((point.x + 0.5, point.y + 0.5, height)), size=10)
            point = Point3((point.x + 0.5, point.y + 0.5, height + height_offset))
            client.debug_box2_out(point, color=(255, 0, 0))

    def _get_terrain_z_height(self, pos: Point2) -> float:
        pos = pos.rounded
        return -16 + 32 * self.game_info.terrain_height[pos] / 255

    def __repr__(self):
        return f"{self.__class__.__name__}(points_len={len(self.titles)}, surrounding_len={len(self.surrounding)}, destructables={self.destructables}, minerals={self.minerals})"


@dataclass(frozen=True)
class Ramp(Passage):
    low_titles: tuple[Point2, ...]
    high_titles: tuple[Point2, ...]

    def draw_surrounding(self, client: Client, height_offset: int = -0.1):
        for point in self.low_titles:
            height = self._get_terrain_z_height(point)
            x = self.game_info.terrain_height[point]

            client.debug_text_world(f"{x:.2f}", Point3((point.x + 0.5, point.y + 0.5, height)), size=10)
            point = Point3((point.x + 0.5, point.y + 0.5, height + height_offset))
            client.debug_box2_out(point, color=(255, 0, 0))

        for point in self.high_titles:
            height = self._get_terrain_z_height(point)
            x = self.game_info.terrain_height[point]

            client.debug_text_world(f"{x:.2f}", Point3((point.x + 0.5, point.y + 0.5, height)), size=10)
            point = Point3((point.x + 0.5, point.y + 0.5, height + height_offset))
            client.debug_box2_out(point, color=(0, 0, 255))

    def __repr__(self):
        return f"{self.__class__.__name__}(points_len={len(self.titles)}, surrounding_len={len(self.surrounding)}, destructables={self.destructables}, minerals={self.minerals})"


@dataclass(frozen=True)
class ChokePoint(Passage):
    def __repr__(self):
        return f"{self.__class__.__name__}(points_len={len(self.titles)}, surrounding_len={len(self.surrounding)}, destructables={self.destructables}, minerals={self.minerals})"
