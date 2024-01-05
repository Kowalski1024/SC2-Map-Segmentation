from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.position import Point3, Point2
import numpy as np
import math
from sc2.unit import UnitTypeId

from mapsegmentation.map_segmentation import map_segmentation, filter_obtuse_points, scan_unbuildable_points
from mapsegmentation.pathing import Djikstra

import matplotlib.pyplot as plt

class ReaperExample(BotAI):
    def __init__(self):
        super().__init__()
        self.map = None
        self.path = None
        self.djikstra = None

    async def on_start(self):
        grid = self.game_info.placement_grid.data_numpy.T
        map_center = self.game_info.map_center
        self.points = [(scan_unbuildable_points(base, grid, map_center), base) for base in self.expansion_locations_list]
        self.filtered_points = [(filter_obtuse_points(points, base), base) for points, base in self.points]


    async def on_step(self, iteration):
        # if iteration % 10 == 0 and self.units:
        #     reaper = self.units(UnitTypeId.REAPER)
        #     if reaper:
        #         reaper = reaper[0]
        #         position = reaper.position
        #         target = self.enemy_start_locations[0]
        #         path = self.djikstra(position, target, None, (Ramp,))
        #         self.path = path

        # if self.path:
        #     for a, b in zip(self.path, self.path[1:]):
        #         height_a = self.get_terrain_z_height(a) + 1
        #         height_b = self.get_terrain_z_height(b) + 1
        #         self.client.debug_line_out(Point3((a.x + 0.5, a.y + 0.5, height_a)), Point3((b.x + 0.5, b.y + 0.5, height_b)), color=(255, 0, 0))
        # for location in self.locations:
        #     height = self.get_terrain_z_height(location) + 1
        #     self.client.debug_box2_out(Point3((location.x + 0.5, location.y + 0.5, height)), color=(0, 0, 255))

        for points, base in self.filtered_points[::-1]:
            self.draw_points(points, base)
        map_center = self.game_info.map_center
        height = self.get_terrain_z_height(map_center) + 2
        self.client.debug_box2_out(Point3((map_center.x, map_center.y, height)), color=(0, 0, 255))
        self.client.debug_text_world(f"Map center:\n{map_center}", Point3((map_center.x, map_center.y, height+1)), color=(255, 255, 255))

        # for passage in self.map.passages:
        #     passage.draw_boxes(self.game_info, self.client)

        # for region in self.map.regions.values():
        #     region.draw_center(self.game_info, self.client)

        # for point_a, point_b in zip(self.points, self.points[1:]):
        #     height_a = self.get_terrain_z_height(point_a) + 1
        #     height_b = self.get_terrain_z_height(point_b) + 1
        #     self.client.debug_line_out(Point3((point_a.x + 0.5, point_a.y + 0.5, height_a)), Point3((point_b.x + 0.5, point_b.y + 0.5, height_b)), color=(255, 0, 0))

        # for passage in self.map.passages:
        #     center = passage.center()
        #     height = self.get_terrain_z_height(center) + 1
        #     self.client.debug_box2_out(Point3((center.x + 0.5, center.y + 0.5, height)), color=(0, 255, 0))
        #     self.client.debug_text_world(f"{center}", Point3((center.x + 0.5, center.y + 0.5, height+1)), color=(255, 255, 255))

    def draw_points(self, points, base):
        def angle_between_points(center: Point2, point1: Point2, point2: Point2) -> float:
            """Returns the angle between two points"""
            vector1 = tuple(point1 - center)
            vector2 = tuple(point2 - center)
            unit_vector1 = vector1 / np.linalg.norm(vector1)
            unit_vector2 = vector2 / np.linalg.norm(vector2)
            return math.acos(np.clip(np.dot(unit_vector1, unit_vector2), -1.0, 1.0))

        color = (0, 255, 0)
        height = self.get_terrain_z_height(base) + 0.1

        self.client.debug_box2_out(Point3((base.x + 0.5, base.y + 0.5, height)), color=color + (255,))
        self.client.debug_text_world(f"{base}", Point3((base.x, base.y, height+1)), color=(255, 255, 255))

        for point in points:
            self.client.debug_box2_out(Point3((point.x + 0.5, point.y + 0.5, height)), color=color)
        offset = Point2((0.5, 0.5))
        for a, b in zip(points, points[1:] + points[:1]):
            if a.manhattan_distance(b) > 2:
                self.client.debug_line_out(Point3((a.x + 0.5, a.y + 0.5, height)), Point3((b.x + 0.5, b.y + 0.5, height)), color=(255, 0, 0))
                angle_a = angle_between_points(a + offset, base, b + offset)
                angle_b = angle_between_points(b + offset, base, a + offset)
                self.client.debug_text_world(f"{angle_a:.4f}\n{a}", Point3((a.x + 0.5, a.y + 0.5, height+1)), color=(255, 255, 255))
                self.client.debug_text_world(f"{angle_b:.4f}\n{b}", Point3((b.x + 0.5, b.y + 0.5, height+1)), color=(255, 255, 255))
            else:
                self.client.debug_line_out(Point3((a.x + 0.5, a.y + 0.5, height)), Point3((b.x + 0.5, b.y + 0.5, height)), color=color)


def main():
    # EquilibriumAIE
    # GoldenauraAIE
    # GresvanAIE
    # HardLeadAIE
    # OceanbornAIE
    # SiteDeltaAIE

    run_game(
        maps.get("SiteDeltaAIE"),
        [Bot(Race.Protoss, ReaperExample()), Computer(Race.Protoss, Difficulty.Easy)],
        realtime=True,
    )


if __name__ == "__main__":
    main()