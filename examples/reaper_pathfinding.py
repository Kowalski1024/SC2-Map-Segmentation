from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.position import Point3
from sc2.unit import UnitTypeId

from MapSegmentation.map_segmentation import map_segmentation
from MapSegmentation.pathing import Djikstra
from MapSegmentation.dataclasses.passage import Ramp, Passage, ChokePoint, Cliff

import matplotlib.pyplot as plt

"""
This example shows how to use the MapSegmentation library to segment a map.
A Reaper is spawned and a path is calculated from its position to the enemy start location, avoiding ramps.
You can change the `avoid` parameter to exclude different types of passages or None to include all passages.

The map is segmented using the map_segmentation function, which returns a SegmentedMap object.

Debug drawing is used to show the path and the regions and passages of the map.
"""

class ReaperExample(BotAI):
    def __init__(self):
        super().__init__()
        self.map = None
        self.path = None
        self.djikstra = None

    async def on_start(self):
        self.map = map_segmentation(self)

        print("Close the plot window to continue")
        plt.figure(figsize=(12, 6))
        plt.axis("off")
        plt.title(f"Map: {self.game_info.map_name}")
        plt.subplot(1, 2, 2)
        self.map.imshow("Segmented grid")
        plt.subplot(1, 2, 1)
        plt.imshow(self.game_info.placement_grid.data_numpy)
        plt.title("Placement grid")
        plt.savefig(f"{self.game_info.map_name}.png")
        plt.show()

        self.djikstra = Djikstra(self.map)

        # spawn one reaper
        await self.client.debug_create_unit([[UnitTypeId.REAPER, 1, self.start_location, 1]])

    async def on_step(self, iteration):
        # calculate path every 10 iterations
        if iteration % 10 == 0 and self.units:
            reaper = self.units(UnitTypeId.REAPER)
            if reaper:
                reaper = reaper[0]
                position = reaper.position
                target = self.enemy_start_locations[0]

                # add avoid parameter to exclude different types of passages
                path = self.djikstra(position, target, None, avoid=(Ramp,))
                self.path = path

        # debug drawing
        if self.path:
            for a, b in zip(self.path, self.path[1:]):
                height_a = self.get_terrain_z_height(a) + 1
                height_b = self.get_terrain_z_height(b) + 1
                self.client.debug_line_out(Point3((a.x + 0.5, a.y + 0.5, height_a)), Point3((b.x + 0.5, b.y + 0.5, height_b)), color=(255, 0, 0))

        map_center = self.game_info.map_center
        height = self.get_terrain_z_height(map_center) + 2
        self.client.debug_box2_out(Point3((map_center.x, map_center.y, height)), color=(0, 0, 255))
        self.client.debug_text_world(f"Map center:\n{map_center}", Point3((map_center.x, map_center.y, height+1)), color=(255, 255, 255))

        for passage in self.map.passages:
            passage.draw_boxes(self.game_info, self.client)

        for region in self.map.regions.values():
            region.draw_center(self.game_info, self.client)


def main():
    # EquilibriumAIE
    # GoldenauraAIE
    # GresvanAIE
    # HardLeadAIE
    # OceanbornAIE
    # SiteDeltaAIE

    run_game(
        maps.get("GresvanAIE"),
        [Bot(Race.Terran, ReaperExample()), Computer(Race.Protoss, Difficulty.Easy)],
        realtime=True,
    )


if __name__ == "__main__":
    main()
