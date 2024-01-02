from typing import Optional, NamedTuple

import numpy as np

from sc2.client import Client
from sc2.game_info import GameInfo
from sc2.position import Point2, Point3

import SC2MapSegmentation.utils.debug_utils as debug
from SC2MapSegmentation.utils.misc_utils import get_terrain_z_height


class Passage(NamedTuple):
    surrounding_tiles: frozenset[Point2]
    tiles: Optional[frozenset[Point2]] = None

    vision_blockers: Optional[frozenset[Point2]] = None
    destructables: Optional[set[int]] = None
    minerals: Optional[set[int]] = None

    connections: dict[int, tuple[Point2, ...]] = dict()

    def center(self) -> Point2:
        points = self.tiles if self.tiles else self.surrounding_tiles
        return Point2.center(points)
    
    def tiles_indices(self) -> tuple[np.array, np.array]:
        x, y = zip(*self.tiles)
        return np.array(x), np.array(y)
    
    def __hash__(self) -> int:
        return hash(self.tiles)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"center={self.center}, "
            f"surrounding_length={len(self.surrounding_tiles)}, "
            f"tiles_length={len(self.tiles)}, "
            f"vision_blockers_length={len(self.vision_blockers)}, "
            f"destructables_length={len(self.destructables)}, "
            f"minerals_length={len(self.minerals)}, "
            f"connections_keys={self.connections.keys()})"
        )
    
    def draw_boxes(self, game_info: GameInfo, client: Client, height_offset: int = -0.15):
        for point in self.tiles:
            debug.draw_point(game_info, client, point, height_offset + 0.1, debug.RED)

        self.draw_surrounding_regions(game_info, client, height_offset)

        center = self.center()
        center = Point3((center.x, center.y, get_terrain_z_height(game_info, center)))
        client.debug_sphere_out(center, r=1, color=debug.WHITE)

    def draw_surrounding_regions(self, game_info: GameInfo, client: Client, height_offset: int = -0.15):
        if self.connections:
            for region, points in self.connections.items():
                for point in points:
                    debug.draw_point(game_info, client, point, height_offset, debug.GREEN, f"{region}")
        else:
            for point in self.surrounding_tiles:
                debug.draw_point(game_info, client, point, height_offset, debug.GREEN)
    

class Ramp(Passage):
    pass


class ChokePoint(Passage):
    pass


class Cliff(Passage):
    pass