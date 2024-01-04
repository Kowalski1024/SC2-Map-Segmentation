from typing import NamedTuple, Optional

import numpy as np

from sc2.game_info import GameInfo
from sc2.position import Point2
from sc2.units import Units

from .region import Region
from .passage import Passage
from MapSegmentation.algorithms import flood_fill


class SegmentedMap(NamedTuple):
    name: str
    regions_grid: np.ndarray
    regions: dict[int, Region]
    passages: tuple[Passage]
    base_locations: tuple[Point2, ...]

    game_info: GameInfo
    config: dict[str, int]

    def region_at(self, point: Point2) -> Optional[Region]:
        point = point.rounded
        try:
            return self.regions[self.regions_grid[point]]
        except KeyError:
            return None
        
    def update_passability(self, destructables: Units, minerals: Units):
        destructables = {d.tag for d in destructables}
        minerals = {m.tag for m in minerals}

        for passage in self.passages:
            passage.destructables.intersection_update(destructables)
            passage.minerals.intersection_update(minerals)

            if len(passage.destructables) == 0 and len(passage.minerals) == 0:
                passage.passable = True
            else:
                titles = set.union(passage.tiles, passage.surrounding_tiles)
                random_point = next(iter(passage.surrounding_tiles))
                