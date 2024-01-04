from typing import NamedTuple, Optional

import numpy as np

from sc2.game_info import GameInfo
from sc2.position import Point2

from .region import Region
from .passage import Passage


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