from dataclasses import dataclass

import numpy as np

from sc2.game_info import GameInfo
from sc2.position import Point2

from .region import Region
from .passage import Passage


@dataclass(frozen=True)
class Map:
    name: str
    regions_grid: np.ndarray
    regions: dict[int, Region]
    passages: tuple[Passage]
    base_locations: tuple[Point2, ...]

    game_info: GameInfo