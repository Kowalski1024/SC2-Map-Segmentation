from dataclasses import dataclass
from functools import cached_property, lru_cache
import numpy as np

from sc2.position import Point2
from sc2.game_info import GameInfo

from .passage import Passage


@dataclass(frozen=True)
class Region:
    indices: tuple[np.ndarray, np.ndarray]
    passages: list[Passage]
    game_info: GameInfo

    # base information
    base_location: Point2

    watch_towers: tuple[Point2, ...]

    @cached_property
    def tiles(self) -> frozenset[Point2]:
        return frozenset(Point2((x, y)) for x, y in zip(self.indices[1], self.indices[0]))
    
    @cached_property
    def center(self) -> Point2:
        return Point2((np.mean(self.indices[1]), np.mean(self.indices[0])))
