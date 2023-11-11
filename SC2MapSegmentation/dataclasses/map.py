from dataclasses import dataclass

import numpy as np

from .region import Region


@dataclass(frozen=True)
class Map:
    name: str
    regions_grid: np.ndarray
    regions: dict[int, Region]

