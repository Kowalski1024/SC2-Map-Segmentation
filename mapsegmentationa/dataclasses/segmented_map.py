import pickle
from dataclasses import dataclass
from itertools import chain
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from mapsegmentationa.algorithms import flood_fill
from mapsegmentationa.utils.destructables import change_destructable_status
from sc2.game_info import GameInfo
from sc2.position import Point2
from sc2.units import Units

from .passage import Passage
from .region import Region


@dataclass
class SegmentedMap:
    """
    Represents a segmented map.

    Attributes:
        name (str): The name of the map
        regions_grid (np.ndarray): The grid representing the regions of the map.
        regions (dict[int, Region]): A dictionary mapping region IDs to Region objects.
        passages (tuple[Passage]): A tuple of Passage objects.
        base_locations (tuple[Point2, ...]): A tuple of base locations.

        game_info (GameInfo): The game information.
        config (dict[str, int]): A dictionary containing configuration settings.
    """

    name: str
    regions_grid: np.ndarray
    regions: dict[int, Region]
    passages: tuple[Passage]
    base_locations: tuple[Point2, ...]

    game_info: GameInfo
    config: dict[str, int]

    def region_at(self, point: Point2) -> Optional[Region]:
        """
        Returns the region at the given point.

        Args:
            point (Point2): The point to check.

        Returns:
            Optional[Region]: The region at the given point, or None if no region is found.
        """
        point = point.rounded
        try:
            return self.regions[self.regions_grid[point]]
        except KeyError:
            return None

    def save(self, path: str) -> None:
        """
        Saves the segmented map to a file.

        Args:
            path (str): The path to the file.
        """
        self.game_info = None
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str, game_info: GameInfo) -> "SegmentedMap":
        """
        Loads a segmented map from a file.

        Args:
            path (str): The path to the file.
            game_info (GameInfo): The game information.

        Returns:
            SegmentedMap: The loaded segmented map.
        """
        with open(path, "rb") as f:
            segmented_map = pickle.load(f)
        segmented_map.game_info = game_info
        return segmented_map

    def imshow(self, title=None, region_labels=True) -> None:
        """
        Display the segmented map.

        Args:
            title (str, optional): The title of the plot. Defaults to None.
            region_labels (bool, optional): Whether to display region labels. Defaults to True.
        """
        cmap = plt.cm.get_cmap('plasma', len(self.regions))
        cmap.set_under(color='#440154')

        plt.imshow(self.regions_grid.T, cmap=cmap, vmin=0.01)

        if title:
            plt.title(title)

        if region_labels:
            for region in self.regions.values():
                plt.text(
                    *region.center,
                    region.id,
                    color="black",
                    ha="center",
                    va="center",
                    bbox={"facecolor": "white", "edgecolor": "black", "pad": 1},
                    size=8,
                )


    def update_passability(self, destructables: Units, minerals: Units) -> None:
        """
        Updates the passability of the map passages based on the presence of destructibles and minerals.

        Args:
            destructables (Units): A collection of destructible units.
            minerals (Units): A collection of mineral units.
        """
        for passage in self.passages:
            if len(passage.destructables) == 0 and len(passage.minerals) == 0:
                passage.passable = True
                continue

            # update destructables and minerals
            passage.destructables.intersection_update({d.position for d in destructables})
            passage.minerals.intersection_update({m.position for m in minerals})

            if passage.passable:
                continue

            if len(passage.destructables) == 0 and len(passage.minerals) == 0:
                passage.passable = True
                continue

            # prepare pathing grid
            pathing_grid = self.game_info.pathing_grid.data_numpy.copy().T
            for unit in chain(destructables, minerals):
                change_destructable_status(pathing_grid, unit, False)

            # flood fill from a random point in the passage
            surrounding_tiles = passage.surrounding_tiles
            random_point = next(iter(surrounding_tiles))
            tiles = frozenset.union(passage.tiles, surrounding_tiles)
            filled = flood_fill(
                random_point,
                lambda p, tiles=tiles, pathing_grid=pathing_grid: p in tiles and pathing_grid[p],
            )

            # if surrounding_tiles is subset of flood, passage is passable
            if surrounding_tiles.issubset(filled):
                passage.passable = True
                continue
