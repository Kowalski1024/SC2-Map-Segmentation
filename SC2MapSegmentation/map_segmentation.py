from .dataclasses.map import Map
from sc2.bot_ai_internal import BotAIInternal
from typing import Iterable
from itertools import chain

import numpy as np

from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from .utils import change_destructable_status_in_grid
from .dataclasses.passage import Passage, Ramp, ChokePoint


def map_segmentation(bot: BotAIInternal, rich: bool = True) -> Map:
    placement_grid = bot.game_info.placement_grid.data_numpy.copy().T
    pathing_grid = bot.game_info.pathing_grid.data_numpy.copy().T


def regions_grid(
) -> np.ndarray:
    pass


def create_regions():
    pass


def bases_locations():
    pass


