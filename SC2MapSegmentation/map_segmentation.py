from .dataclasses.map import Map
from sc2.bot_ai_internal import BotAIInternal


def map_segmentation(bot: BotAIInternal) -> Map:
    placement_grid = bot.game_info.placement_grid.data_numpy
    pathing_grid = bot.game_info.pathing_grid.data_numpy


def create_regions():
    pass



