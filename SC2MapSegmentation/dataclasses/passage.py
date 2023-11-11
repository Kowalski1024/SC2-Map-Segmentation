from dataclasses import dataclass

from sc2.game_info import GameInfo
from sc2.bot_ai_internal import BotAIInternal
from sc2.position import Point2


@dataclass(frozen=True)
class Passage:
    game_info: GameInfo
    points: list[set[Point2]]
    destructables: set[int] | None
    minerals: set[int] | None


@dataclass(frozen=True)
class Ramp(Passage):
    pass


@dataclass(frozen=True)
class ChokePoint(Passage):
    pass
