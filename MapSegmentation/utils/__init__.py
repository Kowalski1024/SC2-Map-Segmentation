from typing import NamedTuple, Union

from sc2.position import Point2


class TuplePoint(NamedTuple):
    x: int
    y: int


Point = Union[Point2, TuplePoint]
