from collections import defaultdict
from queue import PriorityQueue
from typing import Callable, Any

from sc2.position import Point2

from .dataclasses.passage import Passage
from .dataclasses.region import Region
from .dataclasses.segmented_map import SegmentedMap


def default_cost_func(
    current: Point2, neighbor: Point2, costs: dict[Point2, float] = None
) -> float:
    distance = current.distance_to(neighbor)
    return distance + (costs.get(neighbor, 0) if costs else 0)


class Djikstra:
    def __init__(
        self,
        map: SegmentedMap,
        cost_func: Callable[[Point2, Point2, Any], float] = default_cost_func,
    ):
        self.map = map
        self.cost_func = cost_func

        self.mapping = {passage.center(): passage for passage in map.passages}

    def _neighbors(
        self, point: Point2, end: Point2, avoid: tuple[type[Passage]]
    ) -> list[Point2]:
        passage = self.mapping[point]
        neighbors = []

        for region in passage.connections:
            region = self.map.regions[region]

            if self._closest_region(end) is region:
                neighbors.append(end)
                continue

            for next_passage in region.passages:
                if next_passage is passage or avoid and isinstance(next_passage, avoid):
                    continue

                neighbors.append(next_passage.center())

        return neighbors

    def _closest_region(self, point: Point2) -> Region:
        if region := self.map.region_at(point):
            return region

        directions = (Point2((0, 1)), Point2((0, -1)), Point2((-1, 0)), Point2((1, 0)))

        for distance in range(1, 10):
            for direction in directions:
                neighbor = point + direction * distance

                if region := self.map.region_at(neighbor):
                    return region

    def __call__(
        self,
        start: Point2,
        end: Point2,
        costs: Any = None,
        avoid: tuple[type[Passage]] = (),
    ) -> list[Point2]:
        start_region = self._closest_region(start)
        end_region = self._closest_region(end)

        if start_region is end_region:
            return [start, end]

        visited: set[Point2] = {start}
        previous: dict[Point2, Point2] = {}
        queue: PriorityQueue[tuple[float, Point2]] = PriorityQueue()
        distances: dict[Point2, float] = defaultdict(lambda: float("inf"))

        distances[start] = 0

        for passage in start_region.passages:
            if avoid and isinstance(passage, avoid):
                continue

            distance = start.distance_to(passage.center())
            queue.put((distance, passage.center()))
            previous[passage.center()] = start
            distances[passage.center()] = distance

        while not queue.empty():
            _, current = queue.get()

            if current is end:
                path = [current]
                while current in previous:
                    current = previous[current]
                    path.append(current)

                return path[::-1]

            if current in visited:
                continue

            visited.add(current)

            for neighbor in self._neighbors(current, end, avoid):
                if neighbor in visited:
                    continue

                distance = distances[current] + self.cost_func(current, neighbor, costs)

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current
                    queue.put((distance, neighbor))

        return []
