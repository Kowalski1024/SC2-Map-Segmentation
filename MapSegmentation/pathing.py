from collections import defaultdict
from queue import PriorityQueue
from typing import Any, Callable

from sc2.position import Point2

from .dataclasses.passage import Passage
from .dataclasses.region import Region
from .dataclasses.segmented_map import SegmentedMap


def default_cost_func(
    current: Point2, neighbor: Point2, costs: dict[Point2, float] = None
) -> float:
    """
    Calculates the default cost between two points in a path.
    The cost is the distance between the points plus the cost of the neighbor.

    Args:
        current (Point2): The current point.
        neighbor (Point2): The neighboring point.
        costs (dict[Point2, float], optional): A dictionary of costs for each point. Defaults to None.

    Returns:
        float: The cost between the current and neighboring points.
    """
    distance = current.distance_to(neighbor)
    return distance + (costs.get(neighbor, 0) if costs else 0)


class Djikstra:
    """
    Djikstra class for pathfinding in a segmented map.

    Args:
        map (SegmentedMap): The segmented map.
        cost_func (Callable[[Point2, Point2, Any], float], optional): The cost function for calculating the cost between two points. Defaults to default_cost_func.

    Attributes:
        map (SegmentedMap): The segmented map.
        cost_func (Callable[[Point2, Point2, Any], float]): The cost function for calculating the cost between two points.
        mapping (Dict[Point2, Passage]): A mapping of points to passages in the map.
    """

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
        """
        Returns a list of neighboring points from the given point.

        Args:
            point (Point2): The starting point.
            end (Point2): The end point.
            avoid (tuple[type[Passage]]): A tuple of passage types to avoid.

        Returns:
            list[Point2]: A list of neighboring points.
        """
        passage = self.mapping[point]
        neighbors = []

        for region in passage.connections:
            region = self.map.regions[region]

            if self._closest_region(end) is region:
                neighbors.append(end)
                continue

            for next_passage in region.passages:
                if (
                    next_passage is passage
                    or not next_passage.passable
                    or avoid
                    and isinstance(next_passage, avoid)
                ):
                    continue

                neighbors.append(next_passage.center())

        return neighbors

    def _closest_region(self, point: Point2) -> Region:
        """
        Finds the closest region to the given point.

        Args:
            point (Point2): The point to find the closest region for.

        Returns:
            Region: The closest region to the given point.
        """
        if region := self.map.region_at(point):
            return region

        directions = (Point2((0, 1)), Point2((0, -1)), Point2((-1, 0)), Point2((1, 0)))

        for distance in range(1, 10):
            for direction in directions:
                neighbor = point + direction * distance

                if region := self.map.region_at(neighbor):
                    return region

    def __call__(  # noqa: C901
        self,
        start: Point2,
        end: Point2,
        costs: Any = None,
        avoid: tuple[type[Passage]] = (),
    ) -> list[Point2]:
        """
        Calculate the shortest path from the start point to the end point.

        This method uses the Dijkstra's algorithm to find the shortest path
        between two points on the segmented map. The cost of moving from one
        point to another is determined by the cost function provided when
        the Djikstra object was created.

        Args:
            start (Point2): The starting point of the path.
            end (Point2): The ending point of the path.
            costs (Any, optional): An optional parameter that can be used to calculate the cost of moving from one point to another. Defaults to None.
            avoid (tuple[type[Passage]], optional): A tuple of Passage types that should be avoided when calculating the path. Defaults to an empty tuple.

        Returns:
            list[Point2]: The shortest path from the start point to the end point. If no path can be found, returns an empty list.
        """
        # find the closest region to the start and end points
        start_region = self._closest_region(start)
        end_region = self._closest_region(end)

        # if the start and end points are in the same region, return the path
        if start_region is end_region:
            return [start, end]

        # initialize the algorithm
        visited: set[Point2] = {start}
        previous: dict[Point2, Point2] = {}
        queue: PriorityQueue[tuple[float, Point2]] = PriorityQueue()
        distances: dict[Point2, float] = defaultdict(lambda: float("inf"))
        distances[start] = 0

        # for each passage in the start region, calculate the distance to the passage center and add it to the queue
        for passage in start_region.passages:
            if not passage.passable or avoid and isinstance(passage, avoid):
                continue

            distance = start.distance_to(passage.center())
            queue.put((distance, passage.center()))
            previous[passage.center()] = start
            distances[passage.center()] = distance

        # while the queue is not empty, get the next point from the queue
        while not queue.empty():
            _, current = queue.get()

            # if the current point is the end point, return the path
            if current is end:
                path = [current]
                while current in previous:
                    current = previous[current]
                    path.append(current)

                return path[::-1]

            # if the current point has already been visited, continue
            if current in visited:
                continue

            visited.add(current)

            # for each neighbor of the current point, calculate the distance to the neighbor and add it to the queue
            for neighbor in self._neighbors(current, end, avoid):
                if neighbor in visited:
                    continue

                distance = distances[current] + self.cost_func(current, neighbor, costs)

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current
                    queue.put((distance, neighbor))

        # if no path can be found, return an empty list
        return []
