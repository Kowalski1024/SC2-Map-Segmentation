from collections import defaultdict
from typing import Any, Callable, Optional
from queue import PriorityQueue


from sc2.position import Point2

from SC2MapSegmentation.dataclasses import SegmentedMap, Passage


class Djikstra:
    def __init__(self, map: SegmentedMap, cost_func: Callable = None):
        self.map = map
        self.cost_func = cost_func

        self.mapping = {passage.center(): passage for passage in map.passages}

    def _neighbors(self, point: Point2, end: Point2) -> list[Point2]:
        passage = self.mapping[point]
        neighbors = []

        for region in passage.connections:
            region = self.map.regions[region]

            if end.rounded in region.tiles:
                neighbors.append(end)
                continue

            for next_passage in region.passages:
                if next_passage is passage:
                    continue
                
                neighbors.append(next_passage.center())

        return neighbors
        
    def __call__(self, start: Point2, end: Point2, costs: dict[int, float] = None) -> list[Point2]:
        start_region = self.map.region_at(start)
        end_region = self.map.region_at(end)

        if start_region is end_region:
            return [start, end]
        
        print("start", start, "end", end)
        
        visited: set[Point2] = {start}
        previous: dict[Point2, Point2] = {}
        queue: PriorityQueue[tuple[float, Point2]] = PriorityQueue()
        distances: dict[Point2, float] = defaultdict(lambda: float("inf"))

        distances[start] = 0
        
        for passage in start_region.passages:
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

            for neighbor in self._neighbors(current, end):
                if neighbor in visited:
                    continue

                distance = distances[current] + current.distance_to(neighbor)

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current
                    queue.put((distance, neighbor))

        print("no path found")
        return []

        

        


    