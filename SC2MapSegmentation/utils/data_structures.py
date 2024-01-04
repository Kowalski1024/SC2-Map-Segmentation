from typing import Hashable, Iterable, NamedTuple, TypeVar

T = TypeVar("T", bound=Hashable)


class Point(NamedTuple):
    x: int
    y: int


class FindUnion:
    """
    Find-Union data structure

    Args:
        items (Iterable[T]): hashable items to be stored in the data structure
    """

    def __init__(self, items: Iterable[T]) -> None:
        self.items = list(items)
        self.parents = {item: item for item in items}

    def add(self, item: T) -> None:
        """
        Adds an item to the data structure

        Args:
            item (T): item to add
        """
        if item not in self.parents:
            self.items.append(item)
            self.parents[item] = item

    def find(self, item: T) -> T:
        """
        Finds the parent of an item

        Args:
            item (T): item to find the parent of

        Returns:
            T: parent of the item
        """
        if self.parents[item] != item:
            self.parents[item] = self.find(self.parents[item])
        return self.parents[item]

    def union(self, item1: T, item2: T) -> None:
        """
        Unions two items

        Args:
            item1 (T): first item to union
            item2 (T): second item to union
        """
        self.parents[self.find(item1)] = self.find(item2)

    def groups(self) -> list[set[T]]:
        """
        Returns the groups of items in the data structure

        Returns:
            list[set[T]]: groups of items in the data structure
        """
        groups = {}
        for item in self.items:
            parent = self.find(item)
            if parent in groups:
                groups[parent].add(item)
            else:
                groups[parent] = {item}
        return list(groups.values())
