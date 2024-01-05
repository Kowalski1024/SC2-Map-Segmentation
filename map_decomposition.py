from collections import defaultdict
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def show_map(_ndarray, colors=('white', 'black')):
    cmap = ListedColormap(colors)
    plt.matshow(_ndarray, cmap=cmap)
    plt.show()


def octile_offset(distance):
    octile = np.fromfunction(lambda i, j: 1.414 * np.minimum(i, j) + np.absolute(i - j), (distance + 1, distance + 1))
    for x, y in zip(*np.where(octile.astype(int) == distance)):
        yield from {(x, y), (x, -y), (-x, y), (-x, -y)}


def get_depth_map(map_array):
    depth_map = np.zeros_like(map_array, dtype=int)
    walls = set(zip(*np.where(map_array == 0)))
    depth_list = defaultdict(list)

    current_depth = 0
    for ix, iy in zip(*np.where(map_array == 1)):
        current_depth -= 1
        while True:
            if depth_map[ix][iy] != 0:
                break

            for offset_x, offset_y in octile_offset(current_depth):
                if (ix + offset_x, iy + offset_y) in walls:
                    depth = math.floor(math.log(current_depth, 2) + 1)
                    depth_map[ix][iy] = depth
                    depth_list[depth].append((ix, iy))
                    break
            else:
                current_depth += 1

    return depth_map, depth_list


def build_zone_map(depth_map, depth_list):
    zone_map = np.zeros(depth_map.shape, dtype=np.int32)
    gate_cluster_map = np.zeros(depth_map.shape, dtype=np.int32)

    current_water_level = np.max(depth_map)
    while current_water_level >= 0:
        for ix, iy in sorted(depth_list[current_water_level]):
            neighbors = get_labeled_neighbors(ix, iy, zone_map)

            if len(neighbors) > 0:
                if len(set(neighbors)) > 1:
                    gate_cluster_map[ix][iy] = 1
                zone_map[ix][iy] = min(neighbors)
            else:
                zone_map[ix][iy] = get_new_label(zone_map)

        current_water_level -= 1

    return zone_map, gate_cluster_map


def get_labeled_neighbors(x, y, zone_map):
    neighbors = []

    for dy, dx in {(0, 0), (0, 1), (0, -1), (0, 2), (0, -2), (0, 3), (0, -3), (0, -4), (0, 4), (0, -5), (0, 5), (1, 0),
                   (-1, 0), (-1, 1), (1, 1), (-1, -1), (1, -1), (-1, -2), (1, 2), (1, -2), (-1, 2), (1, -3), (-1, 3),
                   (1, 3), (-1, -3), (1, -4), (-1, 4), (1, 4), (-1, -4), (-2, 0), (2, 0), (2, -1), (-2, -1), (-2, 1),
                   (2, 1), (2, -2), (-2, 2), (-2, -2), (2, 2), (2, 3), (-2, 3), (2, -3), (-2, -3), (2, 4), (-2, 4),
                   (2, -4), (-2, -4), (-3, 0), (3, 0), (3, 1), (-3, -1), (3, -1), (-3, 1), (-3, 2), (3, 2), (3, -2),
                   (-3, -2), (-3, 3), (-3, -3), (3, 3), (3, -3), (3, -4), (-3, 4), (3, 4), (-3, -4), (4, 0), (-4, 0),
                   (4, -1), (4, 1), (-4, -1), (-4, 1), (4, -2), (-4, -2), (4, 2), (-4, 2), (-4, 3), (-4, -3), (4, -3),
                   (4, 3), (-5, 0), (5, 0)}:
        ny, nx = y + dy, x + dx
        if zone_map.shape[0] > nx >= 0 and zone_map.shape[1] > ny >= 0 and zone_map[nx][ny] > 0:
            neighbors.append(zone_map[nx][ny])

    return neighbors


def get_new_label(zone_map):
    return np.max(zone_map) + 1


def decompose():
    placement_grid = np.load('placement_grid.npy')
    depth_map, depth_list = get_depth_map(placement_grid)
    zone_map, gate_map = build_zone_map(depth_map, depth_list)


if __name__ == '__main__':
    decompose()
