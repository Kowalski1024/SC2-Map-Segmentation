# Starcraft 2 Map Segmentation
It's plugin for Python SC2 API [BurnySc2/python-sc2](https://github.com/BurnySc2/python-sc2). 

**SC2 Map Segmentation** is fully automated map segmentation for Starcraft 2 maps. the goal was to create something similar to [SC2MapAnalysis](https://github.com/spudde123/SC2MapAnalysis) but focused on segmentation of map into smaller pieces (what SC2MapAnalysis doesn't have) and finding passages between them, like choke points, ramps, etc. It has also implemented basic pathfinding algorithm - Dijkstra.

It's still in development, has much less features than SC2MapAnalysis.
### Example of segmentation
![SiteDeltaAIE segmentation](data/map_pool/SiteDeltaAIE.png)
Top view of the map: https://liquipedia.net/commons/images/7/77/Site_Delta.jpg

Other examples from _Sc2 AI Arena 2023 Season 3_ map pool are in [data/map_pool](data/map_pool) folder.

### How it works
The algorithm operates by identifying key locations (referred to as 'seed points') on the map, such as expansions, ramps, etc. It then scans the surrounding area for the nearest unpathable/unbuildable tiles.

Tiles that create a large angle between the previous tile and the seed point are filtered out. Points that are significantly distant from the previous point are identified as potential choke points. These choke points are then marked on the map.

Next, the algorithm performs a flood fill operation from the seed point, avoiding crossing through any choke points.

Example of depth scan, white point is seed point, greed points are scanned points, red lines are choke points:
![Depth scan image](data/depth_scan.png)

Finally, the map undergoes a cleanup process. This includes steps such as removing small regions and merging regions that are in close proximity to each other.

## Installation
1. Clone this repository
2. Install requirements
3. Copy `mapsegmentation` folder to your project

## Usage
The example of usage is in [examples/reaper_pathfinding.py](examples\reaper_pathfinding.py). It's simple bot which finds path for reaper to enemy base using Dijkstra algorithm.

The segmented map can be pickled and saved to file.

There are three main dataclasses to reprezent segmented map and can be find in [mapsegmentation/dataclasses](mapsegmentation\dataclasses) folder:
- `SegmentedMap` - contains all segments and passages between them
- `Region` - represents one segment of map
- `Passage` - represents one passage between regions


## Limitations / TODO / Contribution
- There are still issues with segmentation, like the segmented regions aren't simetrical on both sides.
- It doesn't find cliff passages in places where e.g. reaper need to jump twice to get to the other side.
- There aren't much features implemented yet, like in MapAnalysis.

If you want to contribute, feel free to create pull request. I will be happy for any help, because I don't have much time to work on this project.