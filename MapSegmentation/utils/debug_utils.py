from sc2.client import Client
from sc2.game_info import GameInfo
from sc2.position import Point2, Point3

from .misc_utils import get_terrain_z_height

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)


def draw_point(
    game_info: GameInfo,
    client: Client,
    point: Point2,
    height_offset: int,
    color: tuple[int, int, int] = WHITE,
    text: str = "",
):
    """
    Draws a point on the map

    Args:
        game_info (GameInfo): The game info of the map
        client (Client): The client to draw the point with
        point (Point2): The point to draw
        height_offset (int): The height offset of the point
        color (tuple[int, int, int], optional): The color of the point
        text (str, optional): The text to draw
    """
    height = get_terrain_z_height(game_info, point)

    p = Point3((point.x + 0.5, point.y + 0.5, height + height_offset))
    client.debug_box2_out(p, color=color)

    if text:
        p = Point3((point.x + 0.5, point.y + 0.5, height + height_offset + 0.5))
        client.debug_text_world(text, p, color=color, size=12)
