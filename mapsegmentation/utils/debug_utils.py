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
) -> None:
    """
    Draws a point on the game map with a specified color and optional text.

    Parameters:
        game_info (GameInfo): The game information.
        client (Client): The client object for debugging.
        point (Point2): The point to draw on the map.
        height_offset (int): The offset to adjust the height of the point.
        color (tuple[int, int, int], optional): The color of the point. Defaults to WHITE.
        text (str, optional): The text to display near the point. Defaults to "".
    """
    height = get_terrain_z_height(game_info, point)

    p = Point3((point.x + 0.5, point.y + 0.5, height + height_offset))
    client.debug_box2_out(p, color=color)

    if text:
        p = Point3((point.x + 0.5, point.y + 0.5, height + height_offset + 0.5))
        client.debug_text_world(text, p, color=color, size=12)
