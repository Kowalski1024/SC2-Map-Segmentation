import numpy as np

from sc2.ids.unit_typeid import UnitTypeId
from sc2.unit import Unit


def change_destructable_status(  # noqa: C901
    grid: np.ndarray, unit: Unit, status: int
) -> None:
    """
    Source:
    https://github.com/spudde123/SC2MapAnalysis/blob/master/MapAnalyzer/utils.py

    Set destructable positions to status, modifies the grid in place

    Args:
        grid (np.ndarray): grid to modify
        unit (Unit): sc2 unit
        status (int): status to set
    """
    type_id = unit.type_id
    pos = unit.position
    name = unit.name

    # this is checked with name because the id of the small mineral destructables
    # has changed over patches and may cause problems
    if name == "MineralField450":
        x = int(pos[0]) - 1
        y = int(pos[1])
        grid[x : (x + 2), y] = status
    elif type_id in destructable_2x2:
        w = 2
        h = 2
        x = int(pos[0] - w / 2)
        y = int(pos[1] - h / 2)
        grid[x : (x + w), y : (y + h)] = status
    elif type_id in destructable_2x4:
        w = 2
        h = 4
        x = int(pos[0] - w / 2)
        y = int(pos[1] - h / 2)
        grid[x : (x + w), y : (y + h)] = status
    elif type_id in destructable_2x6:
        w = 2
        h = 6
        x = int(pos[0] - w / 2)
        y = int(pos[1] - h / 2)
        grid[x : (x + w), y : (y + h)] = status
    elif type_id in destructable_4x2:
        w = 4
        h = 2
        x = int(pos[0] - w / 2)
        y = int(pos[1] - h / 2)
        grid[x : (x + w), y : (y + h)] = status
    elif type_id in destructable_4x4:
        w = 4
        h = 4
        x = int(pos[0] - w / 2)
        y = int(pos[1] - h / 2)
        grid[x : (x + w), (y + 1) : (y + h - 1)] = status
        grid[(x + 1) : (x + w - 1), y : (y + h)] = status
    elif type_id in destructable_6x2:
        w = 6
        h = 2
        x = int(pos[0] - w / 2)
        y = int(pos[1] - h / 2)
        grid[x : (x + w), y : (y + h)] = status
    elif type_id in destructable_6x6:
        w = 6
        h = 6
        x = int(pos[0] - w / 2)
        y = int(pos[1] - h / 2)
        grid[x : (x + w), (y + 1) : (y + h - 1)] = status
        grid[(x + 1) : (x + w - 1), y : (y + h)] = status
    elif type_id in destructable_12x4:
        w = 12
        h = 4
        x = int(pos[0] - w / 2)
        y = int(pos[1] - h / 2)
        grid[x : (x + w), y : (y + h)] = status
    elif type_id in destructable_4x12:
        w = 4
        h = 12
        x = int(pos[0] - w / 2)
        y = int(pos[1] - h / 2)
        grid[x : (x + w), y : (y + h)] = status
    elif type_id in destructable_BLUR:
        x_ref = int(pos[0] - 5)
        y_pos = int(pos[1])
        grid[(x_ref + 6) : (x_ref + 6 + 2), y_pos + 4] = status
        grid[(x_ref + 5) : (x_ref + 5 + 4), y_pos + 3] = status
        grid[(x_ref + 4) : (x_ref + 4 + 6), y_pos + 2] = status
        grid[(x_ref + 3) : (x_ref + 3 + 7), y_pos + 1] = status
        grid[(x_ref + 2) : (x_ref + 2 + 7), y_pos] = status
        grid[(x_ref + 1) : (x_ref + 1 + 7), y_pos - 1] = status
        grid[(x_ref + 0) : (x_ref + 0 + 7), y_pos - 2] = status
        grid[(x_ref + 0) : (x_ref + 0 + 6), y_pos - 3] = status
        grid[(x_ref + 1) : (x_ref + 1 + 4), y_pos - 4] = status
        grid[(x_ref + 2) : (x_ref + 2 + 2), y_pos - 5] = status

    elif type_id in destructable_ULBR:
        x_ref = int(pos[0] - 5)
        y_pos = int(pos[1])
        grid[(x_ref + 6) : (x_ref + 6 + 2), y_pos - 5] = status
        grid[(x_ref + 5) : (x_ref + 5 + 4), y_pos - 4] = status
        grid[(x_ref + 4) : (x_ref + 4 + 6), y_pos - 3] = status
        grid[(x_ref + 3) : (x_ref + 3 + 7), y_pos - 2] = status
        grid[(x_ref + 2) : (x_ref + 2 + 7), y_pos - 1] = status
        grid[(x_ref + 1) : (x_ref + 1 + 7), y_pos] = status
        grid[(x_ref + 0) : (x_ref + 0 + 7), y_pos + 1] = status
        grid[(x_ref + 0) : (x_ref + 0 + 6), y_pos + 2] = status
        grid[(x_ref + 1) : (x_ref + 1 + 4), y_pos + 3] = status
        grid[(x_ref + 2) : (x_ref + 2 + 2), y_pos + 4] = status


"""
https://github.com/DrInfy/sharpy-sc2/blob/develop/sharpy/managers/unit_value.py
"""
destructable_2x2 = {UnitTypeId.ROCKS2X2NONCONJOINED, UnitTypeId.DEBRIS2X2NONCONJOINED}

destructable_4x4 = {
    UnitTypeId.DESTRUCTIBLECITYDEBRIS4X4,
    UnitTypeId.DESTRUCTIBLEDEBRIS4X4,
    UnitTypeId.DESTRUCTIBLEICE4X4,
    UnitTypeId.DESTRUCTIBLEROCK4X4,
    UnitTypeId.DESTRUCTIBLEROCKEX14X4,
}

destructable_4x2 = {
    UnitTypeId.DESTRUCTIBLECITYDEBRIS2X4HORIZONTAL,
    UnitTypeId.DESTRUCTIBLEICE2X4HORIZONTAL,
    UnitTypeId.DESTRUCTIBLEROCK2X4HORIZONTAL,
    UnitTypeId.DESTRUCTIBLEROCKEX12X4HORIZONTAL,
}

destructable_2x4 = {
    UnitTypeId.DESTRUCTIBLECITYDEBRIS2X4VERTICAL,
    UnitTypeId.DESTRUCTIBLEICE2X4VERTICAL,
    UnitTypeId.DESTRUCTIBLEROCK2X4VERTICAL,
    UnitTypeId.DESTRUCTIBLEROCKEX12X4VERTICAL,
}

destructable_6x2 = {
    UnitTypeId.DESTRUCTIBLECITYDEBRIS2X6HORIZONTAL,
    UnitTypeId.DESTRUCTIBLEICE2X6HORIZONTAL,
    UnitTypeId.DESTRUCTIBLEROCK2X6HORIZONTAL,
    UnitTypeId.DESTRUCTIBLEROCKEX12X6HORIZONTAL,
}

destructable_2x6 = {
    UnitTypeId.DESTRUCTIBLECITYDEBRIS2X6VERTICAL,
    UnitTypeId.DESTRUCTIBLEICE2X6VERTICAL,
    UnitTypeId.DESTRUCTIBLEROCK2X6VERTICAL,
    UnitTypeId.DESTRUCTIBLEROCKEX12X6VERTICAL,
}

destructable_4x12 = {
    UnitTypeId.DESTRUCTIBLEROCKEX1VERTICALHUGE,
    UnitTypeId.DESTRUCTIBLEICEVERTICALHUGE,
}

destructable_12x4 = {
    UnitTypeId.DESTRUCTIBLEROCKEX1HORIZONTALHUGE,
    UnitTypeId.DESTRUCTIBLEICEHORIZONTALHUGE,
}

destructable_6x6 = {
    UnitTypeId.DESTRUCTIBLECITYDEBRIS6X6,
    UnitTypeId.DESTRUCTIBLEDEBRIS6X6,
    UnitTypeId.DESTRUCTIBLEICE6X6,
    UnitTypeId.DESTRUCTIBLEROCK6X6,
    UnitTypeId.DESTRUCTIBLEROCKEX16X6,
}

destructable_BLUR = {
    UnitTypeId.DESTRUCTIBLECITYDEBRISHUGEDIAGONALBLUR,
    UnitTypeId.DESTRUCTIBLEDEBRISRAMPDIAGONALHUGEBLUR,
    UnitTypeId.DESTRUCTIBLEICEDIAGONALHUGEBLUR,
    UnitTypeId.DESTRUCTIBLEROCKEX1DIAGONALHUGEBLUR,
    UnitTypeId.DESTRUCTIBLERAMPDIAGONALHUGEBLUR,
}

destructable_ULBR = {
    UnitTypeId.DESTRUCTIBLECITYDEBRISHUGEDIAGONALULBR,
    UnitTypeId.DESTRUCTIBLEDEBRISRAMPDIAGONALHUGEULBR,
    UnitTypeId.DESTRUCTIBLEICEDIAGONALHUGEULBR,
    UnitTypeId.DESTRUCTIBLEROCKEX1DIAGONALHUGEULBR,
    UnitTypeId.DESTRUCTIBLERAMPDIAGONALHUGEULBR,
}
