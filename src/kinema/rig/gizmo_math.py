"""Where a viewport handle puts a control: sliding and turning a pose. Blender-free.

The IK control's handles work along the *tool's* axes, and no bone channel can say that:
the control rotates as a quaternion and slides in its rest frame. So a drag is applied to
the control's whole matrix and Blender splits the result back into channels. Everything
here is plain arithmetic on 4x4 arrays, so ``uv run pytest`` checks it without Blender.
"""

from __future__ import annotations

import numpy as np

#: Below this length an axis has no direction to slide or turn along.
DEGENERATE = 1e-12


def _unit(axis) -> np.ndarray:
    vector = np.asarray(axis, dtype=float)[:3]
    length = float(np.linalg.norm(vector))
    if length < DEGENERATE:
        raise ValueError("a handle axis needs a direction")
    return vector / length


def rotation(axis, angle: float) -> np.ndarray:
    """The right-handed 3x3 rotation by ``angle`` radians about ``axis``."""
    x, y, z = _unit(axis)
    c, s = np.cos(angle), np.sin(angle)
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )


def slide(matrix, axis, distance: float) -> np.ndarray:
    """``matrix`` moved ``distance`` along ``axis``, its orientation untouched."""
    out = np.array(matrix, dtype=float, copy=True)
    out[:3, 3] += _unit(axis) * distance
    return out


def turn(matrix, axis, angle: float) -> np.ndarray:
    """``matrix`` turned by ``angle`` about ``axis`` through its own origin.

    About its own origin because that origin is the tool point: turning the tool
    has to leave the point where it is, or a rotation drag would also move it.
    """
    out = np.array(matrix, dtype=float, copy=True)
    out[:3, :3] = rotation(axis, angle) @ out[:3, :3]
    return out


def axis_basis(frame, index: int) -> np.ndarray:
    """``frame`` with its axes cycled so that axis ``index`` becomes Z.

    Blender's dial turns about its own Z and its arrow points along it, so a handle
    for the tool's X or Y needs that axis moved into Z. Cycling the columns rather
    than swapping two keeps the frame right-handed, so a positive angle still means
    the same turn.
    """
    order = ((1, 2, 0), (2, 0, 1), (0, 1, 2))[index]
    source = np.asarray(frame, dtype=float)
    out = source.copy()
    out[:, :3] = source[:, list(order)]
    return out
