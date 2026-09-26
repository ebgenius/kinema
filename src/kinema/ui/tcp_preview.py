"""Show where the TCP would go while Edit TCP is open.

Drawn, not built, as the external-axis preview is: no object, no bone, no undo step,
nothing solved. Two frames as axis triads -- X red, Y green, Z blue -- the new one
bright and the current one faint, and a line from the parent link's origin to the
new TCP, so a tool picked off the wrong joint shows at a glance.

What to draw is plain matrix maths (:func:`shapes`), which is what the tests check.
Draw handlers never run in ``--background``.
"""

from __future__ import annotations

import bpy
import numpy as np

from ..rig import builder
from .axis_preview import _apply, _np4, _overlays_shown, _redraw

LINE_WIDTH = 2.5
#: The triad's arms, as a fraction of the robot's extent.
TRIAD_SCALE = 0.08
AXIS_COLORS = ((1.0, 0.25, 0.25), (0.35, 0.9, 0.35), (0.3, 0.55, 1.0))
LINK_COLOR = (1.0, 0.85, 0.3)
#: Alpha of the new frame and of the current one.
NEW_ALPHA, CURRENT_ALPHA = 1.0, 0.35

_state: dict = {}
_handles: list = []


def triad(frame, length: float, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """``(points, colors)``: three segments from ``frame``'s origin along its axes."""
    frame = np.asarray(frame, dtype=float)
    origin = frame[:3, 3]
    points, colors = [], []
    for axis, rgb in enumerate(AXIS_COLORS):
        points += [origin, origin + frame[:3, axis] * length]
        colors += [(*rgb, alpha)] * 2
    return np.array(points), np.array(colors)


def shapes(new_frame, current_frame, link_frame, length: float) -> tuple[np.ndarray, np.ndarray]:
    """Everything drawn, in armature space: the new triad, the current, the link line.

    ``new_frame`` is None when the fields cannot place a TCP; the current one alone
    is drawn then. ``current_frame`` is None when the rig has no TCP yet.
    """
    parts = []
    if current_frame is not None:
        parts.append(triad(current_frame, length, CURRENT_ALPHA))
    if new_frame is not None:
        parts.append(triad(new_frame, length, NEW_ALPHA))
        if link_frame is not None:
            line = np.array([np.asarray(link_frame)[:3, 3], np.asarray(new_frame)[:3, 3]])
            parts.append((line, np.array([(*LINK_COLOR, NEW_ALPHA)] * 2)))
    if not parts:
        return np.empty((0, 3)), np.empty((0, 4))
    return np.concatenate([p for p, _ in parts]), np.concatenate([c for _, c in parts])


def _current_tool(rig) -> np.ndarray | None:
    """The TCP's tool frame where it stands now, in armature space."""
    tcp = rig.pose.bones.get(rig.get(builder.PROP_TCP_BONE) or builder.TCP_BONE)
    if tcp is None:
        return None
    return _np4(tcp.matrix @ builder.BONE_TO_TOOL)


def _length(rig) -> float:
    heads = np.array([b.head_local for b in rig.data.bones], dtype=float)
    extent = float(np.linalg.norm(heads.max(axis=0) - heads.min(axis=0))) if len(heads) else 1.0
    return max(extent, 0.1) * TRIAD_SCALE


# --------------------------------------------------------------------------
# the dialog's side
# --------------------------------------------------------------------------
def start(operator, rig) -> None:
    """Begin previewing for the dialog ``operator`` is about to open on ``rig``."""
    stop()
    if rig is None:
        return
    _state.update(operator=operator, rig=rig.name, length=_length(rig), key=None, drawn=None)
    space = bpy.types.SpaceView3D
    _handles.append(space.draw_handler_add(_draw, (), "WINDOW", "POST_VIEW"))


def show(operator, new_frame, link_frame) -> None:
    """The TCP the fields place now -- None if they cannot -- and its link's frame."""
    if _state.get("operator") is not operator:
        return
    rig = bpy.data.objects.get(_state["rig"])
    if rig is None:
        return
    new = None if new_frame is None else _np4(new_frame)
    link = None if link_frame is None else _np4(link_frame)
    current = _current_tool(rig)
    key = tuple(
        None if m is None else np.round(m, 9).tobytes() for m in (new, link, current)
    )
    if key == _state.get("key"):
        return
    _state["key"] = key
    _state["drawn"] = shapes(new, current, link, _state["length"])
    _redraw()


def stop() -> None:
    """Take the preview away. Safe to call when there is none."""
    while _handles:
        bpy.types.SpaceView3D.draw_handler_remove(_handles.pop(), "WINDOW")
    if _state:
        _state.clear()
        _redraw()


def active() -> bool:
    return bool(_handles)


def _live():
    """``(rig, (points, colors))`` to draw, or None -- and gone once the dialog is.

    A dialog that closed without calling ``cancel`` leaves its operator freed, and
    reading it raises ReferenceError; the handler is then removed on the next tick,
    not from inside its own call.
    """
    operator = _state.get("operator")
    if operator is None:
        return None
    try:
        operator.source  # noqa: B018 -- reading it is the test
    except ReferenceError:
        if not bpy.app.timers.is_registered(stop):
            bpy.app.timers.register(stop, first_interval=0.0)
        return None
    rig = bpy.data.objects.get(_state.get("rig", ""))
    drawn = _state.get("drawn")
    if rig is None or drawn is None or not len(drawn[0]):
        return None
    return rig, drawn


def _draw() -> None:
    context = bpy.context
    live = _live()
    if live is None or not _overlays_shown(context):
        return
    rig, (points, colors) = live

    import gpu
    from gpu_extras.batch import batch_for_shader

    shader = gpu.shader.from_builtin("POLYLINE_SMOOTH_COLOR")
    positions = _apply(_np4(rig.matrix_world), points).astype(np.float32)
    batch = batch_for_shader(
        shader, "LINES", {"pos": positions, "color": colors.astype(np.float32)}
    )
    region = context.region
    # Put back whatever was set: the next draw handler inherits what this leaves.
    depth, blend = gpu.state.depth_test_get(), gpu.state.blend_get()
    gpu.state.blend_set("ALPHA")
    # In front of the robot, as the rig's own TCP marker is.
    gpu.state.depth_test_set("NONE")
    try:
        shader.uniform_float("viewportSize", (region.width, region.height))
        shader.uniform_float("lineWidth", LINE_WIDTH * context.preferences.system.ui_scale)
        batch.draw(shader)
    finally:
        gpu.state.depth_test_set(depth)
        gpu.state.blend_set(blend)
