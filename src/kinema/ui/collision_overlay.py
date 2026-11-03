"""Draw the robot's capsules and the obstacles, as the Motion Check measures them.

So a capsule that fits its link badly, or an obstacle whose box is not the
shape it looks, is seen before the check is trusted. A capsule and an obstacle
that touch on the current frame are drawn red.

Draw handlers never run in ``--background``. What to draw is worked out by
:func:`shape_lines`, which is plain geometry and is what the tests check.
"""

from __future__ import annotations

import math

import bpy
import numpy as np

from ..rig import builder
from ..rig import collision as geometry

CLEAR = (0.25, 0.8, 1.0, 0.6)
HIT = (1.0, 0.18, 0.12, 1.0)
LINE_WIDTH = 1.5
#: Segments in a capsule's rings and a sphere's circles.
SEGMENTS = 16
#: A floor is drawn as a square this many metres across, centred on its origin.
FLOOR_SIZE = 4.0

_handles: list = []
#: A capsule's outline, worked out once for every capsule: each point as
#: (across the axis, the other way across it, along it as a fraction of its
#: length, past its ends), the first two and last in units of the radius.
_outline: np.ndarray | None = None
#: (what the scene is at, the obstacles there and each one's outline).
_obstacles_drawn: tuple = ((), [], [])


def shape_lines(context) -> tuple[np.ndarray, np.ndarray]:
    """World-space segments: those clear of everything, and those touching something.

    Each an (N, 2, 3) array of segments' ends. Every visible rig's capsules,
    and every obstacle, as they stand on the current frame.
    """
    from ..ops import collision

    obstacles, obstacle_outlines = _obstacles_now(context)
    clear, hit = [], []
    touched = set()
    for rig in context.view_layer.objects:
        if not builder.is_kinema_rig(rig) or rig.mode == "EDIT" or not rig.visible_get():
            continue
        capsules = collision.capsules_of(rig)
        if not capsules:
            continue
        a, b, radii = collision.capsule_ends(rig, capsules)
        touching = np.zeros((len(capsules), len(obstacles)), dtype=bool)
        if obstacles:
            touching = geometry.clearances(a, b, radii, obstacles) < 0.0
        # As measured: round each capsule's ends in world space, at its world
        # radius, so a robot scaled unevenly is drawn as the check holds it.
        for index in range(len(capsules)):
            lines = _placed_outline(a[index], b[index], float(radii[index]))
            (hit if touching[index].any() else clear).append(lines)
        touched |= {obstacles[j].name for j in np.nonzero(touching.any(axis=0))[0]}
    for obstacle, lines in zip(obstacles, obstacle_outlines, strict=True):
        (hit if obstacle.name in touched else clear).append(lines)
    return _stacked(clear), _stacked(hit)


def _stacked(parts: list) -> np.ndarray:
    return np.concatenate(parts) if parts else np.zeros((0, 2, 3))


def _placed_outline(a: np.ndarray, b: np.ndarray, radius: float) -> np.ndarray:
    """What :func:`capsule_lines` draws for this capsule, as an (S, 2, 3) array.

    Placed from one outline worked out once, with numpy: building each from
    its rings and arcs in Python on every redraw took milliseconds.
    """
    global _outline
    if _outline is None:
        unit = np.array(capsule_lines((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0)).reshape(-1, 3)
        u, v = _square_to(np.array([0.0, 0.0, 1.0]))
        along = np.clip(unit[:, 2], 0.0, 1.0)
        _outline = np.column_stack([unit @ u, unit @ v, along, unit[:, 2] - along])
    axis = b - a
    length = float(np.linalg.norm(axis))
    axis = axis / length if length > 1e-9 else np.array([0.0, 0.0, 1.0])
    u, v = _square_to(axis)
    points = (
        a
        + np.outer(_outline[:, 2], b - a)
        + radius * (np.outer(_outline[:, 0], u) + np.outer(_outline[:, 1], v)
                    + np.outer(_outline[:, 3], axis))
    )
    return points.reshape(-1, 2, 3)


def _obstacles_now(context) -> tuple[list, list]:
    """The obstacles and their outlines, worked out again only once the scene has changed.

    A sphere's size reads its whole mesh: not something to do on every redraw.
    """
    global _obstacles_drawn
    from ..ops import collision

    scene = context.scene
    shapes = tuple(
        (obj.name, obj.kinema_obstacle_shape) for obj in collision.obstacle_objects(scene)
    )
    key = (collision._epoch, collision._generation, scene.name, scene.frame_current, shapes)
    if _obstacles_drawn[0] != key:
        found = collision.obstacles(context)
        outlines = [np.array(obstacle_lines(o)).reshape(-1, 2, 3) for o in found]
        _obstacles_drawn = (key, found, outlines)
    return _obstacles_drawn[1], _obstacles_drawn[2]


def forget() -> None:
    """A new file: its rigs and obstacles share names, not shapes, with the last one's."""
    global _obstacles_drawn
    _obstacles_drawn = ((), [], [])


def capsule_lines(a, b, radius: float) -> list:
    """A capsule's outline: a ring at each end, four lines along it, and its two caps."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    axis = b - a
    length = float(np.linalg.norm(axis))
    axis = axis / length if length > 1e-9 else np.array([0.0, 0.0, 1.0])
    u, v = _square_to(axis)
    lines = _circle(a, u * radius, v * radius) + _circle(b, u * radius, v * radius)
    for side in (u, v, -u, -v):
        lines.append((a + side * radius, b + side * radius))
    for side in (u, v):
        lines += _arc(b, side * radius, axis * radius) + _arc(a, side * radius, -axis * radius)
    return lines


def obstacle_lines(obstacle: geometry.Obstacle) -> list:
    matrix = np.asarray(obstacle.matrix, dtype=float)
    centre, rotation = matrix[:3, 3], matrix[:3, :3]
    if obstacle.kind == geometry.KIND_SPHERE:
        radius = float(obstacle.size[0])
        x, y, z = (rotation[:, index] * radius for index in range(3))
        return _circle(centre, x, y) + _circle(centre, y, z) + _circle(centre, z, x)
    if obstacle.kind == geometry.KIND_FLOOR:
        lines = []
        half = FLOOR_SIZE / 2.0
        x, y = rotation[:, 0], rotation[:, 1]
        for step in np.linspace(-half, half, 9):
            lines.append((centre + x * step - y * half, centre + x * step + y * half))
            lines.append((centre + y * step - x * half, centre + y * step + x * half))
        return lines
    corners = np.array([
        centre + rotation @ (np.array(signs) * obstacle.size)
        for signs in ((x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1))
    ])
    return [
        (corners[i], corners[j])
        for i in range(8)
        for j in range(i + 1, 8)
        if bin(i ^ j).count("1") == 1
    ]


def _square_to(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    return u, np.cross(axis, u)


def _circle(centre, x, y) -> list:
    points = [
        centre + x * math.cos(t) + y * math.sin(t)
        for t in np.linspace(0.0, 2.0 * math.pi, SEGMENTS + 1)
    ]
    return list(zip(points, points[1:], strict=False))


def _arc(centre, side, out) -> list:
    """A half-circle from ``side`` over ``out`` to ``-side``: one cap's profile."""
    points = [
        centre + side * math.cos(t) + out * math.sin(t)
        for t in np.linspace(0.0, math.pi, SEGMENTS // 2 + 1)
    ]
    return list(zip(points, points[1:], strict=False))


def _draw() -> None:
    context = bpy.context
    scene = context.scene
    props = getattr(scene, "kinema", None)
    if props is None or not props.show_collision:
        return
    overlay = getattr(context.space_data, "overlay", None)
    if overlay is not None and not overlay.show_overlays:
        return
    clear, hit = shape_lines(context)
    if not len(clear) and not len(hit):
        return

    import gpu
    from gpu_extras.batch import batch_for_shader

    shader = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
    region = context.region
    depth, blend = gpu.state.depth_test_get(), gpu.state.blend_get()
    gpu.state.blend_set("ALPHA")
    # Through the meshes: a capsule sits round its link, mostly inside it.
    gpu.state.depth_test_set("NONE")
    try:
        shader.uniform_float("viewportSize", (region.width, region.height))
        shader.uniform_float("lineWidth", LINE_WIDTH * context.preferences.system.ui_scale)
        for lines, color in ((clear, CLEAR), (hit, HIT)):
            if not len(lines):
                continue
            points = np.ascontiguousarray(lines.reshape(-1, 3), dtype=np.float32)
            shader.uniform_float("color", color)
            batch_for_shader(shader, "LINES", {"pos": points}).draw(shader)
    finally:
        gpu.state.depth_test_set(depth)
        gpu.state.blend_set(blend)


def register_draw() -> None:
    if not _handles:
        _handles.append(
            bpy.types.SpaceView3D.draw_handler_add(_draw, (), "WINDOW", "POST_VIEW")
        )


def unregister_draw() -> None:
    while _handles:
        bpy.types.SpaceView3D.draw_handler_remove(_handles.pop(), "WINDOW")
