"""Show where an external axis will go while Add External Axis is open.

Drawn in the viewport, not built: no object, no bone, no undo step. The dialog's
fields change what is drawn as they are edited, and closing it -- OK or Cancel --
takes it away.

* The axis itself: its rail and carriage, or base and plate, as the placeholder
  would be made, when Placeholder is ticked.
* Its travel: the carriage outlined at both end stops, or an arc from stop to stop,
  with the limits written beside them -- ticked or not, so it always shows where
  the axis is and how far it goes.
* The robot where it will stand, for an axis under it that moves it: a ghost of
  its meshes and the axes it already rides, coarsened when the dialog opens and
  moved as one rigid block. Nothing is solved.

What to draw is plain matrix maths (:func:`shapes`, and ``rig/external_axes.py``);
the tests check that. Draw handlers never run in ``--background``.
"""

from __future__ import annotations

import bpy
import numpy as np

from ..rig import external_axes as ext

GHOST_COLOR = (0.35, 0.7, 1.0, 0.22)
FILL_COLOR = (0.85, 0.45, 0.1, 0.4)
TRAVEL_COLOR = (1.0, 0.62, 0.2, 0.95)
LINE_WIDTH = 2.0
LABEL_SIZE = 12
#: The ghost's grid: this many cells across the robot's extent.
GHOST_CELLS = 60

#: The open dialog's preview: its operator, rig, ghost, and what it last showed.
_state: dict = {}
_handles: list = []


# --------------------------------------------------------------------------
# what to draw
# --------------------------------------------------------------------------
def _np4(matrix) -> np.ndarray:
    return np.array([[matrix[r][c] for c in range(4)] for r in range(4)])


def _apply(matrix: np.ndarray, points) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def ghost_mesh(rig, objects, depsgraph) -> tuple[np.ndarray, np.ndarray]:
    """``(points, triangles)`` of ``objects`` as they stand now, in armature space.

    Coarsened by :func:`ext.cluster_triangles` to about :data:`GHOST_CELLS` cells
    across, so a detailed robot costs the viewport a few thousand triangles.
    """
    to_armature = _np4(rig.matrix_world.inverted_safe())
    points, tris, offset = [], [], 0
    for obj in objects:
        if obj.type != "MESH" or not obj.visible_get():
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            count, faces = len(mesh.vertices), len(mesh.loop_triangles)
            coords = np.empty(count * 3, dtype=np.float32)
            corners = np.empty(faces * 3, dtype=np.int32)
            mesh.vertices.foreach_get("co", coords)
            mesh.loop_triangles.foreach_get("vertices", corners)
        finally:
            evaluated.to_mesh_clear()
        if not count or not faces:
            continue
        points.append(_apply(to_armature @ _np4(obj.matrix_world), coords))
        tris.append(corners.reshape(-1, 3).astype(np.int64) + offset)
        offset += count
    if not points:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)
    points, tris = np.concatenate(points), np.concatenate(tris)
    extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    return ext.cluster_triangles(points, tris, extent / GHOST_CELLS)


def shapes(spec: ext.AxisSpec, placed) -> dict:
    """What the preview draws for ``spec``, in armature space.

    ``placed`` is ``(parent, frame, shift)`` from ``ops.external_axes.placement``.
    Returns the placeholder's triangles (``fill``, empty when unticked), the travel
    as line segments, the limit labels as ``(point, text)``, and the shift the ghost
    moves by -- None when it does not move.
    """
    _, frame, shift = placed
    bone = ext.bone_matrix(frame, spec.axis)
    fill = np.empty((0, 3))
    if spec.placeholder:
        static, moving = ext.placeholder(spec)
        fill = _apply(bone, np.concatenate([ext.triangles(static), ext.triangles(moving)]))
    lines, labels = ext.travel(spec)
    moves = spec.mount == "BEFORE" and not np.allclose(shift, np.eye(4), atol=1e-6)
    return {
        "fill": fill,
        "lines": _apply(bone, lines),
        "labels": [(_apply(bone, point)[0], text) for point, text in labels],
        "shift": np.asarray(shift, dtype=float) if moves else None,
    }


# --------------------------------------------------------------------------
# the dialog's side
# --------------------------------------------------------------------------
def start(operator, context, rig, ghost_objects) -> None:
    """Begin previewing for the dialog ``operator`` is about to open on ``rig``."""
    stop()
    if rig is None:
        return
    points, tris = ghost_mesh(rig, ghost_objects, context.evaluated_depsgraph_get())
    _state.update(operator=operator, rig=rig.name, ghost=(points, tris), shapes=None, key=None)
    space = bpy.types.SpaceView3D
    _handles.append(space.draw_handler_add(_draw_shapes, (), "WINDOW", "POST_VIEW"))
    _handles.append(space.draw_handler_add(_draw_labels, (), "WINDOW", "POST_PIXEL"))


def show(operator, spec: ext.AxisSpec, placed) -> None:
    """What the dialog shows now; ``placed`` is None when it cannot be placed."""
    if _state.get("operator") is not operator:
        return
    key = (
        repr(spec),
        None if placed is None else (np.round(placed[1], 9).tobytes(),
                                     np.round(placed[2], 9).tobytes()),
    )
    if key == _state.get("key"):
        return
    _state["key"] = key
    _state["shapes"] = None if placed is None else shapes(spec, placed)
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


def _redraw() -> None:
    manager = getattr(bpy.context, "window_manager", None)
    for window in manager.windows if manager is not None else ():
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _live():
    """``(rig, shapes)`` to draw, or None -- and gone for good once the dialog is.

    Blender frees a dialog's operator when it closes. If it closed without calling
    ``cancel`` the handlers are still here, and reading the operator is what tells:
    it raises ReferenceError. They are then removed on the next tick, not from inside
    their own call.
    """
    operator = _state.get("operator")
    if operator is None:
        return None
    try:
        operator.mount  # noqa: B018 -- reading it is the test
    except ReferenceError:
        if not bpy.app.timers.is_registered(stop):
            bpy.app.timers.register(stop, first_interval=0.0)
        return None
    rig = bpy.data.objects.get(_state.get("rig", ""))
    found = _state.get("shapes")
    if rig is None or found is None:
        return None
    return rig, found


def _overlays_shown(context) -> bool:
    overlay = getattr(context.space_data, "overlay", None)
    return overlay is None or overlay.show_overlays


def _draw_shapes() -> None:
    context = bpy.context
    live = _live()
    if live is None or not _overlays_shown(context):
        return
    rig, found = live

    import gpu
    from gpu_extras.batch import batch_for_shader

    world = _np4(rig.matrix_world)
    fill_shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    line_shader = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
    region = context.region
    # Put back whatever was set, not the defaults: Blender does not reset GPU
    # state between draw handlers, so the next one inherits what this leaves.
    depth, blend, mask = (
        gpu.state.depth_test_get(), gpu.state.blend_get(), gpu.state.depth_mask_get()
    )
    gpu.state.blend_set("ALPHA")
    gpu.state.depth_mask_set(False)
    try:
        # Solid parts sit among the scene, hidden where the robot is in front.
        gpu.state.depth_test_set("LESS_EQUAL")
        points, tris = _state["ghost"]
        if found["shift"] is not None and len(tris):
            positions = _apply(world @ found["shift"], points).astype(np.float32)
            batch = batch_for_shader(
                fill_shader, "TRIS", {"pos": positions}, indices=tris.tolist()
            )
            fill_shader.uniform_float("color", GHOST_COLOR)
            batch.draw(fill_shader)
        if len(found["fill"]):
            positions = _apply(world, found["fill"]).astype(np.float32)
            batch = batch_for_shader(fill_shader, "TRIS", {"pos": positions})
            fill_shader.uniform_float("color", FILL_COLOR)
            batch.draw(fill_shader)
        # The travel in front of everything, as the rig's own bones are.
        gpu.state.depth_test_set("NONE")
        if len(found["lines"]):
            positions = _apply(world, found["lines"]).astype(np.float32)
            batch = batch_for_shader(line_shader, "LINES", {"pos": positions})
            line_shader.uniform_float("viewportSize", (region.width, region.height))
            line_shader.uniform_float(
                "lineWidth", LINE_WIDTH * context.preferences.system.ui_scale
            )
            line_shader.uniform_float("color", TRAVEL_COLOR)
            batch.draw(line_shader)
    finally:
        gpu.state.depth_test_set(depth)
        gpu.state.blend_set(blend)
        gpu.state.depth_mask_set(mask)


def _draw_labels() -> None:
    context = bpy.context
    live = _live()
    if live is None or not _overlays_shown(context):
        return
    rig, found = live
    if not found["labels"]:
        return

    import blf
    from bpy_extras.view3d_utils import location_3d_to_region_2d
    from mathutils import Vector

    world = _np4(rig.matrix_world)
    region, view = context.region, context.region_data
    scale = context.preferences.system.ui_scale
    font = 0
    blf.size(font, LABEL_SIZE * scale)
    blf.color(font, *TRAVEL_COLOR)
    blf.enable(font, blf.SHADOW)
    blf.shadow(font, 3, 0.0, 0.0, 0.0, 0.9)
    blf.shadow_offset(font, 1, -1)
    try:
        for point, text in found["labels"]:
            anchor = location_3d_to_region_2d(region, view, Vector(_apply(world, point)[0]))
            if anchor is None:
                continue
            blf.position(font, anchor.x + 8 * scale, anchor.y + 6 * scale, 0)
            blf.draw(font, text)
    finally:
        blf.disable(font, blf.SHADOW)
