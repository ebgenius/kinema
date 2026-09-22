"""Draw a joint that is over its velocity limit in red, in the 3D viewport.

The joint's own widget -- the dial or arrow the rig already draws -- is traced
over itself in red, thicker, with the joint's name and how far over it is
beside it. So the warning sits on the thing that is moving too fast, not in a
panel the user has scrolled past.

Drawn, not recoloured. Tinting the bone would write to the file on every frame
of playback, push depsgraph updates while it did, and overwrite any colours
the user had given their bones.

Draw handlers never run in ``--background``, so the drawing itself is only seen
in a window. What to draw is worked out by :func:`warning_lines`, which is
plain matrix maths and is what the tests check.
"""

from __future__ import annotations

import math

import bpy
from mathutils import Matrix, Vector

from ..rig import builder

COLOR = (1.0, 0.18, 0.12, 1.0)
LINE_WIDTH = 3.0
LABEL_SIZE = 12

#: Handles returned by draw_handler_add, kept to remove them again.
_handles: list = []


def _shape_edges(pose_bone) -> list[tuple[Vector, Vector]]:
    """The bone's custom shape as edges, in the shape's own space.

    Falls back to a dial like the revolute widget when the shape is missing
    or not a mesh -- a user may have swapped it -- so the warning still has
    something to draw.
    """
    shape = pose_bone.custom_shape
    if shape is not None and shape.type == "MESH" and len(shape.data.edges):
        points = [vertex.co.copy() for vertex in shape.data.vertices]
        return [(points[a], points[b]) for a, b in (e.vertices for e in shape.data.edges)]

    segments = 24
    ring = [
        Vector((
            0.35 * math.cos(2 * math.pi * i / segments),
            0.5,
            0.35 * math.sin(2 * math.pi * i / segments),
        ))
        for i in range(segments)
    ]
    return [(ring[i], ring[(i + 1) % segments]) for i in range(segments)]


def _shape_matrix(rig, pose_bone) -> Matrix:
    """Shape space to world space, placed the way Blender places the widget."""
    source = pose_bone.custom_shape_transform or pose_bone
    length = pose_bone.bone.length if pose_bone.use_custom_shape_bone_size else 1.0
    scale = Vector(pose_bone.custom_shape_scale_xyz) * length
    return (
        rig.matrix_world
        @ source.matrix
        @ Matrix.Translation(pose_bone.custom_shape_translation)
        @ pose_bone.custom_shape_rotation_euler.to_matrix().to_4x4()
        @ Matrix.Diagonal((*scale, 1.0))
    )


def warning_lines(rig, over) -> list[tuple[Vector, Vector]]:
    """World-space line segments tracing the widgets of the joints in ``over``."""
    lines = []
    for reading in over:
        pose_bone = rig.pose.bones.get(reading.joint)
        if pose_bone is None:
            continue
        matrix = _shape_matrix(rig, pose_bone)
        lines += [(matrix @ a, matrix @ b) for a, b in _shape_edges(pose_bone)]
    return lines


def label_anchor(rig, pose_bone) -> Vector:
    """Where a joint's label hangs: the middle of the bone, which is the dial's centre."""
    return rig.matrix_world @ ((pose_bone.head + pose_bone.tail) * 0.5)


def flagged(context) -> list[tuple[bpy.types.Object, list]]:
    """Every visible rig in the view layer with a joint over its limit now."""
    from ..ops import velocity

    found = []
    for obj in context.view_layer.objects:
        if not builder.is_kinema_rig(obj) or obj.mode == "EDIT":
            continue
        if not obj.visible_get():
            continue
        over = velocity.warnings(obj, context.scene)
        if over:
            found.append((obj, over))
    return found


def _overlays_shown(context) -> bool:
    space = context.space_data
    overlay = getattr(space, "overlay", None)
    return overlay is None or overlay.show_overlays


def _draw_lines() -> None:
    context = bpy.context
    if not _overlays_shown(context):
        return
    points = [
        point
        for rig, over in flagged(context)
        for segment in warning_lines(rig, over)
        for point in segment
    ]
    if not points:
        return

    import gpu
    from gpu_extras.batch import batch_for_shader

    shader = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
    batch = batch_for_shader(shader, "LINES", {"pos": points})
    region = context.region
    gpu.state.blend_set("ALPHA")
    # In front of the robot, as the rig's own bones are.
    gpu.state.depth_test_set("NONE")
    shader.uniform_float("viewportSize", (region.width, region.height))
    shader.uniform_float("lineWidth", LINE_WIDTH * context.preferences.system.ui_scale)
    shader.uniform_float("color", COLOR)
    batch.draw(shader)
    gpu.state.blend_set("NONE")


def _draw_labels() -> None:
    context = bpy.context
    if not _overlays_shown(context):
        return
    found = flagged(context)
    if not found:
        return

    import blf
    from bpy_extras.view3d_utils import location_3d_to_region_2d

    region, view = context.region, context.region_data
    scale = context.preferences.system.ui_scale
    font = 0
    blf.size(font, LABEL_SIZE * scale)
    blf.color(font, *COLOR)
    blf.enable(font, blf.SHADOW)
    blf.shadow(font, 3, 0.0, 0.0, 0.0, 0.9)
    blf.shadow_offset(font, 1, -1)
    try:
        for rig, over in found:
            for reading in over:
                pose_bone = rig.pose.bones.get(reading.joint)
                if pose_bone is None:
                    continue
                point = location_3d_to_region_2d(
                    region, view, label_anchor(rig, pose_bone)
                )
                if point is None:
                    continue
                blf.position(font, point.x + 10 * scale, point.y + 6 * scale, 0)
                blf.draw(font, f"{reading.joint}  {reading.ratio:.0%}")
    finally:
        blf.disable(font, blf.SHADOW)


def register_draw() -> None:
    if _handles:
        return
    space = bpy.types.SpaceView3D
    _handles.append(space.draw_handler_add(_draw_lines, (), "WINDOW", "POST_VIEW"))
    _handles.append(space.draw_handler_add(_draw_labels, (), "WINDOW", "POST_PIXEL"))


def unregister_draw() -> None:
    while _handles:
        bpy.types.SpaceView3D.draw_handler_remove(_handles.pop(), "WINDOW")
