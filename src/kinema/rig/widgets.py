"""Custom bone shapes for Kinema rigs.

Aligning a bone's Y axis to its joint axis is what makes each bone a true 1-DoF
control, but it costs the usual visual cue: the bone no longer points at its
child link, so a default octahedral bone tells an animator nothing useful. A
custom shape puts the meaning back -- a dial around the axis it actually turns
about, an arrow along the axis it actually slides along.

Shapes are drawn in bone space: +Y runs head to tail, and with
``use_custom_shape_bone_size`` the whole widget scales with bone length, so one
unit-sized widget serves every robot.

Widgets are wireframe (vertices and edges, no faces) so they read as controls
rather than geometry, and are deliberately left unlinked from any collection --
Blender happily uses an unlinked object as a custom shape, and it keeps the
outliner free of a dozen helper objects. ``use_fake_user`` keeps them alive
across save/reload.
"""

from __future__ import annotations

import math

import bpy

PREFIX = "WGT-kinema"


def _make_widget(name: str, vertices, edges) -> bpy.types.Object:
    """Create (or reuse) an unlinked wireframe object usable as a bone shape."""
    full_name = f"{PREFIX}-{name}"
    existing = bpy.data.objects.get(full_name)
    if existing is not None:
        return existing

    mesh = bpy.data.meshes.new(full_name)
    mesh.from_pydata([tuple(v) for v in vertices], [tuple(e) for e in edges], [])
    mesh.update()

    obj = bpy.data.objects.new(full_name, mesh)
    # Never linked to a collection: it is a widget, not scene content.
    obj.use_fake_user = True
    return obj


def _ring(radius: float, offset_y: float, segments: int = 24):
    """A circle in the XZ plane -- i.e. around the bone's Y axis."""
    vertices = [
        (
            radius * math.cos(2 * math.pi * i / segments),
            offset_y,
            radius * math.sin(2 * math.pi * i / segments),
        )
        for i in range(segments)
    ]
    edges = [(i, (i + 1) % segments) for i in range(segments)]
    return vertices, edges


def _revolute_widget() -> bpy.types.Object:
    """A dial around the rotation axis, with a tick showing zero."""
    vertices, edges = _ring(0.35, 0.5)
    base = len(vertices)
    # Tick mark at the dial's zero, so the joint's current angle is readable.
    vertices += [(0.35, 0.5, 0.0), (0.55, 0.5, 0.0)]
    edges += [(base, base + 1)]
    # Short stub along the axis itself, so the axis direction stays visible.
    axis_base = len(vertices)
    vertices += [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    edges += [(axis_base, axis_base + 1)]
    return _make_widget("revolute", vertices, edges)


def _prismatic_widget() -> bpy.types.Object:
    """A double-headed arrow along the slide axis."""
    shaft = [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    edges = [(0, 1)]
    vertices = list(shaft)
    for tip_y, direction in ((1.0, -1.0), (0.0, 1.0)):
        base = len(vertices)
        vertices += [
            (0.12, tip_y + 0.18 * direction, 0.0),
            (-0.12, tip_y + 0.18 * direction, 0.0),
            (0.0, tip_y + 0.18 * direction, 0.12),
            (0.0, tip_y + 0.18 * direction, -0.12),
        ]
        tip = 0 if tip_y == 0.0 else 1
        edges += [(tip, base), (tip, base + 1), (tip, base + 2), (tip, base + 3)]
    return _make_widget("prismatic", vertices, edges)


def _root_widget() -> bpy.types.Object:
    """A wide ground ring with cross-hairs, sitting at the robot base."""
    vertices, edges = _ring(1.0, 0.0, segments=32)
    base = len(vertices)
    vertices += [(-1.2, 0.0, 0.0), (1.2, 0.0, 0.0), (0.0, 0.0, -1.2), (0.0, 0.0, 1.2)]
    edges += [(base, base + 1), (base + 2, base + 3)]
    return _make_widget("root", vertices, edges)


def _tcp_widget() -> bpy.types.Object:
    """An axis tripod: the tool frame's X, Y and Z."""
    vertices = [
        (0.0, 0.0, 0.0),
        (0.6, 0.0, 0.0),   # X
        (0.0, 1.0, 0.0),   # Y (along the bone)
        (0.0, 0.0, 0.6),   # Z
    ]
    edges = [(0, 1), (0, 2), (0, 3)]
    # A small square at the tip marks the tool point itself.
    base = len(vertices)
    s = 0.1
    vertices += [(-s, 1.0, -s), (s, 1.0, -s), (s, 1.0, s), (-s, 1.0, s)]
    edges += [(base, base + 1), (base + 1, base + 2), (base + 2, base + 3), (base + 3, base)]
    return _make_widget("tcp", vertices, edges)


def _ik_target_widget() -> bpy.types.Object:
    """A cube outline: the movable IK goal (populated in M5)."""
    s = 0.35
    corners = [
        (x * s, y * s + 0.5, z * s)
        for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)
    ]
    edges = [
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
    ]
    return _make_widget("ik-target", corners, edges)


def ensure_widgets() -> dict[str, bpy.types.Object]:
    """Create the widget objects if absent and return them by role."""
    return {
        "revolute": _revolute_widget(),
        "prismatic": _prismatic_widget(),
        "root": _root_widget(),
        "tcp": _tcp_widget(),
        "ik_target": _ik_target_widget(),
    }
