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
    """The tool frame, drawn so its three axes can be told apart.

    Shapes are in *bone* space, and the tool frame rides the bone permuted (see
    ``builder.BONE_TO_TOOL``): the tool's Z is the bone's +Y, its X the bone's
    +Z, its Y the bone's +X. So the long arrowed arm running head to tail is
    the approach direction, which is the one an operator actually looks for.

    The previous version gave X and Z the same length and no arrowhead, which
    left the marker's orientation unreadable -- three identical sticks tell you
    where the tool is but not which way it faces.
    """
    vertices: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    edges: list[tuple[int, int]] = []

    # Tool Z: long, along the bone, with an arrowhead. The approach direction.
    tip = len(vertices)
    vertices.append((0.0, 1.0, 0.0))
    edges.append((0, tip))
    head_y, spread = 0.78, 0.09
    for dx, dz in ((spread, 0.0), (-spread, 0.0), (0.0, spread), (0.0, -spread)):
        barb = len(vertices)
        vertices.append((dx, head_y, dz))
        edges.append((tip, barb))

    # Tool X: medium, on the bone's +Z. Tool Y: short, on the bone's +X.
    for length, axis in ((0.55, (0.0, 0.0, 1.0)), (0.3, (1.0, 0.0, 0.0))):
        end = len(vertices)
        vertices.append(tuple(component * length for component in axis))
        edges.append((0, end))

    # A small square about the tool point, so the origin reads as a frame and
    # not as the end of yet another stick.
    base = len(vertices)
    s = 0.08
    vertices += [(-s, 0.0, -s), (s, 0.0, -s), (s, 0.0, s), (-s, 0.0, s)]
    edges += [(base, base + 1), (base + 1, base + 2), (base + 2, base + 3), (base + 3, base)]

    # New name, not a redraw of "tcp": _make_widget reuses any object it finds
    # by name, so a .blend saved with the old shape would otherwise keep it.
    return _make_widget("tcp-axes", vertices, edges)


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
