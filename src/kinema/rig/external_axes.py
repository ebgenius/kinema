"""External axes: the part that needs no Blender.

An external axis is a joint the robot description does not have. It might be a rail the
robot rides on, a turntable under its base, a positioner holding the workpiece, or a
spindle on the tool. Kinema adds it to the rig as one more joint bone, so it gets a
slider, a hold pin, a velocity check, keys and bake like any imported joint.

Three mounts:

* ``BEFORE`` -- the robot rides the axis: a linear track, a rotary base.
* ``AFTER`` -- the axis rides the tool: a spindle, a tool slide.
* ``EXTERNAL`` -- the axis stands on its own: a positioner or turntable holding the part.

Two placements, both a location plus roll/pitch/yaw about fixed X, Y and Z (URDF's own
convention), describe where it goes:

* **base** -- where the axis sits, relative to the current robot base (BEFORE), the
  robot's own base (EXTERNAL) or the tool frame (AFTER). This is the axis's joint frame
  at rest, and ``direction`` picks which of its axes the joint moves along.
* **offset** -- where what the axis carries sits, relative to the axis's moving frame:
  the current robot base on a BEFORE axis, the TCP on an AFTER axis. An EXTERNAL axis
  carries nothing until something is attached to it.

"The current robot base" is the robot together with any BEFORE axes already under it: a
new one goes in at the bottom, standing on the :func:`footing` of the lowest, so a
second track carries the first. For a BEFORE axis, ``base @ offset`` is where that base
ends up. The default of both at zero leaves everything where it stands; anything else
moves it, which is :func:`robot_shift`.

Placeholder geometry is generated here as plain vertex and face lists, so nothing extra
ships in the extension and the shapes can be tested without Blender. It is drawn in the
axis bone's own space: +Y along the axis, the joint's zero position at the origin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .kinematics import make_transform

MOUNTS = ("BEFORE", "AFTER", "EXTERNAL")
KINDS = ("LINEAR", "ROTARY")

#: The axis the joint moves along, in its base frame.
DIRECTIONS = {
    "X": (1.0, 0.0, 0.0), "-X": (-1.0, 0.0, 0.0),
    "Y": (0.0, 1.0, 0.0), "-Y": (0.0, -1.0, 0.0),
    "Z": (0.0, 0.0, 1.0), "-Z": (0.0, 0.0, -1.0),
}


@dataclass
class AxisSpec:
    """Everything the Add External Axis dialog asks for."""

    name: str
    mount: str = "BEFORE"
    kind: str = "LINEAR"
    direction: str = "X"
    #: Metres for LINEAR, radians for ROTARY. Ignored when ``continuous``.
    lower: float = -1.0
    upper: float = 1.0
    #: ROTARY only: no end stops, like a spindle or a positioner that turns forever.
    continuous: bool = False
    #: Top speed, m/s or rad/s. Zero or None means no velocity limit.
    velocity: float | None = None
    base_location: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    offset_location: tuple[float, float, float] = (0.0, 0.0, 0.0)
    offset_rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: Positioned by hand: IK leaves it where it is put.
    hold: bool = True
    placeholder: bool = True
    #: Placeholder size in metres: the rail's width, the turntable's radius.
    size: float = 0.3

    @property
    def joint_type(self) -> str:
        if self.kind == "LINEAR":
            return "prismatic"
        return "continuous" if self.continuous else "revolute"

    @property
    def axis(self) -> np.ndarray:
        return np.array(DIRECTIONS[self.direction], dtype=float)

    @property
    def limits(self) -> tuple[float, float] | None:
        if self.kind == "ROTARY" and self.continuous:
            return None
        return (float(self.lower), float(self.upper))

    @property
    def speed(self) -> float | None:
        return float(self.velocity) if self.velocity and self.velocity > 0.0 else None

    @property
    def base(self) -> np.ndarray:
        return make_transform(self.base_location, self.base_rotation)

    @property
    def offset(self) -> np.ndarray:
        return make_transform(self.offset_location, self.offset_rotation)

    def problems(self) -> list[str]:
        """What is wrong with this spec, in words; empty when it can be built."""
        found = []
        if not self.name.strip():
            found.append("give the axis a name")
        if self.mount not in MOUNTS:
            found.append(f"unknown mount {self.mount!r}")
        if self.kind not in KINDS:
            found.append(f"unknown kind {self.kind!r}")
        if self.direction not in DIRECTIONS:
            found.append(f"unknown direction {self.direction!r}")
        if self.limits is not None and not self.lower < self.upper:
            found.append("the lower limit must be below the upper one")
        if self.size <= 0.0:
            found.append("the placeholder size must be positive")
        return found


# --------------------------------------------------------------------------------------
# presets: a small catalogue of the axes robots are usually installed with
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Preset:
    label: str
    description: str
    values: dict = field(default_factory=dict)


PRESETS: dict[str, Preset] = {
    "LINEAR_TRACK": Preset(
        "Linear Track", "A rail the robot rides along, under its base",
        dict(name="track", mount="BEFORE", kind="LINEAR", direction="X",
             lower=-1.5, upper=1.5, velocity=1.0),
    ),
    "ROTARY_BASE": Preset(
        "Rotary Base", "A turntable under the robot, turning the whole arm",
        dict(name="rotary_base", mount="BEFORE", kind="ROTARY", direction="Z",
             lower=-math.pi, upper=math.pi, continuous=False, velocity=math.radians(90.0)),
    ),
    "POSITIONER": Preset(
        "Positioner", "A turntable beside the robot, turning the workpiece",
        dict(name="positioner", mount="EXTERNAL", kind="ROTARY", direction="Z",
             continuous=True, velocity=math.radians(120.0), base_location=(1.0, 0.0, 0.0)),
    ),
    "TOOL_SPINDLE": Preset(
        "Tool Spindle", "A spindle on the tool, turning about the tool's Z",
        dict(name="spindle", mount="AFTER", kind="ROTARY", direction="Z",
             continuous=True, velocity=2.0 * math.pi),
    ),
    "TOOL_SLIDE": Preset(
        "Tool Slide", "A slide on the tool, pushing along the tool's Z",
        dict(name="slide", mount="AFTER", kind="LINEAR", direction="Z",
             lower=0.0, upper=0.1, velocity=0.1),
    ),
}


def from_preset(key: str, **overrides) -> AxisSpec:
    """An AxisSpec filled from a preset, with any field overridden."""
    values = dict(PRESETS[key].values)
    values.update(overrides)
    return AxisSpec(**values)


def default_size(mount: str, reach: float) -> float:
    """A placeholder size in proportion to the robot, whose extent is ``reach`` metres.

    A quarter of the robot for something it stands on or beside, a few percent for
    something on its tool -- so a spindle on a small arm and a track under a large
    one both come out looking like parts of the robot they belong to.
    """
    reach = float(reach) if reach and reach > 0.0 else 1.0
    if mount == "AFTER":
        return float(np.clip(reach * 0.06, 0.02, 0.3))
    return float(np.clip(reach * 0.25, 0.05, 1.0))


# --------------------------------------------------------------------------------------
# where things go
# --------------------------------------------------------------------------------------
def base_frame(spec: AxisSpec, reference: np.ndarray) -> np.ndarray:
    """The axis's joint frame at rest: ``reference`` then the spec's base placement.

    ``reference`` is the robot base for BEFORE and EXTERNAL, and the tool frame for
    AFTER, both in armature space.
    """
    return np.asarray(reference, dtype=float) @ spec.base


def robot_shift(spec: AxisSpec, robot_base: np.ndarray) -> np.ndarray:
    """How far a BEFORE axis moves the robot, as a transform applied on the left.

    The robot base ends up at ``base @ offset`` on the axis's carriage. Everything the
    axis carries -- the robot, its TCP and IK target, any BEFORE axis already under it --
    moves by this one rigid transform. Identity when base and offset cancel, which the
    defaults do.
    """
    robot_base = np.asarray(robot_base, dtype=float)
    return base_frame(spec, robot_base) @ spec.offset @ np.linalg.inv(robot_base)


def bone_placement(frame: np.ndarray, axis) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(head, y_direction, z_reference)`` for the axis bone in armature space.

    The bone's +Y is the joint axis, as for every Kinema joint. Its roll points +Z along
    whichever base-frame axis is furthest from the joint axis -- the frame's own Z unless
    the joint moves along Z, then its X -- so a rail along X keeps "up" up.
    """
    frame = np.asarray(frame, dtype=float)
    rotation = frame[:3, :3]
    axis = np.asarray(axis, dtype=float)
    y_direction = rotation @ axis
    local_z = np.array([1.0, 0.0, 0.0]) if abs(axis[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
    return frame[:3, 3].copy(), y_direction, rotation @ local_z


def bone_matrix(frame: np.ndarray, axis) -> np.ndarray:
    """The axis bone's rest matrix, as Blender builds it from :func:`bone_placement`.

    +Y along the joint axis, +Z as close to the reference as a right angle to Y allows,
    and X completing a right-handed frame -- which is what ``align_roll`` settles on.
    The preview draws with it before any bone exists.
    """
    head, y_direction, z_reference = bone_placement(frame, axis)
    y = y_direction / np.linalg.norm(y_direction)
    x = np.cross(y, z_reference)
    x /= np.linalg.norm(x)
    matrix = np.eye(4)
    matrix[:3, 0], matrix[:3, 1], matrix[:3, 2] = x, y, np.cross(x, y)
    matrix[:3, 3] = head
    return matrix


def footing(spec: AxisSpec) -> np.ndarray:
    """Where the next axis down stands, in this axis's frame: just under its fixed part.

    A straight drop along the frame's -Z, as deep as the placeholder's rail or base
    reaches -- so a second track goes in under the first rather than through it. Taken
    from the shape even with Placeholder off, so stacking does not depend on a tickbox.
    """
    static, _ = placeholder(spec)
    in_frame = bone_matrix(np.eye(4), spec.axis)[:3, :3] @ np.array(static[0], dtype=float).T
    drop = np.eye(4)
    drop[2, 3] = min(0.0, float(in_frame[2].min()))
    return drop


# --------------------------------------------------------------------------------------
# procedural placeholders, in the axis bone's own space
# --------------------------------------------------------------------------------------
Mesh = tuple[list[tuple[float, float, float]], list[tuple[int, ...]]]


def box(x: tuple[float, float], y: tuple[float, float], z: tuple[float, float]) -> Mesh:
    """An axis-aligned box: 8 vertices, 6 outward-facing quads."""
    vertices = [(x[i], y[j], z[k]) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    faces = [
        (0, 1, 3, 2),  # -X
        (4, 6, 7, 5),  # +X
        (0, 4, 5, 1),  # -Y
        (2, 3, 7, 6),  # +Y
        (0, 2, 6, 4),  # -Z
        (1, 5, 7, 3),  # +Z
    ]
    return vertices, faces


def cylinder(radius: float, y: tuple[float, float], segments: int = 32) -> Mesh:
    """A closed cylinder around the Y axis, from ``y[0]`` to ``y[1]``."""
    vertices = []
    for height in y:
        for index in range(segments):
            angle = 2.0 * math.pi * index / segments
            vertices.append((radius * math.cos(angle), height, radius * math.sin(angle)))
    # Angle runs from +X towards +Z, so a loop in that order faces -Y: the end at
    # y[0] as it is, the end at y[1] reversed, and each side quad wound to face out.
    faces: list[tuple[int, ...]] = [
        (index, segments + index, segments + (index + 1) % segments, (index + 1) % segments)
        for index in range(segments)
    ]
    faces.append(tuple(range(segments)))
    faces.append(tuple(range(2 * segments - 1, segments - 1, -1)))
    return vertices, faces


def merge(*meshes: Mesh) -> Mesh:
    vertices: list = []
    faces: list = []
    for mesh_vertices, mesh_faces in meshes:
        start = len(vertices)
        vertices += mesh_vertices
        faces += [tuple(start + i for i in face) for face in mesh_faces]
    return vertices, faces


def placeholder(spec: AxisSpec) -> tuple[Mesh, Mesh]:
    """``(static, moving)`` meshes for the axis, sized by ``spec.size``.

    A linear axis gets a rail spanning its whole travel and a carriage on it; a rotary
    one a base and a turning plate with a pointer, so its angle reads at a glance. The
    moving part's top sits at the joint's zero, which is where the offset places what the
    axis carries.
    """
    s = float(spec.size)
    if spec.kind == "LINEAR":
        lower, upper = spec.limits or (-s, s)
        rail = box((-0.25 * s, 0.25 * s), (lower - 0.3 * s, upper + 0.3 * s),
                   (-0.45 * s, -0.2 * s))
        carriage = box((-0.5 * s, 0.5 * s), (-0.35 * s, 0.35 * s), (-0.2 * s, 0.0))
        return rail, carriage
    base = cylinder(0.5 * s, (-0.35 * s, -0.08 * s))
    plate = cylinder(0.45 * s, (-0.08 * s, 0.0))
    # A tab out past the plate's rim, so the angle reads without looking at bones.
    pointer = box((0.3 * s, 0.55 * s), (-0.08 * s, 0.0), (-0.04 * s, 0.04 * s))
    return base, merge(plate, pointer)


# --------------------------------------------------------------------------------------
# the preview drawn while the dialog is open: plain arrays, drawn by ui/axis_preview.py
# --------------------------------------------------------------------------------------
def triangles(mesh: Mesh) -> np.ndarray:
    """The mesh as a ``(n, 3, 3)`` array of triangles, each face fanned from its first."""
    vertices, faces = mesh
    points = np.array(vertices, dtype=float)
    fans = [(face[0], face[i], face[i + 1]) for face in faces for i in range(1, len(face) - 1)]
    return points[np.array(fans, dtype=int)]


def edges(mesh: Mesh) -> np.ndarray:
    """The mesh's edges, each once, as a ``(n, 2, 3)`` array of segments."""
    vertices, faces = mesh
    pairs = {
        tuple(sorted((face[i], face[(i + 1) % len(face)])))
        for face in faces for i in range(len(face))
    }
    points = np.array(vertices, dtype=float)
    return points[np.array(sorted(pairs), dtype=int)]


def _moved(spec: AxisSpec, value: float) -> np.ndarray:
    """The joint at ``value``, in the bone's space: along or about the bone's +Y."""
    if spec.kind == "LINEAR":
        return make_transform((0.0, value, 0.0), (0.0, 0.0, 0.0))
    return make_transform((0.0, 0.0, 0.0), (0.0, value, 0.0))


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def travel(spec: AxisSpec, segments: int = 48) -> tuple[np.ndarray, list]:
    """How far the axis goes, in the bone's space: ``(line segments, labels)``.

    A linear axis shows its carriage outlined at both end stops, joined by a line down
    the middle; a rotary one an arc at the plate's rim from stop to stop -- a full
    circle when it is continuous -- with the pointer outlined at each stop. Labels are
    ``(point, text)`` at the stops.
    """
    s = float(spec.size)
    static, moving = placeholder(spec)
    outline = edges(moving)
    lines: list[np.ndarray] = []
    labels: list = []
    if spec.kind == "LINEAR":
        lower, upper = spec.limits or (-s, s)
        for value in (lower, upper):
            lines += list(_apply(_moved(spec, value), outline.reshape(-1, 3)).reshape(-1, 2, 3))
            labels.append((np.array([0.0, value, 0.0]), f"{value:+.2f} m"))
        lines.append(np.array([[0.0, lower, 0.0], [0.0, upper, 0.0]]))
        return np.array(lines), labels

    radius = 0.55 * s
    limits = spec.limits
    start, end = limits if limits is not None else (0.0, 2.0 * math.pi)
    angles = np.linspace(start, end, segments + 1)
    # Turning about +Y by a carries +X to (cos a, 0, -sin a).
    arc = np.stack([radius * np.cos(angles), np.zeros_like(angles), -radius * np.sin(angles)], 1)
    lines += list(np.stack([arc[:-1], arc[1:]], 1))
    if limits is not None:
        for value in limits:
            lines += list(_apply(_moved(spec, value), outline.reshape(-1, 3)).reshape(-1, 2, 3))
            point = np.array([radius * math.cos(value), 0.0, -radius * math.sin(value)])
            labels.append((point, f"{math.degrees(value):+.0f}°"))
    return np.array(lines), labels


def cluster_triangles(points, tris, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """A coarser copy of a triangle mesh: vertices snapped together on a grid of ``cell``.

    Every vertex in one cell becomes their average, and triangles that collapse -- two
    corners in one cell -- or come out twice are dropped. A robot of a million
    triangles comes down to a few thousand that still read as the robot, which is all a
    ghost of where it is about to go needs.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    if not len(points) or not len(tris) or cell <= 0.0:
        return points.copy(), tris.copy()
    keys = np.floor(points / cell).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    merged = np.zeros((len(counts), 3))
    np.add.at(merged, inverse, points)
    merged /= counts[:, None]
    remapped = inverse[tris]
    alive = (
        (remapped[:, 0] != remapped[:, 1])
        & (remapped[:, 1] != remapped[:, 2])
        & (remapped[:, 0] != remapped[:, 2])
    )
    remapped = remapped[alive]
    _, first = np.unique(np.sort(remapped, axis=1), axis=0, return_index=True)
    return merged, remapped[np.sort(first)]
