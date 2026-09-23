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

* **base** -- where the axis sits, relative to the robot base (BEFORE, EXTERNAL) or to
  the tool frame (AFTER). This is the axis's joint frame at rest, and ``direction`` picks
  which of its axes the joint moves along.
* **offset** -- where what the axis carries sits, relative to the axis's moving frame:
  the robot base on a BEFORE axis, the TCP on an AFTER axis. An EXTERNAL axis carries
  nothing until something is attached to it.

For a BEFORE axis, ``base @ offset`` is where the robot base ends up. The default of both
at zero leaves the robot where it stands; anything else moves it, which is
:func:`robot_shift`.

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
