"""Kinema's public API, for scripts and other add-ons (#102).

Blender extensions can't declare a dependency on another extension, so an add-on
that works with Kinema finds it at runtime, as the README's *Scripting Kinema*
section shows, and calls what is here. Everything else in Kinema
is private and changes without notice; this module doesn't.

``VERSION`` is ``(major, minor)``. The minor number grows when something is
added, the major number when something changes or goes. A test records every
name and signature here, and fails on any change the version doesn't say.

What it takes and gives:

* Blender objects for rigs: the armature a robot import makes.
* NumPy arrays for joint values, in the rig's joint order, in radians and metres.
* NumPy 4x4 arrays for frames, in world space, in metres.
* Small frozen dataclasses for the rest. No solver types.

Nothing here pushes an undo step: a script or an operator that calls it pushes
its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VERSION = (1, 0)

__all__ = [
    "VERSION",
    "Error",
    "Joint",
    "Tool",
    "is_rig",
    "rigs",
    "joints",
    "joint_values",
    "set_joint_values",
    "tools",
    "tool_frame",
    "Solution",
    "solve",
    "Capsule",
    "capsules",
    "Obstacle",
    "obstacles",
    "add_obstacle",
    "remove_obstacle",
    "Clearance",
    "clearance",
    "Reach",
    "reachable",
]

#: How far past a limit a joint may stand and still be within it: float noise.
_LIMIT_TOLERANCE = 1e-6


class Error(Exception):
    """What Kinema can't do as asked: the message says why."""


@dataclass(frozen=True)
class Joint:
    """One moving joint of a rig, with the limits the rig holds.

    The rig's limits, not the description's: Load Joint Limits and external axes
    write them to the rig. The Motion Check and Optimize read the range, speed and
    acceleration there; jerk and effort are given for reference, and nothing
    checks them.
    """

    #: The joint's name in the description, as ``joint3``.
    name: str
    #: The pose bone that drives it.
    bone: str
    #: ``"revolute"``, ``"continuous"`` or ``"prismatic"``.
    type: str
    #: Its range, in radians or metres; None for a joint without one.
    lower: float | None
    upper: float | None
    #: Its top speed (rad/s, m/s), acceleration (/s²) and jerk (/s³), and the
    #: most torque or force its motor exerts (N·m, N); None where none is known.
    velocity: float | None
    acceleration: float | None
    jerk: float | None
    effort: float | None
    #: An external axis' mount: ``"BEFORE"`` (the robot rides it), ``"AFTER"`` (it
    #: rides the tool) or ``"EXTERNAL"`` (it stands on its own); None for the
    #: robot's own joints.
    external: str | None


@dataclass(frozen=True)
class Tool:
    """A rig's tool centre point: where its IK aims, and its waypoints are taught.

    A rig has one today; :func:`tools` returns a list so that more can follow.
    """

    #: The bone that marks it.
    name: str
    #: The link it is measured from: the mounting flange, ``tool0``, where the
    #: description has one.
    link: str
    #: The joint bone it rides.
    parent: str
    #: Where it sits from that link, in metres, and turned, as roll, pitch and
    #: yaw in radians: the numbers a URDF ``<origin>`` gives for the same tool.
    offset: tuple[float, float, float]
    rotation: tuple[float, float, float]


@dataclass(frozen=True)
class Solution:
    """What :func:`solve` found: joint values, and how near they put the tool."""

    #: Every joint, in the rig's joint order: the ones the tool hangs from solved,
    #: the rest as seeded.
    values: np.ndarray
    #: Whether they put the tool on the pose, within a tenth of a millimetre and a
    #: hundredth of a degree.
    reached: bool
    #: How far off they leave it, in metres and radians.
    position_error: float
    rotation_error: float


@dataclass(frozen=True)
class Capsule:
    """A capsule the rig's clearance is measured with: a segment and a radius, on a
    bone, in that bone's frame, in metres."""

    bone: str
    a: np.ndarray
    b: np.ndarray
    radius: float


@dataclass(frozen=True)
class Obstacle:
    """Something the robot is kept clear of, as it stands now.

    ``shape`` is ``"BOX"``, ``"SPHERE"`` or ``"FLOOR"``. ``frame`` is a 4x4 in world
    space: a box's centre and rotation, a sphere's centre, or a point on the floor
    with its Z the floor's normal (everything below is solid). ``size`` is a box's
    half-extents, a sphere's radius, or nothing for a floor.
    """

    name: str
    shape: str
    frame: np.ndarray
    size: tuple[float, ...]


@dataclass(frozen=True)
class Clearance:
    """How near the rig comes to an obstacle, at its closest."""

    #: In metres: negative where a capsule reaches into an obstacle; infinite where
    #: there is nothing to measure -- no capsules fitted, or no obstacles.
    distance: float
    #: The bone whose capsule comes closest, and the obstacle; None with nothing
    #: measured.
    bone: str | None
    obstacle: str | None


@dataclass(frozen=True)
class Reach:
    """Whether the rig can stand its tool at a pose: :func:`reachable`."""

    #: The solve put the tool on the pose.
    reachable: bool
    #: Every joint within its range.
    within_limits: bool
    #: Its capsules keep clear of every obstacle, by the distance asked.
    clear: bool
    #: The joint values it was judged at, and its clearance there.
    values: np.ndarray
    clearance: Clearance

    @property
    def ok(self) -> bool:
        """Reachable, within limits and clear."""
        return self.reachable and self.within_limits and self.clear


def _rig(obj):
    if not is_rig(obj):
        raise Error(f"{getattr(obj, 'name', obj)!r} isn't a Kinema rig")
    return obj


def _optional(bone, key) -> float | None:
    value = bone.get(key)
    return None if value is None else float(value)


def _np4(matrix) -> np.ndarray:
    return np.array([[float(matrix[row][col]) for col in range(4)] for row in range(4)])


def is_rig(obj) -> bool:
    """Whether ``obj`` is a Kinema rig."""
    from .rig import builder

    return builder.is_kinema_rig(obj)


def rigs(scene=None) -> list:
    """The Kinema rigs in ``scene``, the current one by default."""
    import bpy

    scene = scene if scene is not None else bpy.context.scene
    return [obj for obj in scene.objects if is_rig(obj)]


def joints(rig) -> list[Joint]:
    """``rig``'s moving joints, in the order of its joint values."""
    from .rig import builder

    found = []
    for pose_bone in builder.joint_bones(_rig(rig)):
        bone = pose_bone.bone
        found.append(
            Joint(
                name=str(bone.get(builder.PROP_JOINT_NAME, pose_bone.name)),
                bone=pose_bone.name,
                type=str(bone.get(builder.PROP_JOINT_TYPE, "revolute")),
                lower=_optional(bone, builder.PROP_LOWER),
                upper=_optional(bone, builder.PROP_UPPER),
                velocity=_optional(bone, builder.PROP_VELOCITY),
                acceleration=_optional(bone, builder.PROP_ACCELERATION),
                jerk=_optional(bone, builder.PROP_JERK),
                effort=_optional(bone, builder.PROP_EFFORT),
                external=bone.get(builder.PROP_EXTERNAL),
            )
        )
    return found


def joint_values(rig) -> np.ndarray:
    """Where ``rig``'s joints stand, as its channels hold them, in radians and metres."""
    from .ops import pose

    return np.array(pose.joint_values(_rig(rig)), dtype=float)


def set_joint_values(rig, values) -> None:
    """Stand ``rig``'s joints at ``values``, one per joint, in radians and metres.

    Its IK goal moves to the tool with them, so live IK leaves the joints where
    they were put. Not keyed: a key is set the way any channel's is.
    """
    from .ops import pose

    rig = _rig(rig)
    pose.set_joint_values(rig, _values(rig, values))


def tools(rig) -> list[Tool]:
    """``rig``'s tool centre points."""
    from .rig import builder

    rig = _rig(rig)
    name = rig.get(builder.PROP_TCP_BONE) or builder.TCP_BONE
    if name not in rig.pose.bones:
        return []
    bone = rig.pose.bones[name]
    parent = getattr(rig, "kinema_tcp_parent", "") or (bone.parent.name if bone.parent else "")
    return [
        Tool(
            name=name,
            link=str(rig.get(builder.PROP_TCP_LINK, "")),
            parent=parent,
            offset=tuple(float(v) for v in getattr(rig, "kinema_tcp_offset", (0.0,) * 3)),
            rotation=tuple(float(v) for v in getattr(rig, "kinema_tcp_rpy", (0.0,) * 3)),
        )
    ]


def tool_frame(rig, tool: Tool | None = None) -> np.ndarray:
    """Where ``tool``, by default the rig's first, is now: a 4x4 in world space,
    its Z out of the tool, as the panel shows it and a URDF ``<origin>`` gives it."""
    from .rig import builder

    rig = _rig(rig)
    bone = rig.pose.bones[_tool(rig, tool).name]
    # The marker bone's axes are a joint bone's, Y along it: the tool's are those
    # turned so its Z is out of the tool.
    return _np4(rig.matrix_world) @ _np4(bone.matrix) @ _np4(builder.BONE_TO_TOOL)


def _tool(rig, tool: Tool | None) -> Tool:
    if tool is None:
        found = tools(rig)
        if not found:
            raise Error(f"{rig.name!r} has no tool centre point")
        return found[0]
    if tool.name not in rig.pose.bones:
        raise Error(f"{rig.name!r} has no tool {tool.name!r}")
    return tool


def _values(rig, values) -> np.ndarray:
    """``values`` checked as a joint vector for ``rig``; its joints now for None."""
    if values is None:
        return joint_values(rig)
    try:
        values = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise Error(f"joint values must be numbers: {exc}") from None
    count = len(joints(rig))
    if values.size != count:
        raise Error(f"{rig.name!r} has {count} joints; {values.size} values were given")
    if not np.all(np.isfinite(values)):
        raise Error("joint values must be finite numbers")
    return values


def _pose(pose) -> np.ndarray:
    try:
        pose = np.asarray(pose, dtype=float)
    except (TypeError, ValueError) as exc:
        raise Error(f"a pose is a 4x4 matrix: {exc}") from None
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise Error("a pose is a 4x4 matrix of finite numbers")
    return pose


def solve(rig, pose, seed=None, tool: Tool | None = None) -> Solution:
    """Joint values that stand ``tool`` (the rig's first by default) at ``pose``, a
    4x4 in world space, Z out of the tool -- without moving the rig.

    Solved from ``seed``, a joint vector, or where the joints stand now; the joints
    the rig holds stay at their seed. On the solver the rig is set to: PyRoki,
    which may compile on its first solve, or NumPy.
    """
    from .rig import builder
    from .solver import manager

    rig = _rig(rig)
    tool = _tool(rig, tool)
    goal = _pose(pose)
    start = _values(rig, seed)
    solver = manager.build_solver(rig, "", tool.name)
    if solver is None:
        raise Error(f"{rig.name!r} has no joints to move its tool {tool.name!r}")
    order = [bone.name for bone in builder.joint_bones(rig)]
    on_chain = [order.index(name) for name in solver.chain.bone_names]
    # The tool's frame to its marker bone's, and world space to the solvers'.
    bone_goal = (
        np.linalg.inv(manager.root_pose(rig))
        @ np.linalg.inv(_np4(rig.matrix_world))
        @ goal
        @ np.linalg.inv(_np4(builder.BONE_TO_TOOL))
    )
    mode = getattr(rig, "kinema_solver_mode", manager.MODE_PYROKI)
    held = manager.held_mask(rig, solver.chain)
    result = solver.solve_for(rig, bone_goal, start[on_chain], mode, held=held)
    values = start.copy()
    values[on_chain] = result.q
    return Solution(
        values=values,
        reached=bool(result.converged),
        position_error=float(result.position_error),
        rotation_error=float(result.rotation_error),
    )


def capsules(rig) -> list[Capsule]:
    """The capsules fitted to ``rig``, which its clearance is measured with."""
    from .ops import collision

    return [
        Capsule(bone=c.bone, a=np.array(c.a), b=np.array(c.b), radius=float(c.radius))
        for c in collision.capsules_of(_rig(rig))
    ]


def _scene(scene):
    import bpy

    return scene if scene is not None else bpy.context.scene


def obstacles(scene=None) -> list[Obstacle]:
    """The obstacles in ``scene``, the current one by default, as they stand now."""
    import bpy

    from .ops import collision

    depsgraph = bpy.context.evaluated_depsgraph_get()
    found = []
    for obj in collision.obstacle_objects(_scene(scene)):
        shape = collision.obstacle_of(obj.evaluated_get(depsgraph))
        found.append(
            Obstacle(
                name=shape.name,
                shape=shape.kind,
                frame=np.array(shape.matrix, dtype=float),
                size=tuple(float(v) for v in shape.size),
            )
        )
    return found


def add_obstacle(obj, shape: str | None = None, scene=None) -> None:
    """Keep the robots in ``scene`` clear of ``obj``, as ``shape`` -- ``"BOX"``,
    ``"SPHERE"`` or ``"FLOOR"`` -- or as the shape it has been given."""
    from .ops import collision
    from .rig import collision as geometry

    if collision.on_a_robot(obj):
        raise Error(f"{obj.name!r} is a robot, or rides on one: it can't be an obstacle")
    if shape is not None:
        shapes = (geometry.KIND_BOX, geometry.KIND_SPHERE, geometry.KIND_FLOOR)
        if shape not in shapes:
            raise Error(f"an obstacle's shape is one of {', '.join(shapes)}, not {shape!r}")
        obj.kinema_obstacle_shape = shape
    collision.add_obstacles(_scene(scene), [obj])


def remove_obstacle(obj, scene=None) -> None:
    """Stop keeping the robots in ``scene`` clear of ``obj``. It stays in the scene."""
    from .ops import collision

    collision.remove_obstacles(_scene(scene), [obj])


def clearance(rig, values=None) -> Clearance:
    """How near ``rig`` comes to the obstacles in the current scene with its joints
    at ``values``, or where they stand now -- without moving it."""
    import bpy

    from .ops import collision
    from .ops import pose as pose_ops

    rig = _rig(rig)
    matrices = None if values is None else pose_ops.bone_matrices(rig, _values(rig, values))
    found = collision.least_clearance(rig, collision.obstacles(bpy.context), matrices)
    if found is None:
        return Clearance(distance=float("inf"), bone=None, obstacle=None)
    distance, bone, obstacle = found
    return Clearance(distance=distance, bone=bone, obstacle=obstacle)


def reachable(rig, pose, seed=None, tool: Tool | None = None, keep_clear: float = 0.0) -> Reach:
    """Whether ``rig`` can stand ``tool`` at ``pose`` (as :func:`solve` does): reach
    it, with every joint in its range, and its capsules ``keep_clear`` metres or more
    from every obstacle. Judged without moving the rig."""
    solution = solve(rig, pose, seed, tool)
    within = all(
        (joint.lower is None or value >= joint.lower - _LIMIT_TOLERANCE)
        and (joint.upper is None or value <= joint.upper + _LIMIT_TOLERANCE)
        for joint, value in zip(joints(rig), solution.values, strict=True)
    )
    measured = clearance(rig, solution.values)
    return Reach(
        reachable=solution.reached,
        within_limits=within,
        clear=measured.distance >= keep_clear,
        values=solution.values,
        clearance=measured,
    )
