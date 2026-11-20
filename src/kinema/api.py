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
    "Waypoint",
    "waypoints",
    "add_waypoint",
    "remove_waypoint",
    "Motion",
    "motions",
    "add_motion",
    "remove_motion",
    "add_ik_target",
    "Report",
    "generate",
    "check",
]

#: How far off a pose a solve may leave the tool and have reached it, in metres and
#: radians: what both solvers call converged.
_REACHED_POSITION = 1e-3
_REACHED_ROTATION = 1e-3
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


@dataclass(frozen=True, eq=False)
class Solution:
    """What :func:`solve` found: joint values, and how near they put the tool.

    Compared, and hashed, as itself: it carries an array, which ``==`` can't
    make one answer of.
    """

    #: Every joint, in the rig's joint order: the ones the tool hangs from solved,
    #: the rest as seeded.
    values: np.ndarray
    #: Whether they put the tool on the pose: within a millimetre, and 0.001 rad
    #: (about 0.06°), as Kinema's solvers call a solve done.
    reached: bool
    #: How far off they leave it, in metres and radians.
    position_error: float
    rotation_error: float


@dataclass(frozen=True, eq=False)
class Capsule:
    """A capsule the rig's clearance is measured with: a segment and a radius, on a
    bone, in that bone's frame, in metres. Compared, and hashed, as itself."""

    bone: str
    a: np.ndarray
    b: np.ndarray
    radius: float


@dataclass(frozen=True, eq=False)
class Obstacle:
    """Something the robot is kept clear of, as it stands now.

    ``shape`` is ``"BOX"``, ``"SPHERE"`` or ``"FLOOR"``. ``frame`` is a 4x4 in world
    space: a box's centre and rotation, a sphere's centre, or a point on the floor
    with its Z the floor's normal (everything below is solid). ``size`` is a box's
    half-extents, a sphere's radius, or ``()`` for a floor. Compared, and hashed,
    as itself.
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


@dataclass(frozen=True, eq=False)
class Reach:
    """Whether the rig can stand its tool at a pose: :func:`reachable`. Compared,
    and hashed, as itself."""

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


@dataclass(frozen=True, eq=False)
class Waypoint:
    """A taught pose: where the tool is, and the joints it was taught with.
    Compared, and hashed, as itself."""

    #: What it is called: one name per waypoint on a rig.
    name: str
    #: The joints it was taught with, in the rig's joint order.
    values: np.ndarray
    #: Where it is now: a 4x4 in world space, Z out of the tool, as
    #: :func:`tool_frame` gives it. Its marker, if that has been moved.
    pose: np.ndarray
    #: The object it is taught relative to, by name, which it moves with; None
    #: for the robot's base.
    reference: str | None


@dataclass(frozen=True)
class Motion:
    """One move between two waypoints, over frames of its own."""

    name: str
    #: The waypoints it moves from and to, by name.
    start: str
    end: str
    #: ``"JOINT"``, the joints moving together, or ``"LINEAR"``, the tool along a
    #: straight line, which takes an IK target: see :func:`add_ik_target`.
    move: str
    #: The frame it leaves ``start``, and the frame it is at ``end``.
    frame_start: int
    frame_end: int


@dataclass(frozen=True)
class Report:
    """What :func:`generate` or :func:`check` did, as its operator reports it."""

    #: Whether it was done: False where it was refused, with no keys written. The
    #: job is still caught up first, and a check's findings cleared where it can't
    #: be measured: see :func:`generate` and :func:`check`.
    done: bool
    #: ``"INFO"``, ``"WARNING"`` or ``"ERROR"``.
    level: str
    message: str
    #: How many moves the Motion Check flagged.
    flagged: int


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

    Solved from ``seed``, a joint vector, or where the joints stand now -- a held
    joint where the rig shows it, past a limit or not; the joints the rig holds
    stay at their seed. On the solver the rig is set to: PyRoki,
    which may compile on its first solve, or NumPy.
    """
    from .rig import builder
    from .solver import chain as chain_mod
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
    if seed is None and held.any():
        # A held joint stands where the rig shows it, its limit constraint and
        # all, as live IK seeds it: see chain.displayed_configuration.
        start[on_chain] = chain_mod.displayed_configuration(
            rig, solver.chain, start[on_chain], only=held
        )
    result = solver.solve_for(rig, bone_goal, start[on_chain], mode, held=held)
    values = start.copy()
    values[on_chain] = result.q
    return Solution(
        values=values,
        reached=bool(
            result.position_error < _REACHED_POSITION and result.rotation_error < _REACHED_ROTATION
        ),
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
    from .rig import collision as geometry

    depsgraph = bpy.context.evaluated_depsgraph_get()
    found = []
    for obj in collision.obstacle_objects(_scene(scene)):
        shape = collision.obstacle_of(obj.evaluated_get(depsgraph))
        found.append(
            Obstacle(
                name=shape.name,
                shape=shape.kind,
                frame=np.array(shape.matrix, dtype=float),
                size=()
                if shape.kind == geometry.KIND_FLOOR
                else tuple(float(v) for v in shape.size),
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


def _job(rig):
    """``rig``, checked, with a job saved before motions given its motions."""
    from .ops import waypoints as job

    rig = _rig(rig)
    job.convert_saved_before_motions(rig)
    return rig


def _waypoint_named(rig, name: str):
    found = next((w for w in rig.kinema_waypoints if w.name == name), None)
    if found is None:
        raise Error(f"{rig.name!r} has no waypoint {name!r}")
    return found


def _as_waypoint(rig, waypoint) -> Waypoint:
    from .ops import references
    from .ops import waypoints as job
    from .rig import builder

    reference = references.reference_of(rig, waypoint)
    marker = waypoint.marker
    unparented = marker is not None and marker.parent is None
    if unparented and waypoint.pose_space is not None:
        # Its marker's parent cleared by hand: at the base, where the marker is.
        reference, here = None, np.linalg.inv(_np4(rig.matrix_world)) @ _np4(marker.matrix_world)
    elif reference is None and any(waypoint.taught) and (marker is None or unparented):
        # Its object deleted: back where its joints were taught, at the base.
        here = _np4(references.taught_pose(waypoint))
    else:
        here = _np4(job.pose_of(waypoint))
    return Waypoint(
        name=waypoint.name,
        values=np.array(waypoint.q[: waypoint.dof], dtype=float),
        pose=_np4(rig.matrix_world) @ here @ _np4(builder.BONE_TO_TOOL),
        reference=reference.name if reference is not None else None,
    )


def waypoints(rig) -> list[Waypoint]:
    """``rig``'s waypoints, in the order of its Waypoints list."""
    rig = _job(rig)
    return [_as_waypoint(rig, waypoint) for waypoint in rig.kinema_waypoints]


def add_waypoint(rig, name: str | None = None, values=None, reference=None) -> Waypoint:
    """Teach ``rig`` a waypoint with its joints at ``values``, or where they stand
    now -- without moving it -- relative to ``reference``, an object it then moves
    with, or the robot's base. A name another waypoint has is numbered, as Blender
    names a copy. With its marker."""
    import bpy

    from .ops import references
    from .ops import waypoints as job

    rig = _job(rig)
    values = None if values is None else _values(rig, values)
    if values is not None:
        # Taught as given, where playback would show them limited: refused instead.
        for joint, value in zip(joints(rig), values, strict=True):
            low = -np.inf if joint.lower is None else joint.lower - _LIMIT_TOLERANCE
            high = np.inf if joint.upper is None else joint.upper + _LIMIT_TOLERANCE
            if not low <= value <= high:
                ends = (
                    "none" if end is None else f"{end:.4g}" for end in (joint.lower, joint.upper)
                )
                raise Error(
                    f"{joint.name!r} at {value:.4g} is outside its range, {' to '.join(ends)}"
                )
    if reference is not None and (
        references.PROP_WAYPOINT in reference
        or not references.in_robots_scene(rig, reference)
        or references.relative_to_robot(rig, reference)
    ):
        raise Error(
            f"{reference.name!r} can't hold a waypoint: it is the robot, rides on it, "
            f"is a waypoint's marker, or is in no scene the robot is in"
        )
    # Where everything stands after the script's last change: the reference
    # object's transform among it.
    bpy.context.view_layer.update()
    waypoint, error = job.new_waypoint(rig, name or "", reference, values)
    if error is not None:
        raise Error(error)
    return _as_waypoint(rig, waypoint)


def remove_waypoint(rig, name: str) -> None:
    """Remove ``rig``'s waypoint ``name``, its marker, and the motions to and from it."""
    from .ops import waypoints as job

    rig = _job(rig)
    waypoint = _waypoint_named(rig, name)
    job.delete_waypoint(rig, list(rig.kinema_waypoints).index(waypoint))


def _as_motion(rig, motion) -> Motion:
    from .ops import waypoints as job

    start, end = job.waypoint_by_uid(rig, motion.start), job.waypoint_by_uid(rig, motion.end)
    return Motion(
        name=motion.name,
        start=start.name if start is not None else "",
        end=end.name if end is not None else "",
        move=motion.move,
        frame_start=int(motion.frame_start),
        frame_end=int(motion.frame_end),
    )


def motions(rig) -> list[Motion]:
    """``rig``'s motions, in the order of its Motions list."""
    rig = _job(rig)
    return [_as_motion(rig, motion) for motion in rig.kinema_motions]


def add_motion(
    rig,
    start: str,
    end: str,
    frame_start: int,
    frame_end: int,
    move: str = "JOINT",
    name: str | None = None,
) -> Motion:
    """A motion on ``rig`` from waypoint ``start`` to ``end``, by name, leaving on
    ``frame_start`` and arriving on ``frame_end``; named ``name``, or after ``end``.
    Motions that overlap are found when the job is generated or checked."""
    import bpy

    from .ops import waypoints as job
    from .rig.waypoints import MOVE_TYPES

    rig = _job(rig)
    if move not in MOVE_TYPES:
        raise Error(f"a move is one of {', '.join(MOVE_TYPES)}, not {move!r}")
    if int(frame_end) <= int(frame_start):
        raise Error(f"a motion ends after it starts: frames {frame_start}-{frame_end}")
    frames = bpy.types.Scene.bl_rna.properties["frame_current"]
    if int(frame_start) - 1 < frames.hard_min or int(frame_end) + 1 > frames.hard_max:
        raise Error(
            f"frames {frame_start}-{frame_end} reach past Blender's, "
            f"{frames.hard_min + 1} to {frames.hard_max - 1} for a motion: a check plays "
            f"a frame either side"
        )
    motion = job.new_motion(
        rig,
        _waypoint_named(rig, start),
        _waypoint_named(rig, end),
        int(frame_start),
        int(frame_end),
        move,
        name or "",
    )
    return _as_motion(rig, motion)


def remove_motion(rig, name: str) -> None:
    """Remove ``rig``'s motion ``name``. Its waypoints stay."""
    from .ops import waypoints as job

    rig = _job(rig)
    found = [i for i, motion in enumerate(rig.kinema_motions) if motion.name == name]
    if not found:
        raise Error(f"{rig.name!r} has no motion {name!r}")
    if len(found) > 1:
        raise Error(f"{rig.name!r} has {len(found)} motions called {name!r}: rename one")
    job.delete_motion(rig, found[0])


def add_ik_target(rig) -> None:
    """Give ``rig`` an IK target at its tool, which linear moves are keyed on, with
    live IK on. Nothing changes on a rig that has one."""
    import bpy

    from .ops import ik
    from .ops import waypoints as job

    rig = _rig(rig)
    if job.ik_bone_name(rig) is not None:
        return
    if rig.name not in bpy.context.view_layer.objects:
        raise Error(
            f"{rig.name!r} isn't in the view layer, where its bones are edited: "
            f"include its collection, or switch to a view layer that has it"
        )
    outcome = ik.add_ik_target(bpy.context, rig)
    if not outcome.done:
        raise Error(outcome.message)


def _report(outcome) -> Report:
    return Report(
        done=outcome.done,
        level=outcome.level,
        message=outcome.message,
        flagged=outcome.flagged,
    )


def generate(rig) -> Report:
    """Key ``rig`` through its motions in frame order, and check the job: Generate
    Motion. Refused with why -- motions that overlap, a linear move with no IK
    target -- it keys nothing.

    As Generate Motion does, it first catches the job up, refused or not: a rig
    saved before motions is given them, and a waypoint whose object is gone goes
    back to the base, where its joints were taught.
    """
    import bpy

    from .ops import waypoints as job

    return _report(job.generate_job(bpy.context, _job(rig)))


def check(rig) -> Report:
    """Play ``rig``'s job through once, solving as playback does, and check each
    move -- its line, limits, speeds and clearance: Check Motion. The findings are
    kept on the rig, for the panel. Where the job can't be measured -- a rig with
    no TCP -- those from before are cleared, so none reads as current; the job is
    caught up as :func:`generate` says."""
    import bpy

    from .ops import waypoints as job

    return _report(job.check_motion(bpy.context, _job(rig)))
