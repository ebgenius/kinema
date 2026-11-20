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
]


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
    try:
        values = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise Error(f"joint values must be numbers: {exc}") from None
    count = len(joints(rig))
    if values.size != count:
        raise Error(f"{rig.name!r} has {count} joints; {values.size} values were given")
    if not np.all(np.isfinite(values)):
        raise Error("joint values must be finite numbers")
    pose.set_joint_values(rig, values)


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
    if tool is None:
        found = tools(rig)
        if not found:
            raise Error(f"{rig.name!r} has no tool centre point")
        tool = found[0]
    bone = rig.pose.bones.get(tool.name)
    if bone is None:
        raise Error(f"{rig.name!r} has no tool {tool.name!r}")
    # The marker bone's axes are a joint bone's, Y along it: the tool's are those
    # turned so its Z is out of the tool.
    return _np4(rig.matrix_world) @ _np4(bone.matrix) @ _np4(builder.BONE_TO_TOOL)
