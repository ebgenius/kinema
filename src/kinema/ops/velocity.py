"""Warn when a joint moves faster than its velocity limit.

A robot cannot follow an animation its motors cannot keep up with, and nothing
in Blender says so: a joint keyed through half a turn in two frames plays back
as smoothly as one given two seconds. The limits are already in most robot
descriptions, as ``<limit velocity>``, and the importer keeps them on each
joint bone. This checks the frame on screen against them.

Two places show a joint over its limit: its row in the panel, and its own
widget in the viewport, redrawn in red (``ui/overlay.py``). One tickbox per rig,
**Ignore Velocity Limits**, turns both off -- for a shot that is never going to
a real robot, or a description whose limits are placeholders.

How a speed is measured, and why keyed and IK-driven joints differ, is in
``rig/velocity.py``. This module reads the channels and feeds the history.
"""

from __future__ import annotations

import bpy
from bpy.props import BoolProperty, StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper

from ..io.joint_limits import JointLimitsError, read_velocity_limits
from ..rig import builder
from ..rig.velocity import History, Reading, clamp, speed
from ..solver import manager
from ..solver.chain import chain_bones
from ..ui.panel import active_rig
from .ik import own_fcurve_containers

#: The frames each rig was last seen at, for the joints live IK drives.
_history = History()


def limit_of(pose_bone) -> float | None:
    """This joint's velocity limit, or None if it has none."""
    value = pose_bone.bone.get(builder.PROP_VELOCITY)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0.0 else None


def ignored(rig) -> bool:
    return bool(getattr(rig, "kinema_ignore_velocity", False))


def _is_prismatic(pose_bone) -> bool:
    return pose_bone.bone.get(builder.PROP_JOINT_TYPE, "revolute") == "prismatic"


def _channel(pose_bone) -> str:
    return "location" if _is_prismatic(pose_bone) else "rotation_euler"


def _clamped(pose_bone, value: float) -> float:
    """``value`` as the rig shows it: through the joint's limit constraint.

    A curve keyed past a joint's range moves the channel, but the constraint
    holds the bone still, so the speed that matters is the clamped one.
    Without this a joint pressed against its stop would read as racing.
    """
    constraint = pose_bone.constraints.get(builder.LIMIT_CONSTRAINT)
    if constraint is None or not constraint.enabled:
        return value
    if constraint.type == "LIMIT_ROTATION":
        lower = constraint.min_y if constraint.use_limit_y else None
        upper = constraint.max_y if constraint.use_limit_y else None
    else:
        lower = constraint.min_y if constraint.use_min_y else None
        upper = constraint.max_y if constraint.use_max_y else None
    return value + (clamp(value, lower, upper) - value) * constraint.influence


def _value(pose_bone) -> float:
    return _clamped(pose_bone, float(getattr(pose_bone, _channel(pose_bone))[1]))


def _joint_curves(rig) -> dict[str, bpy.types.FCurve]:
    """Bone name -> the curve animating its one joint channel, where there is one."""
    wanted = {
        (pose_bone.path_from_id(_channel(pose_bone)), 1): pose_bone.name
        for pose_bone in builder.joint_bones(rig)
    }
    found: dict[str, bpy.types.FCurve] = {}
    for container in own_fcurve_containers(rig):
        for curve in container:
            name = wanted.get((curve.data_path, curve.array_index))
            if name is not None and not curve.mute:
                found.setdefault(name, curve)
    return found


def _ik_driven(rig) -> set[str]:
    """Joints live IK writes on this frame, whatever their curves say.

    The same test the live handler makes. The chain is walked from the tip the
    rig aims at now, the way the solver builds it, rather than read off a
    cached solver: there may be none yet -- a file just opened -- and the tip
    is keyframable, so a cached one can describe a different chain. Joints
    past a tip set part-way up are never written, and keep their curves.
    """
    if not getattr(rig, "kinema_ik_enabled", False):
        return set()
    if getattr(rig, "kinema_solver_mode", "PYROKI") == "OFF":
        return set()
    ik_name = rig.get(builder.PROP_IK_BONE)
    if not ik_name or ik_name not in rig.pose.bones:
        return set()
    pose = rig.pose.bones
    return {
        bone.name
        for bone in chain_bones(rig, manager.tip_bone(rig))
        if bone.name in pose and not getattr(pose[bone.name], "kinema_ik_hold", False)
    }


def _key(rig):
    # Not the name: Blender hands a deleted object's name straight to the next
    # one, which would inherit the old rig's history.
    return rig.session_uid


def readings(rig, scene) -> list[Reading]:
    """Every speed-limited joint's speed at the current frame, over or not."""
    limited = [
        (pose_bone, limit)
        for pose_bone in builder.joint_bones(rig)
        if (limit := limit_of(pose_bone)) is not None
    ]
    if not limited:
        return []

    fps = scene.render.fps / scene.render.fps_base
    frame = scene.frame_current
    curves = _joint_curves(rig)
    driven = _ik_driven(rig)
    seen = _history.before(_key(rig), frame)

    result = []
    for pose_bone, limit in limited:
        now = _value(pose_bone)
        curve = curves.get(pose_bone.name)
        if curve is not None and pose_bone.name not in driven:
            before = _clamped(pose_bone, curve.evaluate(frame - 1))
            measured = speed(now, before, 1, fps)
        elif seen is not None and pose_bone.name in seen[1]:
            measured = speed(now, seen[1][pose_bone.name], frame - seen[0], fps)
        elif pose_bone.name not in driven:
            # No curve and no solver: nothing moves it from frame to frame.
            measured = 0.0
        else:
            measured = None
        result.append(
            Reading(pose_bone.name, limit, measured, _is_prismatic(pose_bone))
        )
    return result


def warnings(rig, scene) -> list[Reading]:
    """The joints to flag at the current frame: over their limit, unless ignored."""
    if rig is None or ignored(rig):
        return []
    return [reading for reading in readings(rig, scene) if reading.over]


def observe(scene) -> None:
    """Record where every speed-limited rig stands, after the frame has solved."""
    frame = scene.frame_current
    for obj in scene.objects:
        if not builder.is_kinema_rig(obj):
            continue
        values = {
            pose_bone.name: _value(pose_bone)
            for pose_bone in builder.joint_bones(obj)
            if limit_of(pose_bone) is not None
        }
        if values:
            _history.observe(_key(obj), frame, values)


def forget() -> None:
    _history.forget()


class KINEMA_OT_load_joint_limits(Operator, ImportHelper):
    bl_idname = "kinema.load_joint_limits"
    bl_label = "Load Joint Limits"
    bl_description = (
        "Read velocity limits from a MoveIt or ros2_control joint_limits.yaml. "
        "They replace the robot description's, joint by joint"
    )
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".yaml"
    filter_glob: StringProperty(default="*.yaml;*.yml", options={"HIDDEN"})

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return active_rig(context) is not None

    def execute(self, context: bpy.types.Context) -> set[str]:
        rig = active_rig(context)
        try:
            limits = read_velocity_limits(self.filepath)
        except JointLimitsError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        if not limits:
            self.report({"ERROR"}, "No velocity limits in that file")
            return {"CANCELLED"}

        # By URDF joint name, which is what the file uses. The bone is usually
        # called the same, but Blender renames a bone whose name is taken.
        by_joint = {
            pose_bone.bone.get(builder.PROP_JOINT_NAME, pose_bone.name): pose_bone
            for pose_bone in builder.joint_bones(rig)
        }
        matched = [name for name in limits if name in by_joint]
        if not matched:
            # Said with an example of each side: the usual cause is a prefix,
            # panda_joint1 against joint1, and seeing both makes that obvious.
            ours = next(iter(by_joint), "none")
            theirs = next(iter(limits))
            self.report(
                {"ERROR"},
                f"No joint in that file is on this rig (the file has "
                f"'{theirs}', the rig has '{ours}')",
            )
            return {"CANCELLED"}

        set_count = cleared = 0
        for name in matched:
            bone = by_joint[name].bone
            value = limits[name]
            if value is None:
                if builder.PROP_VELOCITY in bone:
                    del bone[builder.PROP_VELOCITY]
                    cleared += 1
            else:
                bone[builder.PROP_VELOCITY] = float(value)
                set_count += 1

        message = f"Loaded {set_count} velocity limit{'s' if set_count != 1 else ''}"
        if cleared:
            message += f", turned {cleared} off"
        unknown = len(limits) - len(matched)
        if unknown:
            self.report(
                {"WARNING"},
                f"{message}. {unknown} joint{'s' if unknown != 1 else ''} in the "
                f"file {'are' if unknown != 1 else 'is'} not on this rig",
            )
        else:
            self.report({"INFO"}, message)
        return {"FINISHED"}


classes = (KINEMA_OT_load_joint_limits,)


def register_props() -> None:
    # Per rig, and saved: a shot that will never run on hardware stays quiet
    # when the file is reopened, and a second robot in the scene keeps warning.
    bpy.types.Object.kinema_ignore_velocity = BoolProperty(
        name="Ignore Velocity Limits",
        description=(
            "Stop flagging joints that move faster than their velocity limit, "
            "in the panel and in the viewport"
        ),
        default=False,
    )


def unregister_props() -> None:
    del bpy.types.Object.kinema_ignore_velocity
