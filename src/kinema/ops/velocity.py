"""Warn when a joint moves faster, or speeds up harder, than its limits allow.

A robot cannot follow an animation its motors cannot keep up with, and nothing
in Blender says so: a joint keyed through half a turn in two frames plays back
as smoothly as one given two seconds. Velocity limits are already in most robot
descriptions, as ``<limit velocity>``, and the importer keeps them on each
joint bone; acceleration limits come from a ``joint_limits.yaml``. This checks
the frame on screen against both.

Two places show a joint over a limit: its row in the panel, and its own widget
in the viewport, redrawn in red (``ui/overlay.py``). One tickbox per rig,
**Ignore Motion Limits**, turns both off -- for a shot that is never going to a
real robot, or a description whose limits are placeholders. Jerk and effort
limits are shown in the panel but never checked; see ``rig/velocity.py``.

How a speed or acceleration is measured, and why keyed and IK-driven joints
differ, is in ``rig/velocity.py``. This module reads the channels and feeds the
history.
"""

from __future__ import annotations

import bpy
from bpy.props import BoolProperty, StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper

from ..io.joint_limits import KINDS, JointLimitsError, read_joint_limits
from ..rig import builder
from ..rig.velocity import (
    ACCELERATION,
    SPEED,
    History,
    Reading,
    acceleration,
    clamp,
    speed,
)
from ..solver import manager
from ..solver.chain import chain_bones
from ..ui.panel import active_rig
from .ik import joint_curves

#: The frames each rig was last seen at, for the joints live IK drives.
_history = History()

#: Each rig's readings, worked out once and shared until anything they depend
#: on can have changed. One redraw asks for them several times over -- the
#: Joints panel, the Motion Limits header and rows, and the viewport overlay
#: twice for every rig in every 3D view -- and orbiting the view asks again
#: with nothing changed at all. Keyed by kind, rig, frame and frame rate, and
#: cleared by :func:`observe`, which runs after every frame change and every
#: depsgraph update, and by whatever changes limits or Ignore without one.
_readings: dict = {}


def stored_limit(pose_bone, prop: str) -> float | None:
    """The joint's limit stored under ``prop``, or None if it has none."""
    value = pose_bone.bone.get(prop)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0.0 else None


def limit_of(pose_bone) -> float | None:
    """This joint's velocity limit, or None if it has none."""
    return stored_limit(pose_bone, builder.PROP_VELOCITY)


def acceleration_limit_of(pose_bone) -> float | None:
    """This joint's acceleration limit, or None if it has none."""
    return stored_limit(pose_bone, builder.PROP_ACCELERATION)


#: Where the limit each checked kind is measured against is kept.
_PROPS = {SPEED: builder.PROP_VELOCITY, ACCELERATION: builder.PROP_ACCELERATION}


def _limited(rig, kind: str) -> list:
    """(pose bone, limit) for every joint with a ``kind`` limit, in rig order."""
    prop = _PROPS[kind]
    return [
        (pose_bone, limit)
        for pose_bone in builder.joint_bones(rig)
        if (limit := stored_limit(pose_bone, prop)) is not None
    ]


def limits(rig, kind: str) -> list[Reading]:
    """Every joint's ``kind`` limit, with nothing measured.

    What the panel lists for a rig whose limits are ignored: it reads no curve
    and asks nothing of the solver.
    """
    return [
        Reading(pose_bone.name, limit, None, _is_prismatic(pose_bone), kind)
        for pose_bone, limit in _limited(rig, kind)
    ]


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


def joint_value(pose_bone) -> float:
    """Where this joint stands as the rig shows it: its channel, through its limit."""
    return _clamped(pose_bone, float(getattr(pose_bone, _channel(pose_bone))[1]))


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


def _fps(scene) -> float:
    return scene.render.fps / scene.render.fps_base


def readings(rig, scene) -> list[Reading]:
    """Every speed-limited joint's speed at the current frame, over or not."""
    return _shared(SPEED, rig, scene, _speeds)


def acceleration_readings(rig, scene) -> list[Reading]:
    """Every acceleration-limited joint's acceleration at the current frame, over or not.

    The change of speed over the last two frames: from the joint's curve where
    it has one and nothing else drives it, from the frames seen before where
    live IK does.
    """
    return _shared(ACCELERATION, rig, scene, _accelerations)


def _shared(kind: str, rig, scene, measure) -> list[Reading]:
    """What ``measure`` finds for ``rig``, worked out once for its frame and frame rate."""
    key = (kind, _key(rig), scene.frame_current, _fps(scene))
    found = _readings.get(key)
    if found is None:
        found = _readings[key] = measure(rig, scene)
    return list(found)


def _speeds(rig, scene) -> list[Reading]:
    limited = _limited(rig, SPEED)
    if not limited:
        return []

    fps = _fps(scene)
    frame = scene.frame_current
    curves = joint_curves(rig)
    driven = _ik_driven(rig)
    seen = _history.before(_key(rig), frame)

    result = []
    for pose_bone, limit in limited:
        now = joint_value(pose_bone)
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
            Reading(pose_bone.name, limit, measured, _is_prismatic(pose_bone), SPEED)
        )
    return result


def _accelerations(rig, scene) -> list[Reading]:
    limited = _limited(rig, ACCELERATION)
    if not limited:
        return []

    fps = _fps(scene)
    frame = scene.frame_current
    curves = joint_curves(rig)
    driven = _ik_driven(rig)
    seen = _history.two_before(_key(rig), frame)

    result = []
    for pose_bone, limit in limited:
        now = joint_value(pose_bone)
        curve = curves.get(pose_bone.name)
        if curve is not None and pose_bone.name not in driven:
            points = [(frame, now)] + [
                (frame - back, _clamped(pose_bone, curve.evaluate(frame - back)))
                for back in (1, 2)
            ]
            measured = acceleration(points, fps)
        elif seen is not None and all(pose_bone.name in values for _, values in seen):
            points = [(frame, now)] + [(f, values[pose_bone.name]) for f, values in seen]
            measured = acceleration(points, fps)
        elif pose_bone.name not in driven:
            measured = 0.0
        else:
            measured = None
        result.append(
            Reading(pose_bone.name, limit, measured, _is_prismatic(pose_bone), ACCELERATION)
        )
    return result


def warnings(rig, scene) -> list[Reading]:
    """The readings to flag at the current frame: over their limit, unless ignored.

    Speeds first, then accelerations; a joint over both appears twice, once as
    each :data:`~..rig.velocity.SPEED` and :data:`~..rig.velocity.ACCELERATION`.
    """
    if rig is None or ignored(rig):
        return []
    return [
        reading
        for reading in readings(rig, scene) + acceleration_readings(rig, scene)
        if reading.over
    ]


def observe(scene) -> None:
    """Record where every checked rig stands, after the frame has solved.

    A rig whose limits are ignored is skipped: nothing is read back from it.
    """
    # Whatever changed -- the frame, a pose, a key, the solver's answer -- what
    # is shown next is worked out again.
    _readings.clear()
    frame = scene.frame_current
    for obj in scene.objects:
        if not builder.is_kinema_rig(obj) or ignored(obj):
            continue
        values = {
            pose_bone.name: joint_value(pose_bone)
            for pose_bone in builder.joint_bones(obj)
            if limit_of(pose_bone) is not None
            or acceleration_limit_of(pose_bone) is not None
        }
        if values:
            _history.observe(_key(obj), frame, values)


def forget() -> None:
    """Drop every rig's history: a new file, or the add-on going away."""
    _history.forget()
    _readings.clear()


class KINEMA_OT_load_joint_limits(Operator, ImportHelper):
    bl_idname = "kinema.load_joint_limits"
    bl_label = "Load Joint Limits"
    bl_description = (
        "Read velocity, acceleration, jerk and effort limits from a MoveIt or "
        "ros2_control joint_limits.yaml. They replace the robot description's, "
        "joint by joint"
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
            limits = read_joint_limits(self.filepath)
        except JointLimitsError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        if not limits:
            self.report({"ERROR"}, "No joint limits in that file")
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

        loaded = dict.fromkeys(KINDS, 0)
        cleared = 0
        for name in matched:
            bone = by_joint[name].bone
            for kind, value in limits[name].items():
                prop = builder.LIMIT_PROPS[kind]
                if value is None:
                    if prop in bone:
                        del bone[prop]
                        cleared += 1
                else:
                    bone[prop] = float(value)
                    loaded[kind] += 1

        # The solver reads the same limits; hand it the new ones. And what was
        # measured against the old ones is not what to show any more.
        manager.refresh_limits(rig)
        _readings.clear()

        parts = [f"{count} {kind}" for kind, count in loaded.items() if count]
        message = f"Loaded {_listed(parts) or 'no'} limit{'s' if sum(loaded.values()) != 1 else ''}"
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


def _listed(parts: list[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


classes = (KINEMA_OT_load_joint_limits,)


def _on_ignore_changed(rig, context) -> None:
    """Start the rig's history afresh.

    Nothing is recorded while a rig is ignored, so the frames it holds are from
    before -- and the animation may have changed since. A driven joint reads
    unknown until a frame is stepped, rather than against a stale pose. The
    readings shared from before go with it.
    """
    _history.forget(_key(rig))
    _readings.clear()


def register_props() -> None:
    # Per rig, and saved: a shot that will never run on hardware stays quiet
    # when the file is reopened, and a second robot in the scene keeps warning.
    # Named for velocity, which it was first; renaming it would lose the setting
    # in every file saved before.
    bpy.types.Object.kinema_ignore_velocity = BoolProperty(
        name="Ignore Motion Limits",
        description=(
            "Stop flagging joints that move faster, or speed up harder, than their "
            "limits allow, in the panel and in the viewport"
        ),
        default=False,
        update=_on_ignore_changed,
    )


def unregister_props() -> None:
    del bpy.types.Object.kinema_ignore_velocity
