"""Waypoints taught relative to an object: a workpiece, a fixture, a conveyor.

A waypoint's pose is held in its frame of reference: the robot's base, as it
always was, or an object in the scene. Its marker is parented to that object
without a parent inverse, as Blender's *Make Parent without Inverse* parents,
so the marker's own Location and Rotation are the pose in the object's space,
and moving or animating the object takes the waypoint with it. Reparenting a
marker with Blender's own operators changes its frame of reference the same
way: the marker's parent is what is read (:func:`reference_of`).

Where a waypoint is on a frame is the object's pose there, in rig space, times
the waypoint's pose in the object's space. :class:`Placement` reads the
objects' poses over a job's frames once, and Generate Motion, the Motion Check
and Optimize Motion all ask it. A job with no waypoint relative to anything
reads nothing, and is placed exactly as it was before there was a choice.

The joints a waypoint was taught in hold where it was taught, in rig space,
which the waypoint keeps (``taught``). Wherever the job meets it somewhere
else -- its object has moved, or is moving -- the joints are walked there from
those, so every visit keeps the configuration taught.
"""

from __future__ import annotations

import hashlib

import bpy
import numpy as np
from mathutils import Matrix

#: How far a frame of reference may move in a move before the move follows it
#: frame by frame, in rig-space matrix entries: metres, and the rotation's
#: sines and cosines. Below it the object is standing still.
TRACK_TOLERANCE = 1e-6

#: Marks an Empty as belonging to a waypoint (as ``waypoints.PROP_WAYPOINT``).
PROP_WAYPOINT = "kinema_waypoint"

#: An object's own transform channels: what places it, keyed or not.
TRANSFORM = (
    "location",
    "rotation_euler",
    "rotation_quaternion",
    "rotation_axis_angle",
    "scale",
    "delta_location",
    "delta_rotation_euler",
    "delta_rotation_quaternion",
    "delta_scale",
)


def _unflatten(values) -> Matrix:
    return Matrix([[float(values[row * 4 + col]) for col in range(4)] for row in range(4)])


def _flatten(matrix) -> list[float]:
    return [float(matrix[row][col]) for row in range(4) for col in range(4)]


def reference_of(rig, waypoint):
    """The object ``waypoint`` is taught relative to, or None for the robot's base.

    The marker's parent where it has one: the marker is the waypoint, and
    parenting it elsewhere, with Kinema's field or Blender's own operators, is
    what changes its frame. Without a marker, the object its pose was last
    expressed in.
    """
    marker = waypoint.marker
    if marker is not None and marker.parent is not None:
        return None if marker.parent == rig else marker.parent
    return waypoint.pose_space


def local_pose(waypoint) -> Matrix:
    """The waypoint's pose in its frame of reference.

    Its marker's, where it has one: the transform in its parent's space, which
    with no parent inverse -- how Kinema parents it -- is its own Location and
    Rotation. The stored pose otherwise.
    """
    marker = waypoint.marker
    if marker is not None and marker.parent is not None:
        return marker.matrix_parent_inverse @ marker.matrix_basis
    return _unflatten(waypoint.pose)


def taught_pose(waypoint) -> Matrix:
    """Where, in rig space, the waypoint's joint values were taught."""
    if waypoint.pose_space is not None or any(waypoint.taught):
        return _unflatten(waypoint.taught)
    return _unflatten(waypoint.pose)


def in_rig(rig, obj) -> Matrix:
    """``obj``'s pose in ``rig``'s space as the scene stands, or the identity for the base."""
    if obj is None:
        return Matrix.Identity(4)
    return rig.matrix_world.inverted() @ obj.matrix_world


def relative_to_robot(rig, obj) -> bool:
    """Whether ``obj`` moves with the robot: the rig, or anything parented under it.

    A frame of reference the robot carries would have the tool chase its own
    tail, so none may be one.
    """
    while obj is not None:
        if obj == rig:
            return True
        obj = obj.parent
    return False


def carried(rig, waypoint):
    """The object the robot carries that ``waypoint``'s marker is parented to, or None:
    the rig itself for a marker parented to one of its bones.

    The field refuses one, but Blender's own parenting doesn't ask: a marker
    parented by hand to a gripper would have the tool chase its own tail.
    """
    marker = waypoint.marker
    if marker is not None and marker.parent == rig and marker.parent_type == "BONE":
        return rig
    reference = reference_of(rig, waypoint)
    if reference is not None and relative_to_robot(rig, reference):
        return reference
    return None


def place_marker(rig, waypoint, reference, local: Matrix) -> None:
    """Parent the marker to ``reference`` (the rig for None) without an inverse, at ``local``."""
    marker = waypoint.marker
    if marker is None:
        return
    marker.parent = reference if reference is not None else rig
    marker.matrix_parent_inverse = Matrix.Identity(4)
    marker.matrix_basis = local


def reexpress(rig, waypoint, reference) -> bool:
    """Make ``reference`` the waypoint's frame, keeping it where it is in the scene.

    As *Make Parent without Inverse* with *Keep Transform*: the pose is worked
    out again in the new object's space, so nothing moves. The joints stay
    where they were taught.

    A marker reparented by hand is caught up first (:func:`sync`), so where
    its joints were taught is kept before its frame changes again. False, and
    nothing changed, for an object scaled to nothing, which has no space to
    be in.

    Without a marker, a waypoint at the base is where its joints were taught,
    since there is nothing else to hold where it is: put back there, it goes
    back there.
    """
    sync(rig, waypoint)
    was = reference_of(rig, waypoint)
    if was == reference:
        settle(waypoint, reference)
        return True
    try:
        into = in_rig(rig, reference).inverted()
    except ValueError:
        settle(waypoint, was)
        return False
    here = in_rig(rig, was) @ local_pose(waypoint)
    if was is None:
        # From the base, where the stored pose is where the joints were taught.
        waypoint.taught = _flatten(taught_pose(waypoint))
    local = into @ here
    if reference is None:
        # Back to the base, where the stored pose is the taught one again and
        # the marker is where the waypoint is now: the next generation walks
        # the joints from one to the other, as for any marker moved.
        waypoint.pose = _flatten(taught_pose(waypoint))
        waypoint.taught = [0.0] * 16
    else:
        waypoint.pose = _flatten(local)
    place_marker(rig, waypoint, reference, local)
    settle(waypoint, reference)
    return True


def settle(waypoint, reference) -> None:
    """Both of the waypoint's own records of its frame, set without re-expressing."""
    if waypoint.pose_space != reference:
        waypoint.pose_space = reference
    if waypoint.reference != reference:
        waypoint["_settling"] = True
        try:
            waypoint.reference = reference
        finally:
            del waypoint["_settling"]


def sync(rig, waypoint) -> str | None:
    """Bring the waypoint's records up to its marker, after Blender reparented it.

    Parented with *Keep Transform*, the marker has moved nowhere and its numbers
    are new; without, it kept the numbers and moved with the object. Either way
    the marker says where the waypoint is, and this records which object it is
    relative to now, and its pose there.

    An object deleted takes itself out of both the field and the marker's
    parent, and leaves a pose in the space of something that is gone. The
    waypoint goes back where its joints were taught, relative to the robot,
    and what is returned says so. A marker parented to something the robot
    carries is left as it is: see :func:`carried`.
    """
    if carried(rig, waypoint) is not None:
        return None
    reference = reference_of(rig, waypoint)
    marker = waypoint.marker
    note = None
    if reference is None and (waypoint.pose_space is not None or any(waypoint.taught)):
        # Back at the base, by hand or because its object is gone: there the
        # stored pose is the taught one.
        gone = waypoint.pose_space is None and (marker is None or marker.parent is None)
        taught = taught_pose(waypoint)
        waypoint.pose = _flatten(taught)
        waypoint.taught = [0.0] * 16
        if gone:
            place_marker(rig, waypoint, None, taught)
            note = (
                f"'{waypoint.name}': the object it was relative to is gone, so it is "
                f"back where it was taught, relative to the robot"
            )
    elif reference is not None:
        if waypoint.pose_space is None and not any(waypoint.taught):
            # Taught at the base and reparented by hand: the joints hold where
            # the pose was stored.
            waypoint.taught = list(waypoint.pose)
        if waypoint.pose_space != reference and marker is not None:
            # Its pose in its new frame, which is what the marker now holds.
            waypoint.pose = _flatten(local_pose(waypoint))
    settle(waypoint, reference)
    return note


def on_reference_set(waypoint, _context) -> None:
    """The waypoint's Relative To field changed: re-express it in the new object."""
    if waypoint.get("_settling"):
        return
    rig = waypoint.id_data
    reference = waypoint.reference
    if reference is not None and relative_to_robot(rig, reference):
        # Refused by the field's own filter too; this catches a script.
        settle(waypoint, reference_of(rig, waypoint))
        return
    reexpress(rig, waypoint, reference)


def may_be_reference(waypoint, obj) -> bool:
    """The field's filter: any object in a scene but the robot, what it carries, and
    markers. An object in no scene -- a bone's custom shape, kept with a fake
    user -- is nowhere for a waypoint to be."""
    rig = waypoint.id_data
    return (
        PROP_WAYPOINT not in obj
        and bool(obj.users_scene)
        and not relative_to_robot(rig, obj)
    )


def _rounded(values, places: int = 6) -> tuple:
    return tuple(round(float(v), places) for v in values)


def _settings(struct) -> tuple:
    """A struct's own editable values: an NLA strip's timing, a curve modifier's
    kind of shaping. Selection and what is shown are left out, since clicking
    on something changes nothing about where it goes."""
    out = []
    for prop in struct.bl_rna.properties:
        name = prop.identifier
        if (
            prop.is_readonly
            or prop.type not in {"BOOLEAN", "INT", "FLOAT", "ENUM"}
            or name.startswith(("select", "show_"))
            or name in {"active", "is_valid"}
        ):
            continue
        value = getattr(struct, name)
        if isinstance(value, set):
            value = tuple(sorted(value))
        elif hasattr(value, "__len__") and not isinstance(value, str):
            value = _rounded(value)
        elif isinstance(value, float):
            value = round(value, 6)
        out.append((name, value))
    return tuple(out)


#: A key's numbers and its kinds of easing, as ``foreach_get`` reads them: the
#: number of values each holds.
_KEY_FLOATS = (("co", 2), ("handle_left", 2), ("handle_right", 2), ("back", 1),
               ("amplitude", 1), ("period", 1))
_KEY_KINDS = ("interpolation", "easing")


def _read(points, name: str, width: int, dtype) -> np.ndarray:
    values = np.empty(len(points) * width, dtype=dtype)
    if len(points):
        points.foreach_get(name, values)
    return values


def _curve(curve) -> tuple:
    """Everything an F-curve's value on any frame comes from: its keys, with their
    handles and easing; the samples of a baked one; how it carries on past
    them, and its modifiers.

    The keys are read a field at a time, all at once, and digested: a curve
    keyed on every frame of a long job is thousands of keys, and this runs
    on every redraw of the Motion Check panel.
    """
    keys, samples = curve.keyframe_points, curve.sampled_points
    # float32, as Blender holds them: any other type is copied a value at a time.
    floats = [_read(keys, name, width, np.float32) for name, width in _KEY_FLOATS]
    floats.append(_read(samples, "co", 2, np.float32))
    # Rounded as the rest of a fingerprint is; adding zero makes -0.0 read as 0.0.
    numbers = np.round(np.concatenate(floats), 6) + 0.0
    kinds = np.concatenate([_read(keys, name, 1, np.int32) for name in _KEY_KINDS])
    digest = hashlib.sha1(numbers.tobytes() + kinds.tobytes()).hexdigest()
    modifiers = tuple((m.type, _settings(m)) for m in curve.modifiers)
    return (
        curve.data_path,
        int(curve.array_index),
        curve.mute,
        curve.extrapolation,
        len(keys),
        digest,
        modifiers,
    )


def _strip(strip, transform_curves) -> tuple:
    """An NLA strip: its timing, the curves that animate its influence and time,
    and its action's transform curves, or the strips a meta strip holds."""
    animated = set()
    if strip.use_animated_influence:
        animated.add("influence")
    if strip.use_animated_time:
        animated.add("strip_time")
    # An animated control's value as it stands changes as the timeline moves:
    # its curve is what says where the strip goes.
    settings = tuple(item for item in _settings(strip) if item[0] not in animated)
    controls = tuple(_curve(curve) for curve in strip.fcurves)
    action = strip.action
    curves = ()
    if action is not None:
        slot = getattr(strip, "action_slot", None)
        handle = getattr(slot, "handle", None)
        from .ik import _iter_fcurve_containers

        curves = transform_curves(_iter_fcurve_containers(action, handle))
    inside = tuple(_strip(child, transform_curves) for child in strip.strips)
    return (settings, controls, curves, inside)


def _animation(obj) -> tuple[tuple, set]:
    """``obj``'s transform curves, in its active action and its NLA strips, and
    which channels they key."""
    from .ik import own_fcurve_containers

    anim = obj.animation_data
    if anim is None:
        return (), set()
    keyed = set()

    def transform_curves(containers):
        out = []
        for container in containers:
            for curve in container:
                if curve.data_path in TRANSFORM:
                    keyed.add((curve.data_path, int(curve.array_index)))
                    out.append(_curve(curve))
        return tuple(sorted(out, key=lambda c: (c[0], c[1])))

    active = transform_curves(own_fcurve_containers(obj) or ())
    tracks = tuple(
        (_settings(track), tuple(_strip(strip, transform_curves) for strip in track.strips))
        for track in anim.nla_tracks
    )
    return (_settings(anim), active, tracks), keyed


def fingerprint(obj, depth: int = 0) -> tuple:
    """What ``obj``'s place over a job comes from, as far as it can be read without
    playing the job: its transform curves -- keys, interpolation, modifiers --
    in its action and its NLA strips; the channels nothing keys, as they stand;
    and its parent's the same way.

    For a job's fingerprint (``waypoints.job_signature``), so moving a
    workpiece, or re-keying it, marks a Motion Check of it as outdated. Read
    off the data, not the scene -- a keyed channel's value as it stands is left
    out -- so scrubbing the timeline changes nothing. Only the object's own
    transform is in it, so a robot's fingerprint is not its joints' keys. What
    drives it from elsewhere -- a constraint's target, a driver -- is not in it:
    Check Again catches that.
    """
    if obj is None or depth > 16:
        return ()
    animation, keyed = _animation(obj)
    still = tuple(
        (path, index, round(float(value), 6))
        for path in TRANSFORM
        for index, value in enumerate(getattr(obj, path))
        if (path, index) not in keyed
    )
    placed = (
        obj.rotation_mode,
        obj.parent_type,
        obj.parent_bone,
        _rounded(_flatten(obj.matrix_parent_inverse)),
    )
    constraints = tuple(
        (c.type, getattr(getattr(c, "target", None), "name", "")) for c in obj.constraints
    )
    return (obj.name, animation, still, placed, constraints, fingerprint(obj.parent, depth + 1))


class Placement:
    """Where a job's waypoints are, frame by frame, in rig space.

    Each frame of reference's pose in rig space is read for every frame asked
    for, once, by stepping the scene there with the handlers suspended, and the
    scene is put back where it was. With nothing relative to any object there
    is nothing to read: every waypoint is where its marker is, on every frame.
    """

    def __init__(self, rig, waypoints, frames=(), scene=None):
        self.rig = rig
        self.references = {}
        for waypoint in waypoints:
            reference = reference_of(rig, waypoint)
            if reference is not None:
                # By identity: two objects linked from different libraries can
                # share a name.
                self.references[reference.as_pointer()] = reference
        self._at: dict[int, dict[int, Matrix]] = {}
        frames = sorted(set(int(f) for f in frames))
        if self.references and frames:
            self._read(frames, scene or bpy.context.scene)

    def _read(self, frames, scene) -> None:
        from .. import handlers

        original, subframe = scene.frame_current, scene.frame_subframe
        try:
            with handlers.suspended():
                for frame in frames:
                    scene.frame_set(frame)
                    self._at[frame] = {
                        key: in_rig(self.rig, obj) for key, obj in self.references.items()
                    }
        finally:
            scene.frame_set(original, subframe=subframe)

    @property
    def relative(self) -> bool:
        """Whether any waypoint is relative to an object."""
        return bool(self.references)

    def frame_of(self, reference, frame: int | None) -> Matrix:
        """``reference``'s pose in rig space on ``frame``: as the scene stands, for None."""
        if reference is None:
            return Matrix.Identity(4)
        at = self._at.get(int(frame)) if frame is not None else None
        if at is not None and reference.as_pointer() in at:
            return at[reference.as_pointer()].copy()
        return in_rig(self.rig, reference)

    def pose(self, waypoint, frame: int | None = None) -> Matrix:
        """Where ``waypoint`` is on ``frame``, in rig space."""
        reference = reference_of(self.rig, waypoint)
        local = local_pose(waypoint)
        if reference is None:
            return local
        return self.frame_of(reference, frame) @ local

    def moves(self, waypoint, first: int, last: int) -> bool:
        """Whether ``waypoint``'s frame of reference moves between ``first`` and ``last``."""
        reference = reference_of(self.rig, waypoint)
        if reference is None or first == last:
            return False
        start = self.frame_of(reference, first)
        for frame in range(int(first) + 1, int(last) + 1):
            other = self.frame_of(reference, frame)
            if any(
                abs(other[r][c] - start[r][c]) > TRACK_TOLERANCE
                for r in range(4)
                for c in range(4)
            ):
                return True
        return False

    def tracks(self, span) -> bool:
        """Whether ``span`` follows a moving frame of reference on every frame."""
        return self.moves(span.start, span.start_frame, span.end_frame) or self.moves(
            span.end, span.start_frame, span.end_frame
        )
