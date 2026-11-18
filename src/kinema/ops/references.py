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

import bpy
from mathutils import Matrix

#: How far a frame of reference may move in a move before the move follows it
#: frame by frame, in rig-space matrix entries: metres, and the rotation's
#: sines and cosines. Below it the object is standing still.
TRACK_TOLERANCE = 1e-6

#: Marks an Empty as belonging to a waypoint (as ``waypoints.PROP_WAYPOINT``).
PROP_WAYPOINT = "kinema_waypoint"


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


def place_marker(rig, waypoint, reference, local: Matrix) -> None:
    """Parent the marker to ``reference`` (the rig for None) without an inverse, at ``local``."""
    marker = waypoint.marker
    if marker is None:
        return
    marker.parent = reference if reference is not None else rig
    marker.matrix_parent_inverse = Matrix.Identity(4)
    marker.matrix_basis = local


def reexpress(rig, waypoint, reference) -> None:
    """Make ``reference`` the waypoint's frame, keeping it where it is in the scene.

    As *Make Parent without Inverse* with *Keep Transform*: the pose is worked
    out again in the new object's space, so nothing moves. The joints stay
    where they were taught.
    """
    was = reference_of(rig, waypoint)
    if was == reference:
        settle(waypoint, reference)
        return
    here = in_rig(rig, was) @ local_pose(waypoint)
    if was is None:
        # From the base, where the stored pose is where the joints were taught.
        waypoint.taught = _flatten(taught_pose(waypoint))
    local = in_rig(rig, reference).inverted() @ here
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


def sync(rig, waypoint) -> None:
    """Bring the waypoint's field up to its marker, after Blender reparented it.

    Parented with *Keep Transform*, the marker has moved nowhere and its numbers
    are new; without, it kept the numbers and moved with the object. Either way
    the marker says where the waypoint is, and this only records which object
    it is relative to now.
    """
    reference = reference_of(rig, waypoint)
    if reference is not None and waypoint.pose_space is None and not any(waypoint.taught):
        # Taught at the base and reparented by hand: the joints hold where the
        # pose was stored.
        waypoint.taught = list(waypoint.pose)
    elif reference is None and waypoint.pose_space is not None:
        # Back on the rig by hand: at the base the stored pose is the taught one.
        waypoint.pose = list(waypoint.taught)
        waypoint.taught = [0.0] * 16
    settle(waypoint, reference)


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
    """The field's filter: any object but the robot, what it carries, and markers."""
    rig = waypoint.id_data
    return PROP_WAYPOINT not in obj and not relative_to_robot(rig, obj)


def fingerprint(obj, depth: int = 0) -> tuple:
    """What ``obj``'s place over a job comes from, as far as it can be read without
    playing the job: its keys where it is animated, its transform where it is
    not, and its parent's the same way.

    For a job's fingerprint (``waypoints.job_signature``), so moving a
    workpiece, or re-keying it, marks a Motion Check of it as outdated. Read
    off the data, not the scene, so scrubbing the timeline changes nothing.
    What drives it from elsewhere -- a constraint's target, a driver -- is not
    in it: Check Again catches that.
    """
    if obj is None or depth > 16:
        return ()
    from .ik import own_fcurve_containers

    keys = []
    for container in own_fcurve_containers(obj) or ():
        for curve in container:
            points = tuple(
                (round(float(p.co[0]), 3), round(float(p.co[1]), 6)) for p in curve.keyframe_points
            )
            keys.append((curve.data_path, int(curve.array_index), points))
    if keys:
        placed = tuple(sorted(keys))
    else:
        local = obj.matrix_parent_inverse @ obj.matrix_basis
        placed = tuple(round(float(v), 6) for v in _flatten(local))
    constraints = tuple(
        (c.type, getattr(getattr(c, "target", None), "name", "")) for c in obj.constraints
    )
    return (obj.name, placed, constraints, fingerprint(obj.parent, depth + 1))


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
                self.references[reference.name] = reference
        self._at: dict[int, dict[str, Matrix]] = {}
        frames = sorted(set(int(f) for f in frames))
        if self.references and frames:
            self._read(frames, scene or bpy.context.scene)

    def _read(self, frames, scene) -> None:
        from .. import handlers

        original = scene.frame_current
        try:
            with handlers.suspended():
                for frame in frames:
                    scene.frame_set(frame)
                    self._at[frame] = {
                        name: in_rig(self.rig, obj) for name, obj in self.references.items()
                    }
        finally:
            scene.frame_set(original)

    @property
    def relative(self) -> bool:
        """Whether any waypoint is relative to an object."""
        return bool(self.references)

    def frame_of(self, reference, frame: int | None) -> Matrix:
        """``reference``'s pose in rig space on ``frame``: as the scene stands, for None."""
        if reference is None:
            return Matrix.Identity(4)
        at = self._at.get(int(frame)) if frame is not None else None
        if at is not None and reference.name in at:
            return at[reference.name].copy()
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
