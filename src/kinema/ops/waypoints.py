"""Teach a robot a job: record poses, put them on the timeline, generate motion.

Kinema could already rig a robot and solve for it. What it could not do was
say what the robot should *do* -- the only motion authoring was keyframing the
IK target by hand, which is the same workflow as animating any Empty, and so
nothing about it was robot-shaped.

This is the workflow every robot person already knows. Teach a pose, name it,
give it a time, and say how to arrive: joint-space for a fast reposition,
linear for a move whose tool path matters.

**Time is the ordering.** A waypoint carries a frame, not a list index, so
there is no second order to keep in sync and the list is only ever a view onto
the job in time order. Generation writes ordinary keyframes on ordinary
channels, which means Blender's own graph editor shapes the result afterwards
-- and ``kinema.bake_ik`` still turns the whole thing into plain joint curves
that render with the add-on gone.

Those generated keys are output, though, not the waypoints themselves: dragging
them retimes the animation until the next regeneration writes the stored frames
again. Making a waypoint draggable on the timeline -- a real marker, synced both
ways -- is a separate piece of work and is not here yet.

What each move type keys, and why:

*JOINT* keys the joint channels at both ends and switches live IK off for the
span. Joint-space interpolation is what a MoveJ *is*, and it is the cheapest,
most reliable way to cross a workspace.

*LINEAR* keys the IK target's transform with LINEAR interpolation and switches
live IK on. The solver then tracks the target every frame, and the tool travels
the straight line between the two taught poses. Measured on the 6-DoF fixture,
the tool stays on that line to 0.000 mm. The target's rotation turns the short
way, and a long turn gets keys along the way (see :data:`MAX_KEYED_TURN`).

A linear move also keys the joints at both ends, at the configurations they
were taught in. IK overrides those on every frame of the move; what they do is
seed it, so that the arm solves the line in the configuration it was taught in
and not in whatever a neighbouring key happens to hold. A straight line cannot
change configuration, so a linear move whose two ends were taught in different
ones is refused before anything is keyed, as a robot controller would refuse
it. Nor can it change a joint's turns: an end taught a whole turn from where
the line arrives is the same pose, and keeps the turn the line arrives with.

Both taught values are stored: the tool pose *and* the joint vector. The pose
is the portable truth -- it can be replayed on a different robot -- while the
joint vector pins the configuration, which a pose alone cannot, since a pose
has up to eight solutions and nothing else records which one was taught.

The marker Record leaves at the tool is the waypoint. Drag it, and the next
generation re-aims every move to and from it, joint moves included: the joint
vector is walked from the taught one to the marker, so it keeps its
configuration, and both are stored as Update would store them.

Generating ends with a check of the job as it plays: each move's distance from
its line, joint limits, speeds and jumps. It is kept on the rig and shown under
the waypoints; ``rig/motion_check.py`` has what is measured.
"""

from __future__ import annotations

import hashlib
import math

import bpy
import numpy as np
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    IntVectorProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Operator, PropertyGroup
from mathutils import Matrix

from ..rig import builder, motion_check
from ..rig.waypoints import MOVE_JOINT, MOVE_LINEAR, WaypointError, spans
from ..solver import branches, manager, numpy_backend
from ..solver.chain import chain_from_rig
from ..ui.panel import active_rig
from .ik import key_joint_value, own_fcurve_containers
from .velocity import acceleration_limit_of, joint_value, limit_of

#: Empty display size, as a fraction of the robot's reach. Waypoints are read
#: at the scale of the robot, not the scene.
MARKER_SCALE = 0.08

#: The largest turn keyed as one piece of a linear move, in degrees. Blender
#: interpolates a quaternion's four channels independently and normalises the
#: result. That keeps to the shortest turn's path, but not to its pace, and the
#: pace drifts further the longer the turn. Past this, extra keys go in along
#: the way. At 30 degrees the drift is under 0.05 degrees.
MAX_KEYED_TURN = 30.0

#: How finely a linear move is followed when checking that it keeps its
#: configuration: at most this many degrees of turn, or this many metres of
#: travel, per step. Stepping by frames alone would take a short, fast move
#: in strides long enough to jump branches that playback would never jump.
WALK_TURN_STEP = 2.0
WALK_DISTANCE_STEP = 0.005
WALK_MAX_STEPS = 1000

#: Blender caps a FloatVectorProperty at 32, which is the widest joint vector a
#: waypoint can hold. Every industrial arm is far under it -- a 6-axis robot on
#: a rail and a gripper is nine -- but a large humanoid is not, and truncating
#: one silently would store a configuration that restores to the wrong pose.
MAX_STORED_DOF = 32

#: Marks an Empty as belonging to a waypoint, so a stray Empty in the scene is
#: never mistaken for one and deleting a row only deletes what Kinema made.
PROP_WAYPOINT = "kinema_waypoint"


def _joint_values(rig) -> list[float]:
    """The rig's current joint vector, read off the driving channels."""
    values = []
    for pose_bone in builder.joint_bones(rig):
        is_prismatic = (
            pose_bone.bone.get(builder.PROP_JOINT_TYPE, "revolute") == "prismatic"
        )
        values.append(
            pose_bone.location[1] if is_prismatic else pose_bone.rotation_euler[1]
        )
    return values


def _apply_joint_values(rig, values) -> None:
    for pose_bone, value in zip(builder.joint_bones(rig), values, strict=False):
        is_prismatic = (
            pose_bone.bone.get(builder.PROP_JOINT_TYPE, "revolute") == "prismatic"
        )
        if is_prismatic:
            pose_bone.location[1] = float(value)
        else:
            pose_bone.rotation_euler[1] = float(value)


def _tool_bone(rig) -> str:
    return rig.get(builder.PROP_TCP_BONE) or builder.TCP_BONE


def ik_bone_name(rig) -> str | None:
    """This rig's usable IK goal, or None.

    Both the property *and* the bone: deleting the IK bone by hand leaves the
    property behind, and a truthy check on it alone would have the panel offer
    a linear move that the operator then refuses. Shared so the two cannot
    drift apart again.
    """
    name = rig.get(builder.PROP_IK_BONE) if rig is not None else None
    return name if name and name in rig.pose.bones else None


def _tool_matrix(rig) -> Matrix | None:
    """Where the tool is now, in the rig's own space."""
    name = _tool_bone(rig)
    bone = rig.pose.bones.get(name)
    return bone.matrix.copy() if bone is not None else None


def _flatten(matrix) -> list[float]:
    return [float(matrix[row][col]) for row in range(4) for col in range(4)]


def _unflatten(values) -> Matrix:
    return Matrix(
        [[float(values[row * 4 + col]) for col in range(4)] for row in range(4)]
    )


# --------------------------------------------------------------------------
# the stored waypoint
# --------------------------------------------------------------------------
class KinemaWaypoint(PropertyGroup):
    """One taught pose, and when the robot should be there."""

    name: StringProperty(
        name="Name",
        description="What this point is for: home, approach, pick, drop",
        default="waypoint",
    )
    frame: IntProperty(
        name="Frame",
        description=(
            "When the robot should be at this waypoint. This is what orders "
            "the job -- there is no separate list order"
        ),
        default=1,
    )
    move: EnumProperty(
        name="Move",
        description="How the robot arrives here from the previous waypoint",
        items=[
            (
                MOVE_JOINT,
                "Joint",
                "Interpolate the joints. Fast and always reachable, but the "
                "tool takes whatever path falls out",
            ),
            (
                MOVE_LINEAR,
                "Linear",
                "Drive the tool in a straight line. Predictable, and what a "
                "process move needs, but it can run into limits",
            ),
        ],
        default=MOVE_JOINT,
    )
    #: Tool pose in rig space, row-major. The portable half: a pose can be
    #: replayed on another robot, a joint vector cannot.
    pose: FloatVectorProperty(name="Pose", size=16, default=[0.0] * 16)
    #: Joint vector as taught. Pins the configuration a pose cannot.
    q: FloatVectorProperty(
        name="Joints", size=MAX_STORED_DOF, default=[0.0] * MAX_STORED_DOF
    )
    dof: IntProperty(name="DoF", default=0)
    marker: PointerProperty(
        name="Marker",
        type=bpy.types.Object,
        description="The Empty showing this waypoint in the viewport",
    )


class KinemaWaypointOperator(Operator):
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return active_rig(context) is not None


# --------------------------------------------------------------------------
# teaching
# --------------------------------------------------------------------------
def _marker_size(rig) -> float:
    """Scaled to the robot, so a KR120's markers are not the size of a UR's."""
    extent = max(rig.dimensions) if any(rig.dimensions) else 1.0
    return max(extent * MARKER_SCALE, 0.01)


def _make_marker(rig, waypoint, matrix: Matrix):
    """An Empty at the taught pose, parented to the rig so it travels with it.

    An Empty rather than panel-only data because issue #3 asks for waypoints
    taken from scene features -- snapped to a vertex on the part being welded,
    say. That only works if there is something in the viewport to snap.
    """
    empty = bpy.data.objects.new(f"{rig.name}.{waypoint.name}", None)
    empty.empty_display_type = "ARROWS"
    empty.empty_display_size = _marker_size(rig)
    empty[PROP_WAYPOINT] = waypoint.name
    collection = rig.users_collection[0] if rig.users_collection else None
    (collection or bpy.context.scene.collection).objects.link(empty)
    empty.parent = rig
    empty.matrix_parent_inverse = Matrix.Identity(4)
    empty.matrix_basis = matrix
    return empty


def pose_of(waypoint) -> Matrix:
    """Where this waypoint *is*, which is its marker if it still has one.

    The marker is not a read-out. Recording one is what makes "snap it to a
    feature on the part" possible at all -- issue #3's actual request -- and
    that is only true if moving it moves the waypoint. So the Empty's own
    transform wins over the pose stored at teaching time, and dragging it in
    the viewport re-aims every move to and from it: see :func:`follow_markers`.

    Its ``matrix_basis`` *is* the pose in rig space: the marker is parented to
    the rig with an identity parent inverse, exactly so these two are the same
    number and no conversion can drift between them.
    """
    marker = waypoint.marker
    if marker is not None and marker.parent is not None:
        return marker.matrix_basis.copy()
    return _unflatten(waypoint.pose)


def marker_has_moved(waypoint, tolerance: float = 1e-6) -> bool:
    """True if the marker no longer agrees with the pose stored for it.

    The stored joint vector was taught at that pose, so it is stale too until
    :func:`follow_markers` walks it to the marker and the pose is stored again.
    """
    marker = waypoint.marker
    if marker is None or marker.parent is None:
        return False
    stored = _unflatten(waypoint.pose)
    current = marker.matrix_basis
    return any(
        abs(current[row][col] - stored[row][col]) > tolerance
        for row in range(4)
        for col in range(4)
    )


def _record(rig, waypoint) -> str | None:
    """Store the rig's current pose and configuration. Returns an error, or None."""
    matrix = _tool_matrix(rig)
    if matrix is None:
        return "This rig has no TCP; create one first"

    values = _joint_values(rig)
    if len(values) > MAX_STORED_DOF:
        # Refused rather than truncated: a partly stored configuration would
        # restore to a pose the robot was never taught.
        return (
            f"This rig has {len(values)} joints; waypoints can store at most "
            f"{MAX_STORED_DOF}"
        )

    waypoint.pose = _flatten(matrix)
    waypoint.dof = len(values)
    waypoint.q = list(values) + [0.0] * (MAX_STORED_DOF - len(values))
    return None


class KINEMA_OT_add_waypoint(KinemaWaypointOperator):
    bl_idname = "kinema.add_waypoint"
    bl_label = "Record Waypoint"
    bl_description = (
        "Record the robot's current pose as a waypoint on the current frame, "
        "storing both the tool pose and the joint configuration"
    )

    name: StringProperty(name="Name", default="", options={"SKIP_SAVE"})

    def execute(self, context: bpy.types.Context) -> set[str]:
        rig = active_rig(context)
        if not builder.joint_bones(rig):
            self.report({"ERROR"}, "This rig has no joints to record")
            return {"CANCELLED"}

        context.view_layer.update()
        waypoint = rig.kinema_waypoints.add()
        waypoint.name = self.name or f"point{len(rig.kinema_waypoints)}"
        waypoint.frame = context.scene.frame_current
        # Joint by default: it is the move that always works. Linear is a
        # deliberate choice about a tool path, so it should be one.
        waypoint.move = MOVE_JOINT

        error = _record(rig, waypoint)
        if error is not None:
            rig.kinema_waypoints.remove(len(rig.kinema_waypoints) - 1)
            self.report({"ERROR"}, error)
            return {"CANCELLED"}

        waypoint.marker = _make_marker(rig, waypoint, _unflatten(waypoint.pose))
        rig.kinema_active_waypoint = len(rig.kinema_waypoints) - 1
        self.report(
            {"INFO"}, f"Recorded '{waypoint.name}' at frame {waypoint.frame}"
        )
        return {"FINISHED"}


class KINEMA_OT_update_waypoint(KinemaWaypointOperator):
    bl_idname = "kinema.update_waypoint"
    bl_label = "Update Waypoint"
    bl_description = "Re-record this waypoint from the robot's current pose"

    index: IntProperty(name="Index", default=-1, options={"SKIP_SAVE"})

    def execute(self, context: bpy.types.Context) -> set[str]:
        rig = active_rig(context)
        waypoint = _waypoint_at(rig, self.index)
        if waypoint is None:
            self.report({"ERROR"}, "No waypoint to update")
            return {"CANCELLED"}

        context.view_layer.update()
        error = _record(rig, waypoint)
        if error is not None:
            self.report({"ERROR"}, error)
            return {"CANCELLED"}
        if waypoint.marker is not None:
            waypoint.marker.matrix_basis = _unflatten(waypoint.pose)
        self.report({"INFO"}, f"'{waypoint.name}' updated from the current pose")
        return {"FINISHED"}


class KINEMA_OT_goto_waypoint(KinemaWaypointOperator):
    bl_idname = "kinema.goto_waypoint"
    bl_label = "Go To Waypoint"
    bl_description = (
        "Jump to this waypoint's frame and restore the exact configuration it "
        "was taught in, rather than whichever solution the solver picks today"
    )

    index: IntProperty(name="Index", default=-1, options={"SKIP_SAVE"})

    def execute(self, context: bpy.types.Context) -> set[str]:
        rig = active_rig(context)
        waypoint = _waypoint_at(rig, self.index)
        if waypoint is None:
            self.report({"ERROR"}, "No waypoint to go to")
            return {"CANCELLED"}

        context.scene.frame_set(int(waypoint.frame))
        # The stored joint vector, not the pose: a pose has up to eight
        # solutions and re-solving would be free to pick a different one.
        _apply_joint_values(rig, list(waypoint.q)[: waypoint.dof])
        # And put the goal where the tool now is, so live IK does not
        # immediately drag the arm back off the configuration just restored.
        ik_name = rig.get(builder.PROP_IK_BONE)
        context.view_layer.update()
        if ik_name and ik_name in rig.pose.bones:
            rig.pose.bones[ik_name].matrix = pose_of(waypoint)
        context.view_layer.update()
        self.report({"INFO"}, f"At '{waypoint.name}', frame {waypoint.frame}")
        return {"FINISHED"}


class KINEMA_OT_remove_waypoint(KinemaWaypointOperator):
    bl_idname = "kinema.remove_waypoint"
    bl_label = "Remove Waypoint"
    bl_description = "Delete this waypoint and its marker"

    index: IntProperty(name="Index", default=-1, options={"SKIP_SAVE"})

    def execute(self, context: bpy.types.Context) -> set[str]:
        rig = active_rig(context)
        index = self.index if self.index >= 0 else rig.kinema_active_waypoint
        if not 0 <= index < len(rig.kinema_waypoints):
            self.report({"ERROR"}, "No waypoint to remove")
            return {"CANCELLED"}

        waypoint = rig.kinema_waypoints[index]
        name = waypoint.name
        # The row owns its marker, the way an attachment row owns its copy.
        marker = waypoint.marker
        if marker is not None and PROP_WAYPOINT in marker:
            bpy.data.objects.remove(marker, do_unlink=True)
        rig.kinema_waypoints.remove(index)
        # max(0, ...): removing the last row leaves an empty list, and the
        # index would otherwise go to -1 and rely on the property's own min to
        # catch it.
        rig.kinema_active_waypoint = max(0, min(index, len(rig.kinema_waypoints) - 1))
        self.report({"INFO"}, f"Removed '{name}'")
        return {"FINISHED"}


def _waypoint_at(rig, index: int):
    if index < 0:
        index = rig.kinema_active_waypoint
    if 0 <= index < len(rig.kinema_waypoints):
        return rig.kinema_waypoints[index]
    return None


# --------------------------------------------------------------------------
# generating the motion
# --------------------------------------------------------------------------
class KINEMA_OT_generate_motion(KinemaWaypointOperator):
    bl_idname = "kinema.generate_motion"
    bl_label = "Generate Motion"
    bl_description = (
        "Key the robot through its waypoints in frame order: joint channels "
        "for joint moves, the IK target for linear ones"
    )

    def execute(self, context: bpy.types.Context) -> set[str]:
        rig = active_rig(context)
        try:
            plan = spans(rig.kinema_waypoints)
        except WaypointError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        needs_ik = any(span.move == MOVE_LINEAR for span in plan)
        ik_name = ik_bone_name(rig)
        if needs_ik and ik_name is None:
            self.report(
                {"ERROR"},
                "Linear moves need an IK target; add one, or set those "
                "waypoints to Joint",
            )
            return {"CANCELLED"}
        # The marker is the waypoint: one dragged since teaching takes every
        # move to and from it along. Its joints are found there before anything
        # is checked or written, and the checks below see them.
        unreachable, followed = follow_markers(rig, plan)
        if unreachable:
            names = ", ".join(f"'{name}'" for name in unreachable)
            self.report(
                {"ERROR"},
                f"{names}: the robot can't follow the marker there from the "
                f"configuration it was taught in. Move the marker back, or pose "
                f"the robot there and press Update",
            )
            return {"CANCELLED"}
        overrides = {waypoint.as_pointer(): values for waypoint, values in followed}
        refused, turned = (
            settle_configurations(rig, plan, overrides) if needs_ik else ([], [])
        )
        if refused:
            names = ", ".join(f"'{name}'" for name in refused)
            self.report(
                {"ERROR"},
                f"{names}: taught in a different configuration from the waypoint "
                f"before, and a linear move can't change configuration. Make it a "
                f"joint move, or re-teach one of the two",
            )
            return {"CANCELLED"}
        # Only now, with nothing refused: a cancelled operator leaves no undo
        # step, so a change made before a refusal could not be taken back.
        # Stored as Update stores a pose, so the waypoint *is* where its marker
        # is: Go To restores it there, and the next generation starts from it.
        for waypoint, values in followed:
            waypoint.q = values
            waypoint.pose = _flatten(waypoint.marker.matrix_basis)
        for waypoint, values, _ in turned:
            waypoint.q = values

        first, last = plan[0].start_frame, plan[-1].end_frame
        # A linear finish leaves a switch-off key one frame past the end, so
        # the job owns that frame too -- but only then. Claiming it after a
        # joint finish, where nothing is written there, would have the next
        # regeneration delete a key the user had put on the first free frame.
        ends_linear = plan[-1].move == MOVE_LINEAR
        owned = (first, last + 1 if ends_linear else last)
        # Union with what the last generation covered, which the new plan need
        # not still reach: shortening a job by moving its final waypoint
        # earlier has to take the old keys with it, or the robot goes on
        # visiting a frame no waypoint claims any more.
        clear_from, clear_to = _union(_generated_range(rig), owned)

        original = context.scene.frame_current
        # Suspended for the same reason baking suspends: writing keys walks the
        # frame, and the live handler would solve each one on the way past and
        # fight the values being written.
        with _handlers_suspended():
            _clear_generated(rig, ik_name, clear_from, clear_to)
            # The target's last keyed rotation, so the next is keyed in the same
            # hemisphere: see _key_linear_span.
            previous = None
            for span in plan:
                if span.move == MOVE_JOINT:
                    self._key_joint_span(rig, span)
                else:
                    previous = self._key_linear_span(rig, span, ik_name, previous)
            if ends_linear:
                # Leave the switch off past the end. Without this the last
                # linear span's "on" key is the final one on the channel, so
                # the solver goes on running for every frame after the job --
                # overwriting any joint animation out there, and doing the work
                # to no purpose while the user scrubs. Written on the frame
                # `owned` claims above; the two are one decision.
                rig.kinema_ik_enabled = False
                rig.keyframe_insert(data_path="kinema_ik_enabled", frame=owned[1])
            _set_linear_interpolation(rig, ik_name)
        rig.kinema_generated_range = owned
        context.scene.frame_set(original)
        checks = check_job(context, rig, plan)
        # Kept on the move's row too, so it is still there once the report has
        # gone: anyone who meant the taught turn has to re-teach that waypoint.
        notes = {
            int(waypoint.frame): f"{description}: kept as the line arrives"
            for waypoint, _, description in turned
        }
        for check in checks or ():
            check.turn_note = notes.get(check.end_frame, "")
        flagged = store_check(rig, checks)

        message = (
            f"Generated {len(plan)} move{'s' if len(plan) != 1 else ''} "
            f"over frames {first}-{last}"
        )
        if followed:
            # Said, since it rewrites what was taught there.
            names = ", ".join(f"'{waypoint.name}'" for waypoint, _ in followed)
            several = len(followed) != 1
            message += (
                f", with {names} re-solved at {'their' if several else 'its'} "
                f"moved marker{'s' if several else ''}"
            )
        if checks is None:
            message += "; not checked, since this rig has no TCP to measure it by"
        elif flagged:
            message += f"; {_flagged_text(flagged)} in the motion check"
        for waypoint, _, description in turned:
            message += (
                f". '{waypoint.name}' was taught a whole turn from where the line "
                f"arrives, and keeps the turn it arrives with ({description}). "
                f"Re-teach it to change that"
            )
        self.report({"WARNING"} if flagged else {"INFO"}, message)
        return {"FINISHED"}

    @staticmethod
    def _key_joint_span(rig, span) -> None:
        """Key the joints at both ends and switch live IK off for the span.

        The joints are what carry a MoveJ, so the solver must not be writing
        over them while it runs.
        """
        rig.kinema_ik_enabled = False
        rig.keyframe_insert(data_path="kinema_ik_enabled", frame=span.start_frame)
        for waypoint, frame in (
            (span.start, span.start_frame),
            (span.end, span.end_frame),
        ):
            _apply_joint_values(rig, list(waypoint.q)[: waypoint.dof])
            for pose_bone in builder.joint_bones(rig):
                key_joint_value(pose_bone, frame)

    @staticmethod
    def _key_linear_span(rig, span, ik_name: str, previous):
        """Key the IK goal along the move and switch live IK on for the span.

        The solver tracks the goal every frame, so keying the goal's own
        transform LINEAR is what makes the tool travel a straight line: the
        goal moves along it, and the arm follows.

        Each rotation is keyed in the same hemisphere as the one before it
        (``previous``, returned updated for the next move). A pose converts to
        either of two opposite quaternions, and Blender interpolates the four
        channels independently, so a pair keyed from opposite hemispheres turns
        the tool the long way round -- 340 degrees for a 20 degree move.

        The joints are keyed at both ends as well, at the configurations those
        waypoints were taught in. IK overrides them on every frame, but it
        solves *from* them: without these keys it would seed from whatever the
        neighbouring keys hold, and solve the line in that configuration.
        """
        rig.kinema_ik_enabled = True
        rig.keyframe_insert(data_path="kinema_ik_enabled", frame=span.start_frame)
        goal = rig.pose.bones[ik_name]
        goal.rotation_mode = "QUATERNION"
        start, end = pose_of(span.start), pose_of(span.end)
        for frame in _linear_key_frames(span, start, end):
            goal.matrix = _between(start, end, (frame - span.start_frame) / span.frames)
            if previous is not None:
                rotation = goal.rotation_quaternion.copy()
                rotation.make_compatible(previous)
                goal.rotation_quaternion = rotation
            previous = goal.rotation_quaternion.copy()
            goal.keyframe_insert(data_path="location", frame=frame)
            goal.keyframe_insert(data_path="rotation_quaternion", frame=frame)

        for waypoint, frame in (
            (span.start, span.start_frame),
            (span.end, span.end_frame),
        ):
            _apply_joint_values(rig, list(waypoint.q)[: waypoint.dof])
            for pose_bone in builder.joint_bones(rig):
                key_joint_value(pose_bone, frame)
        return previous


def _turn(start: Matrix, end: Matrix) -> float:
    """The angle between two poses' orientations, the short way, in degrees."""
    angle = start.to_quaternion().rotation_difference(end.to_quaternion()).angle
    return math.degrees(min(angle, 2.0 * math.pi - angle))


def _between(start: Matrix, end: Matrix, fraction: float) -> Matrix:
    """The pose ``fraction`` of the way along a linear move.

    On the straight line between the two positions, turned the short way
    between the two orientations.
    """
    first, last = start.to_quaternion(), end.to_quaternion()
    last.make_compatible(first)
    pose = first.slerp(last, fraction).to_matrix().to_4x4()
    pose.translation = start.translation.lerp(end.translation, fraction)
    return pose


def _linear_key_frames(span, start: Matrix, end: Matrix) -> list[int]:
    """The frames a linear move keys its target on: both ends, and enough between.

    Enough that no piece turns further than :data:`MAX_KEYED_TURN`, where the
    move has the frames to spare. Whole frames, spread evenly, so every key
    sits where the line says it should be at that frame and the pace along the
    line stays even.
    """
    pieces = max(1, math.ceil(_turn(start, end) / MAX_KEYED_TURN))
    pieces = min(pieces, max(span.frames, 1))
    return sorted(
        {span.start_frame + round(k * span.frames / pieces) for k in range(pieces + 1)}
    )


# --------------------------------------------------------------------------
# a moved marker re-aims its waypoint
# --------------------------------------------------------------------------
def follow_markers(rig, plan):
    """The joint values each waypoint whose marker has been dragged takes there.

    Returns ``(refused, followed)`` and writes nothing, so a refusal leaves the
    waypoints exactly as they were:

    - ``followed``: ``(waypoint, joint values)`` for every waypoint in the job
      whose marker has moved since it was taught. The values are walked there
      from the taught ones, a few millimetres and degrees at a time, so they
      keep the configuration taught: the same elbow, wrist and turns.
    - ``refused``: the names of those the walk cannot take there -- out of
      reach, or taught with a different set of joints than the rig has now.

    The marker is the waypoint (:func:`pose_of`), so once these are stored
    every move to and from it follows: a joint move replays them, and a linear
    move keys them as IK's seed. Keying the values taught at the marker's old
    place instead left a joint move there, and had the robot jump back wherever
    live IK hands a linear move over to the joint keys.
    """
    moved, seen = [], set()
    for span in plan:
        for waypoint in (span.start, span.end):
            if waypoint.as_pointer() not in seen and marker_has_moved(waypoint):
                seen.add(waypoint.as_pointer())
                moved.append(waypoint)
    if not moved:
        return [], []
    # Built here rather than asked of the solver: a job of joint moves needs no
    # IK target, and the solver is only made for one.
    chain = chain_from_rig(rig, _tool_bone(rig))
    if chain is None:
        return [waypoint.name for waypoint in moved], []

    names = [pose_bone.name for pose_bone in builder.joint_bones(rig)]
    columns = [names.index(name) for name in chain.bone_names]
    held = manager.held_mask(rig, chain)
    to_solver = np.linalg.inv(manager.root_pose(rig))

    refused, followed = [], []
    for waypoint in moved:
        values = None
        if waypoint.dof == len(names):
            values = _walk_to_marker(chain, held, to_solver, columns, waypoint)
        if values is None:
            refused.append(waypoint.name)
        else:
            followed.append((waypoint, values))
    return refused, followed


def _walk_to_marker(chain, held, to_solver, columns, waypoint) -> list[float] | None:
    """A waypoint's stored joint values, walked from its taught pose to its marker.

    Each step is seeded from the last, as a linear move's configuration check
    walks its line: one solve from the taught values straight to a marker
    dragged a long way could land in any configuration that reaches it. Held
    joints keep their taught values. None if a step cannot be solved.
    """
    values = np.array(list(waypoint.q), dtype=float)
    start, end = _unflatten(waypoint.pose), pose_of(waypoint)
    q = values[columns]
    for fraction in _walk_fractions(start, end):
        goal = to_solver @ np.array(_between(start, end, fraction), dtype=float)
        result = numpy_backend.solve(chain, q, goal, held=held)
        if not result.converged:
            return None
        q = result.q
    values[columns] = q
    return values.tolist()


# --------------------------------------------------------------------------
# a linear move keeps its configuration
# --------------------------------------------------------------------------
#: How a linear move's end compares with the configuration it was taught in.
SAME, TURNS, BRANCH = "SAME", "TURNS", "BRANCH"


def settle_configurations(rig, plan, overrides=None):
    """Check every linear move's end against the configuration it was taught in.

    ``overrides`` maps a waypoint's ``as_pointer()`` to joint values to use in
    place of its stored ones: those :func:`follow_markers` found for it, which
    are only written once nothing has been refused.

    Returns ``(refused, turned)`` and writes nothing, so a refusal leaves the
    waypoints exactly as they were:

    - ``refused``: the linear moves whose two ends were taught in different
      configurations, named by the waypoint each arrives at. A straight line
      cannot change configuration, so these are not generated.
    - ``turned``: ``(waypoint, joint values, description)`` for each linear
      move whose end was taught with a joint a whole turn from where the line
      arrives. That is the same pose, and a linear move keeps the turns it
      starts with -- a KUKA LIN ignores its target's Turn the same way -- so
      the waypoint takes the turns the line arrives with. Seeding IK from the
      taught turn instead would pull it over to that turn part-way along the
      line, in a single frame.

    Settled in time order, so a move that starts where an earlier one's turns
    were changed walks from the changed values.
    """
    solver = manager.get_solver(rig, ik_bone_name(rig))
    # A waypoint's pose is the TCP's. With the tip handed to another bone the
    # solver is aiming something else at it, and the walk would prove nothing.
    if solver is None or solver.tip_bone != _tool_bone(rig):
        return [], []
    chain = solver.chain
    refused, turned = [], []
    changed = dict(overrides or {})
    for span in plan:
        if span.move != MOVE_LINEAR:
            continue
        start = changed.get(span.start.as_pointer(), list(span.start.q))
        end = changed.get(span.end.as_pointer(), list(span.end.q))
        arrival = _arrival(rig, span, chain, start, end)
        if arrival is None:
            continue
        kind, values = arrival
        if kind == BRANCH:
            refused.append(span.end.name)
        elif kind == TURNS:
            changed[span.end.as_pointer()] = values
            turned.append((span.end, values, _turn_description(rig, chain, span, end, values)))
    return refused, turned


def _arrival(rig, span, chain, start_values, end_values):
    """Where a linear move arrives, against where its end was taught.

    The robot is walked along the line from the start's configuration, each
    step seeded from the last -- which is all a straight line can do. Where it
    arrives is compared with the configuration the end was taught in:

    - on a different branch, as Find Solutions tells branches apart: BRANCH;
    - on the same branch but with a joint whole turns away, which a comparison
      modulo a turn cannot see: TURNS, with the end's stored values carrying
      the turns the walk arrived with;
    - otherwise SAME.

    Solved with the NumPy backend, which needs no compile and answers the same
    question: can the arm follow this line without changing configuration?

    None whenever it cannot tell:
    - joints added or removed since teaching;
    - a redundant arm, where the end of a line is a whole family of
      configurations and which one the walk lands on says nothing about a
      branch;
    - a line the arm cannot follow at all, which the motion check reports in
      its own terms.
    """
    columns = _chain_columns(rig, chain, span)
    if columns is None:
        return None
    held = manager.held_mask(rig, chain)
    if chain.dof - int(held.sum()) > 6:
        return None

    q_start = np.array(start_values, dtype=float)[columns]
    q_end = np.array(end_values, dtype=float)[columns]
    to_solver = np.linalg.inv(manager.root_pose(rig))
    start, end = pose_of(span.start), pose_of(span.end)

    def solve(seed, pose: Matrix):
        goal = to_solver @ np.array(pose, dtype=float)
        return numpy_backend.solve(chain, seed, goal, held=held)

    # The end's configuration solved for where its marker is: the same thing
    # as the values given, which for a moved marker follow_markers has already
    # walked there.
    reference = solve(q_end, end)
    if not reference.converged:
        return None

    q = q_start
    for fraction in _walk_fractions(start, end, span.frames):
        seed = q.copy()
        # Held joints are not solved for; they go where their keys take them.
        seed[held] = q_start[held] + (q_end[held] - q_start[held]) * fraction
        result = solve(seed, _between(start, end, fraction))
        if not result.converged:
            return None
        q = result.q

    if (
        branches.joint_distance(q, reference.q, chain.is_revolute)
        >= branches.DISTINCT_TOLERANCE
    ):
        return BRANCH, None
    turns = np.where(chain.is_revolute, np.round((q - reference.q) / (2.0 * math.pi)), 0.0)
    if not turns.any():
        return SAME, None
    values = np.array(end_values, dtype=float)
    values[columns] = q_end + turns * 2.0 * math.pi
    return TURNS, values.tolist()


def _turn_description(rig, chain, span, taught, arrived) -> str:
    """Which joints a linear move's end keeps a different turn of, in words."""
    columns = _chain_columns(rig, chain, span) or []
    parts = [
        f"{name} at {math.degrees(arrived[column]):+.0f}°, "
        f"taught at {math.degrees(taught[column]):+.0f}°"
        for name, column in zip(chain.bone_names, columns, strict=True)
        if abs(arrived[column] - taught[column]) > math.pi
    ]
    return "; ".join(parts)


def _chain_columns(rig, chain, span) -> list[int] | None:
    """Where each of the chain's joints sits in a waypoint's stored vector.

    None if the waypoints were taught with a different set of joints than the
    rig has now, since their stored vectors no longer line up with its bones.
    """
    names = [pose_bone.name for pose_bone in builder.joint_bones(rig)]
    if span.start.dof != len(names) or span.end.dof != len(names):
        return None
    index = {name: i for i, name in enumerate(names)}
    if any(name not in index for name in chain.bone_names):
        return None
    return [index[name] for name in chain.bone_names]


def _walk_fractions(start: Matrix, end: Matrix, frames: int = 1) -> list[float]:
    """Where between two poses a walk solves: every frame, or finer on a fast move."""
    distance = (end.translation - start.translation).length
    steps = max(
        frames,
        math.ceil(_turn(start, end) / WALK_TURN_STEP),
        math.ceil(distance / WALK_DISTANCE_STEP),
        1,
    )
    steps = min(steps, WALK_MAX_STEPS)
    return [k / steps for k in range(1, steps + 1)]


# --------------------------------------------------------------------------
# the motion check
# --------------------------------------------------------------------------
class KinemaMoveCheck(PropertyGroup):
    """One move's findings from the last motion check. See rig/motion_check.py."""

    name: StringProperty(name="Waypoint")
    move: StringProperty(name="Move")
    start_frame: IntProperty(name="Start")
    end_frame: IntProperty(name="End")
    line_error: FloatProperty(name="Off the Line", default=-1.0, unit="LENGTH")
    twist_error: FloatProperty(name="Turned Off", default=-1.0, subtype="ANGLE")
    limit_joint: StringProperty(name="At a Limit")
    speed_joint: StringProperty(name="Fastest Joint")
    speed_ratio: FloatProperty(name="Speed Against Limit")
    accel_joint: StringProperty(name="Hardest-Accelerating Joint")
    accel_ratio: FloatProperty(name="Acceleration Against Limit")
    jump_joint: StringProperty(name="Largest Jump")
    jump: FloatProperty(name="Jump")
    turn_note: StringProperty(name="Turn Kept")
    jump_prismatic: BoolProperty(name="Jump Is Linear")
    jump_frame: IntProperty(name="Jump Frame")


def check_job(context, rig, plan) -> list[motion_check.MoveCheck]:
    """Play the job once, the way playback does, and measure every move.

    Each frame is set with the handlers live, so live IK solves it with the
    rig's own backend, from the rig's own keys -- whatever playback would
    show, including a compile if the solver needs one.
    """
    scene = context.scene
    bones = builder.joint_bones(rig)
    joints = [
        motion_check.Joint(
            name=pose_bone.name,
            prismatic=pose_bone.bone.get(builder.PROP_JOINT_TYPE, "revolute")
            == "prismatic",
            lower=_float_or_none(pose_bone.bone.get(builder.PROP_LOWER)),
            upper=_float_or_none(pose_bone.bone.get(builder.PROP_UPPER)),
            velocity=limit_of(pose_bone),
            acceleration=acceleration_limit_of(pose_bone),
        )
        for pose_bone in bones
    ]
    tool = _tool_bone(rig)
    if tool not in rig.pose.bones:
        # Taught with a TCP that has since been deleted: there is nothing to
        # measure a line against. None, not an empty list, so that no caller
        # can mistake a job it could not measure for one with nothing wrong.
        return None
    first, last = plan[0].start_frame, plan[-1].end_frame

    # A frame either side of the job as well. An acceleration is measured
    # from a frame's neighbours, and the job's own ends are where one that
    # sets off or stops at speed changes it all at once.
    samples = []
    original = scene.frame_current
    window = context.window_manager
    window.progress_begin(first - 1, last + 1)
    try:
        for frame in range(first - 1, last + 2):
            scene.frame_set(frame)
            context.view_layer.update()
            samples.append(
                motion_check.Sample(
                    frame=frame,
                    q=np.array([joint_value(pose_bone) for pose_bone in bones]),
                    tool=np.array(rig.pose.bones[tool].matrix, dtype=float),
                )
            )
            window.progress_update(frame)
    finally:
        window.progress_end()
        scene.frame_set(original)

    fps = scene.render.fps / scene.render.fps_base
    return [
        motion_check.check_move(
            span.end.name,
            span.move,
            span.start_frame,
            span.end_frame,
            np.array(pose_of(span.start), dtype=float),
            np.array(pose_of(span.end), dtype=float),
            samples,
            joints,
            fps,
        )
        for span in plan
    ]


def store_check(rig, checks) -> int:
    """Keep the findings on the rig. Returns how many moves have a problem.

    ``checks`` is None for a job that could not be measured, which leaves no
    rows rather than rows that look clean. The waypoints the rows describe
    are fingerprinted alongside them, so the panel can tell when they no
    longer match the job: see :func:`job_signature`.
    """
    rig.kinema_motion_check.clear()
    rig.kinema_motion_check_job = job_signature(rig) if checks is not None else ""
    for check in checks or ():
        item = rig.kinema_motion_check.add()
        for field in motion_check.MoveCheck.__dataclass_fields__:
            setattr(item, field, getattr(check, field))
    return sum(1 for check in checks or () if motion_check.problems(check))


def job_signature(rig) -> str:
    """A fingerprint of the waypoints as they stand: names, frames, moves, poses, joints.

    Stored with a motion check and compared by the panel, so findings about a
    job that has since been edited are marked as such instead of being shown
    as current. The keys are not in it -- editing a curve by hand is only
    caught by Check Again -- but everything Generate Motion reads is.
    """
    parts = []
    for waypoint in sorted(rig.kinema_waypoints, key=lambda w: int(w.frame)):
        pose = pose_of(waypoint)
        parts.append(
            (
                waypoint.name,
                int(waypoint.frame),
                waypoint.move,
                tuple(round(float(pose[r][c]), 6) for r in range(4) for c in range(4)),
                tuple(round(float(v), 6) for v in list(waypoint.q)[: waypoint.dof]),
            )
        )
    return hashlib.sha1(repr(parts).encode("utf-8")).hexdigest()


def _flagged_text(count: int) -> str:
    return f"{count} move{'s' if count != 1 else ''} flagged"


def _float_or_none(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class KINEMA_OT_check_motion(KinemaWaypointOperator):
    bl_idname = "kinema.check_motion"
    bl_label = "Check Motion"
    bl_description = (
        "Play the job through once, solving as playback does, and report each "
        "move: distance from its line, joint limits, speeds and jumps"
    )

    def execute(self, context: bpy.types.Context) -> set[str]:
        rig = active_rig(context)
        try:
            plan = spans(rig.kinema_waypoints)
        except WaypointError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        checks = check_job(context, rig, plan)
        flagged = store_check(rig, checks)
        if checks is None:
            self.report(
                {"WARNING"}, "Nothing checked: this rig has no TCP to measure the job by"
            )
            return {"CANCELLED"}
        message = f"Checked {len(plan)} move{'s' if len(plan) != 1 else ''}"
        if flagged:
            self.report({"WARNING"}, f"{message}; {_flagged_text(flagged)}")
        else:
            self.report({"INFO"}, f"{message}; nothing to report")
        return {"FINISHED"}


def _generated_paths(rig, ik_name: str | None) -> set[str]:
    """Data paths this operator owns, and is therefore allowed to clear."""
    paths = {"kinema_ik_enabled"}
    for pose_bone in builder.joint_bones(rig):
        paths.add(f'pose.bones["{pose_bone.name}"].rotation_euler')
        paths.add(f'pose.bones["{pose_bone.name}"].location')
    if ik_name:
        paths.add(f'pose.bones["{ik_name}"].location')
        paths.add(f'pose.bones["{ik_name}"].rotation_quaternion')
    return paths


def _generated_range(rig) -> tuple[int, int] | None:
    """The frames the last generation covered, or None if it never ran."""
    stored = getattr(rig, "kinema_generated_range", None)
    if stored is None:
        return None
    first, last = int(stored[0]), int(stored[1])
    return (first, last) if last >= first else None


def _union(a: tuple[int, int] | None, b: tuple[int, int]) -> tuple[int, int]:
    if a is None:
        return b
    return (min(a[0], b[0]), max(a[1], b[1]))


def _clear_generated(rig, ik_name: str | None, first: int, last: int) -> None:
    """Clear the keys a previous generation wrote, and only those.

    Regenerating has to start from an empty range or old keys survive between
    the new ones -- a waypoint moved from frame 40 to frame 30 would otherwise
    leave the robot visiting frame 40 as well.

    Bounded to frames rather than dropping whole curves, and that boundary is
    the point. The channels the generator writes are ordinary ones an animator
    may also be using: a hand-keyed pose before the job, or a tool held after
    it, lives on exactly these data paths, and removing the curve would take
    that with it. Only the span the generator claims is its to clear -- which
    is why the caller unions the new range with the previous one rather than
    passing the new one alone, and why that range stops at the last frame
    actually written rather than one beyond it.

    ``first`` and ``last`` are both inclusive.
    """
    wanted = _generated_paths(rig, ik_name)
    for container in own_fcurve_containers(rig):
        for curve in list(container):
            if curve.data_path not in wanted:
                continue
            for point in reversed(list(curve.keyframe_points)):
                if first <= point.co[0] <= last:
                    curve.keyframe_points.remove(point)
            # A curve emptied of every key animates nothing but still counts as
            # animation, which leaves the channel looking driven in the UI.
            if not len(curve.keyframe_points):
                container.remove(curve)
            else:
                curve.update()


def _set_linear_interpolation(rig, ik_name: str | None) -> None:
    """Make the IK goal's own curves LINEAR.

    Blender inserts keys with the user's default, Bezier out of the box, which
    would ease the goal in and out of every waypoint. Between two keys that
    still traces the same straight line -- x, y and z share the shape, so only
    the speed along it changes -- but it is the wrong *default*: a linear move
    should come out linear, and easing it is then something the user chooses in
    the graph editor rather than something they have to undo.
    """
    if not ik_name:
        return
    prefix = f'pose.bones["{ik_name}"]'
    for container in own_fcurve_containers(rig):
        for curve in container:
            if not curve.data_path.startswith(prefix):
                continue
            for point in curve.keyframe_points:
                point.interpolation = "LINEAR"
            curve.update()


def _handlers_suspended():
    from .. import handlers

    return handlers.suspended()


classes = (
    KinemaWaypoint,
    KinemaMoveCheck,
    KINEMA_OT_add_waypoint,
    KINEMA_OT_update_waypoint,
    KINEMA_OT_goto_waypoint,
    KINEMA_OT_remove_waypoint,
    KINEMA_OT_generate_motion,
    KINEMA_OT_check_motion,
)


def register_props() -> None:
    bpy.types.Object.kinema_waypoints = CollectionProperty(type=KinemaWaypoint)
    # The last motion check, one row per move. Kept on the rig so it survives a
    # save, and replaced whole by the next Generate Motion or Check Motion.
    bpy.types.Object.kinema_motion_check = CollectionProperty(type=KinemaMoveCheck)
    # The job those rows describe, as job_signature() saw it then.
    bpy.types.Object.kinema_motion_check_job = StringProperty(
        name="Checked Job",
        description="Fingerprint of the waypoints the motion check measured",
        default="",
    )
    # What the last generation covered, so the next one can clear its own
    # leavings even where the new job no longer reaches. Empty when last < first.
    bpy.types.Object.kinema_generated_range = IntVectorProperty(
        name="Generated Range",
        description="First and last frame the last Generate Motion wrote",
        size=2,
        default=(0, -1),
    )
    bpy.types.Object.kinema_active_waypoint = IntProperty(
        name="Active Waypoint",
        description="Row highlighted in the Waypoints list",
        default=0,
        min=0,
    )


def unregister_props() -> None:
    del bpy.types.Object.kinema_waypoints
    del bpy.types.Object.kinema_motion_check
    del bpy.types.Object.kinema_motion_check_job
    del bpy.types.Object.kinema_generated_range
    del bpy.types.Object.kinema_active_waypoint
