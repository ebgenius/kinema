"""Teach a robot a job: record poses, put them on the timeline, generate motion.

Kinema could already rig a robot and solve for it. What it could not do was
say what the robot should *do* -- the only motion authoring was keyframing the
IK target by hand, which is the same workflow as animating any Empty, and so
nothing about it was robot-shaped.

This is the workflow every robot person already knows. Teach a pose and name
it; then say how the robot moves between two poses, and when: joint-space for
a fast reposition, linear for a move whose tool path matters.

**A waypoint is a pose; a motion is a move between two, with its own frames.**
So one waypoint can end one motion and start the next, or be gone back to, and
a motion is retimed without touching the poses. Motions run in frame order, as
strips in the NLA do; ``rig/waypoints.py`` lays them out, gaps and all.
Recording on a later frame than the last waypoint adds a joint motion between
the two, so teaching a job pose by pose, frame by frame, works as it always
did. A file saved before motions, whose waypoints carried a frame and a move,
is converted when it opens (:func:`convert_saved_before_motions`).

Generation writes ordinary keyframes on ordinary channels, which means
Blender's own graph editor shapes the result afterwards -- and
``kinema.bake_ik`` still turns the whole thing into plain joint curves that
render with the add-on gone.

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
import uuid
from dataclasses import replace

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
from ..rig.waypoints import (
    MADE_HOLD,
    MOVE_JOINT,
    MOVE_LINEAR,
    MotionsOverlap,
    Plan,
    WaypointError,
    spans,
)
from ..rig.waypoints import (
    plan as plan_motions,
)
from ..solver import branches, manager, numpy_backend
from ..solver.chain import chain_from_rig
from ..ui.panel import active_rig
from . import collision as collision_ops
from . import references, strips, velocity
from .ik import key_joint_value, own_fcurve_containers
from .outcome import Outcome, reported
from .pose import apply_joint_values, bone_matrices, joint_values
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
#: What a motion's move can be, as its enum lists them.
MOVE_ITEMS = [
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
]

#: The shape of a rig's job: 0 for a file saved before motions, whose
#: waypoints carry a frame and a move, converted by
#: :func:`convert_saved_before_motions`.
JOB_VERSION = 1


def unique_name(rig, waypoint) -> None:
    """Give ``waypoint`` a name no other waypoint on ``rig`` has, as Blender names a copy.

    A motion's From and To are picked by name, so two waypoints of one name
    would leave one of them out of reach.
    """
    taken = {w.name for w in rig.kinema_waypoints if w.as_pointer() != waypoint.as_pointer()}
    if waypoint.name not in taken:
        return
    base, number = waypoint.name, 1
    while f"{base}.{number:03d}" in taken:
        number += 1
    waypoint.name = f"{base}.{number:03d}"


def _on_waypoint_named(self, _context) -> None:
    rig = self.id_data
    unique_name(rig, self)
    # The marker goes by the waypoint's name, in the Outliner as on the row.
    if self.marker is not None and PROP_WAYPOINT in self.marker:
        self.marker.name = f"{rig.name}.{self.name}"
        self.marker[PROP_WAYPOINT] = self.name


class KinemaWaypoint(PropertyGroup):
    """One taught pose: the tool's, and the joint configuration it was taught in."""

    name: StringProperty(
        name="Name",
        description="What this point is for: home, approach, pick, drop. One name per waypoint",
        default="waypoint",
        update=_on_waypoint_named,
    )
    #: What motions refer to this waypoint by: names change.
    uid: StringProperty(name="Id", default="")
    #: Read only to convert a file saved before motions, where the waypoint
    #: carried when the robot is there and how it arrives.
    frame: IntProperty(name="Frame (before motions)", default=1)
    move: EnumProperty(name="Move (before motions)", items=MOVE_ITEMS, default=MOVE_JOINT)
    #: Tool pose in the waypoint's frame of reference, row-major: rig space
    #: for the robot's base. The portable half: a pose can be replayed on
    #: another robot, a joint vector cannot.
    pose: FloatVectorProperty(name="Pose", size=16, default=[0.0] * 16)
    #: Taught relative to an object: where, in rig space, the joints were
    #: taught, which the pose no longer says. Zero at the base, where the pose
    #: does. See ops/references.py.
    taught: FloatVectorProperty(name="Taught At", size=16, default=[0.0] * 16)
    reference: PointerProperty(
        name="Relative To",
        type=bpy.types.Object,
        description=(
            "The object this waypoint is taught relative to: a workpiece or a "
            "fixture. It follows the object as it moves or is animated. Empty "
            "for the robot's base. Changing it keeps the waypoint where it is"
        ),
        poll=references.may_be_reference,
        update=references.on_reference_set,
    )
    #: The object ``pose`` is in, as Kinema last expressed it: what the field
    #: changed from, where there is no marker to say.
    pose_space: PointerProperty(name="Pose Space", type=bpy.types.Object)
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


def _named(which: str):
    """A motion's start or end as a waypoint's name, for picking it in the panel."""

    def get(self) -> str:
        waypoint = waypoint_by_uid(self.id_data, getattr(self, which))
        return waypoint.name if waypoint is not None else ""

    def set_(self, value: str) -> None:
        found = next((w for w in self.id_data.kinema_waypoints if w.name == value), None)
        setattr(self, which, found.uid if found is not None else "")

    return get, set_


def _duration(self) -> int:
    return int(self.frame_end) - int(self.frame_start)


class KinemaMotion(PropertyGroup):
    """One move between two waypoints, with its own frames, as an NLA strip has."""

    name: StringProperty(name="Name", default="motion")
    #: The waypoints' uids: see :class:`KinemaWaypoint`.
    start: StringProperty(name="Start Id", default="")
    end: StringProperty(name="End Id", default="")
    start_name: StringProperty(
        name="From", description="The waypoint this motion starts at",
        get=_named("start")[0], set=_named("start")[1],
    )
    end_name: StringProperty(
        name="To", description="The waypoint this motion ends at",
        get=_named("end")[0], set=_named("end")[1],
    )
    move: EnumProperty(
        name="Move", description="How the robot moves", items=MOVE_ITEMS, default=MOVE_JOINT
    )
    frame_start: IntProperty(
        name="Start", description="The frame the robot leaves the start waypoint", default=1
    )
    frame_end: IntProperty(
        name="End", description="The frame the robot is at the end waypoint", default=25
    )
    duration: IntProperty(
        name="Frames", description="How many frames the move takes", get=_duration
    )
    #: The frames this motion's NLA strip was laid out at, which are the frames
    #: of the job's action it shows: how a strip moved in the NLA editor is told
    #: from the others. See ops/strips.py.
    laid_frames: IntVectorProperty(size=2, default=(0, 0), options={"HIDDEN"})
    #: The frames this motion last had in step with its strip: laid out at, or
    #: taken from the strip since. Frames of its own other than these were set
    #: in the Motions list.
    synced_frames: IntVectorProperty(size=2, default=(0, 0), options={"HIDDEN"})


def waypoint_by_uid(rig, uid: str):
    """The waypoint ``uid`` names on ``rig``, or None."""
    if not uid:
        return None
    return next((w for w in rig.kinema_waypoints if w.uid == uid), None)


def _new_uid() -> str:
    return uuid.uuid4().hex


def convert_saved_before_motions(rig) -> bool:
    """Give a rig saved before motions the motions its waypoints made. True if it did.

    Its waypoints carried a frame and a move, the move belonging to the
    waypoint arrived at (``rig/waypoints.py``). Each pair in frame order
    becomes a motion with those frames and that move, named after the waypoint
    it arrives at, so the job plays as it did and its Motion Check rows keep
    their names. Recording goes on from its last waypoint.
    """
    if rig.kinema_job_version >= JOB_VERSION or rig.library is not None:
        # Linked from a library: what is written to it isn't saved with this
        # file, so converting it would be lost, and done again, on every open.
        # See job_plan.
        return False
    # The job as it was fingerprinted, before it changes shape: whatever was
    # stored of it then -- a Motion Check, what was generated or optimized --
    # still describes it, converted.
    was = _signature_before_motions(rig)
    for waypoint in rig.kinema_waypoints:
        if not waypoint.uid:
            waypoint.uid = _new_uid()
        unique_name(rig, waypoint)
    if not len(rig.kinema_motions) and len(rig.kinema_waypoints) > 1:
        try:
            old = spans(rig.kinema_waypoints)
        except WaypointError as exc:
            # Two on one frame: no job then either. Kept, so Generate Motion can
            # say why there are no motions now.
            old = []
            rig.kinema_unconverted = str(exc)
        for span in old:
            motion = rig.kinema_motions.add()
            motion.name = span.end.name
            motion.start, motion.end = span.start.uid, span.end.uid
            motion.move = span.move
            motion.frame_start, motion.frame_end = span.start_frame, span.end_frame
    if len(rig.kinema_waypoints):
        last = max(rig.kinema_waypoints, key=lambda waypoint: int(waypoint.frame))
        rig.kinema_last_recorded = last.uid
        rig.kinema_last_recorded_frame = int(last.frame)
    rig.kinema_job_version = JOB_VERSION
    now = job_signature(rig)
    for stored in ("kinema_motion_check_job", "kinema_generated_job", "kinema_optimized_job"):
        if getattr(rig, stored) == was:
            setattr(rig, stored, now)
    return True


def convert_all() -> None:
    """Convert each rig in the file saved before motions (:func:`convert_saved_before_motions`)."""
    for obj in bpy.data.objects:
        if builder.is_kinema_rig(obj) and hasattr(obj, "kinema_job_version"):
            convert_saved_before_motions(obj)


def job_plan(rig):
    """The rig's job, as ``rig/waypoints.plan`` lays out its motions.

    Converts a rig saved before motions first, so it is for operators, never
    for a panel's draw. One linked from a library isn't converted, since what
    is written to it isn't saved with the file, so its job is read from its
    waypoints' frames and moves, as it was saved.
    """
    convert_saved_before_motions(rig)
    for waypoint in rig.kinema_waypoints:
        refused = refusal(rig, waypoint)
        if refused is not None:
            raise WaypointError(refused)
    if rig.kinema_job_version < JOB_VERSION:
        return Plan(spans=spans(rig.kinema_waypoints), notes=[])
    notes = []
    if rig.library is None:
        # A marker parented elsewhere by hand moved its waypoint's frame of
        # reference, and a deleted object took its waypoints' with it: their
        # fields and stored records catch up here.
        for waypoint in rig.kinema_waypoints:
            note = references.sync(rig, waypoint)
            if note:
                notes.append(note)
    if len(rig.kinema_motions) and rig.kinema_unconverted and rig.library is None:
        # Motions made since say more than why the old job made none. Written
        # only then, and not to a rig linked from a library, where it isn't kept.
        rig.kinema_unconverted = ""
    if not len(rig.kinema_motions) and rig.kinema_unconverted:
        raise WaypointError(
            f"This job, saved before motions, couldn't be made into motions: "
            f"{rig.kinema_unconverted}. Add its motions in the Motions list"
        )
    # A motion's strip moved or scaled in the NLA editor retimes it. The job is
    # planned at the new frames, and the operator writes them once nothing will
    # refuse it: a cancelled operator leaves no undo step to take them back.
    retimed, retimes = strips.catch_up(rig)
    motions = list(rig.kinema_motions)
    for motion, frames, _ in retimes:
        motions[motions.index(motion)] = strips.Retimed(motion, frames)
    try:
        job = plan_motions(motions, rig.kinema_waypoints)
    except WaypointError as exc:
        if not retimes:
            raise
        moved = ", ".join(f"'{m.name}' to {a}-{b}" for m, (a, b), _ in retimes)
        raise type(exc)(f"{exc}, with {moved} as its NLA strip was moved") from exc
    job.notes[:0] = notes + retimed
    return replace(job, retimes=retimes)


def refusal(rig, waypoint) -> str | None:
    """Why no operator can place ``waypoint`` as its marker is parented, or None.

    Parented to what the robot carries, the tool would chase its own tail;
    parented to a bone or vertices of another object, its numbers are in a
    space a frame of reference, an object, doesn't have.
    """
    held = references.carried(rig, waypoint)
    if held is not None:
        if held == rig:
            where = f"the robot's bone '{waypoint.marker.parent_bone}'"
        else:
            where = f"'{held.name}', which the robot carries"
        return (
            f"'{waypoint.name}': its marker is parented to {where}, so the tool would "
            f"chase its own tail. Parent it to something else, or back to the robot"
        )
    # A marker whose parent was cleared by hand goes back to the base when it is
    # caught up (references.sync), so the object it was on no longer matters.
    marker = waypoint.marker
    cleared = marker is not None and marker.parent is None
    reference = references.reference_of(rig, waypoint)
    # The scene the operator runs in is the one the job is played in (place_job).
    scene = bpy.context.scene
    if reference is not None and not cleared and scene not in reference.users_scene:
        return (
            f"'{waypoint.name}': it is relative to '{reference.name}', which isn't in "
            f"the scene '{scene.name}', so the job can't follow it there. Link "
            f"'{reference.name}' into it, or choose another object"
        )
    part = references.on_a_part(rig, waypoint)
    if part is not None:
        if waypoint.marker.parent_type == "BONE":
            where, there = f"the bone '{waypoint.marker.parent_bone}' of", "the bone"
        else:
            where, there = "vertices of", "them"
        return (
            f"'{waypoint.name}': its marker is parented to {where} '{part.name}', and "
            f"a waypoint is relative to an object, not a part of one. Parent it to "
            f"'{part.name}' itself, or to an Empty parented to {there}"
        )
    return None


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


def _make_marker(rig, waypoint, matrix: Matrix, reference=None):
    """An Empty at the taught pose, parented to the waypoint's frame of reference:
    the rig, so it travels with it, or the object it is taught relative to.
    ``matrix`` is the pose in that frame.

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
    empty.parent = reference if reference is not None else rig
    empty.matrix_parent_inverse = Matrix.Identity(4)
    empty.matrix_basis = matrix
    return empty


def pose_of(waypoint, placement=None, frame: int | None = None) -> Matrix:
    """Where this waypoint *is*, in rig space, which is its marker if it still has one.

    On ``frame`` as ``placement`` read it, for one taught relative to an
    object; as the scene stands without them. See ops/references.py.

    The marker is not a read-out. Recording one is what makes "snap it to a
    feature on the part" possible at all -- issue #3's actual request -- and
    that is only true if moving it moves the waypoint. So the Empty's own
    transform wins over the pose stored at teaching time, and dragging it in
    the viewport re-aims every move to and from it: see :func:`follow_markers`.

    Its ``matrix_basis`` *is* the pose in its frame -- rig space, at the base:
    the marker is parented with an identity parent inverse, exactly so these
    two are the same number and no conversion can drift between them.
    """
    if placement is not None:
        return placement.pose(waypoint, frame)
    rig = waypoint.id_data
    reference = references.reference_of(rig, waypoint)
    return references.in_rig(rig, reference) @ references.local_pose(waypoint)


def marker_has_moved(waypoint, tolerance: float = 1e-6) -> bool:
    """True if the marker no longer agrees with the pose stored for it.

    The stored joint vector was taught at that pose, so it is stale too until
    :func:`follow_markers` walks it to the marker and the pose is stored again.
    """
    marker = waypoint.marker
    if marker is None or marker.parent is None:
        return False
    stored = _unflatten(waypoint.pose)
    current = references.local_pose(waypoint)
    return any(
        abs(current[row][col] - stored[row][col]) > tolerance
        for row in range(4)
        for col in range(4)
    )


def _record(rig, waypoint, reference=None, values=None) -> str | None:
    """Store the rig's current pose and configuration. Returns an error, or None.

    Relative to ``reference`` where given: the pose in its space, and where the
    joints were taught in rig space. At ``values``, a joint vector, where given:
    the pose the tool would have there, with the rig left where it stands.
    """
    matrix = _tool_matrix(rig)
    if matrix is None:
        return "This rig has no TCP; create one first"

    if values is None:
        values = joint_values(rig)
    else:
        values = [float(v) for v in values]
        matrix = Matrix(bone_matrices(rig, values)[_tool_bone(rig)].tolist())
    if len(values) > MAX_STORED_DOF:
        # Refused rather than truncated: a partly stored configuration would
        # restore to a pose the robot was never taught.
        return (
            f"This rig has {len(values)} joints; waypoints can store at most "
            f"{MAX_STORED_DOF}"
        )

    if reference is None:
        waypoint.pose = _flatten(matrix)
        waypoint.taught = [0.0] * 16
    else:
        try:
            into = references.in_rig(rig, reference).inverted()
        except ValueError:
            return f"'{reference.name}' is scaled to nothing, so nothing can be relative to it"
        waypoint.pose = _flatten(into @ matrix)
        waypoint.taught = _flatten(matrix)
    references.settle(waypoint, reference)
    waypoint.dof = len(values)
    waypoint.q = list(values) + [0.0] * (MAX_STORED_DOF - len(values))
    return None


class KINEMA_OT_add_waypoint(KinemaWaypointOperator):
    bl_idname = "kinema.add_waypoint"
    bl_label = "Record Waypoint"
    bl_description = (
        "Record the robot's current pose as a waypoint, storing both the tool "
        "pose and the joint configuration. On a frame after the last waypoint "
        "recorded, also add a joint motion from it to this one, ending here"
    )

    name: StringProperty(name="Name", default="", options={"SKIP_SAVE"})
    motion: BoolProperty(
        name="Add Motion",
        description=(
            "Also add a motion from the last waypoint recorded, ending on this "
            "frame, when this frame is after the one that was recorded on"
        ),
        default=True,
        options={"SKIP_SAVE"},
    )

    def execute(self, context: bpy.types.Context) -> set[str]:
        rig = active_rig(context)
        if not builder.joint_bones(rig):
            self.report({"ERROR"}, "This rig has no joints to record")
            return {"CANCELLED"}

        convert_saved_before_motions(rig)
        context.view_layer.update()
        # Taught relative to what the highlighted waypoint is: a job taught on
        # a workpiece stays on it, one point after another.
        highlighted = _waypoint_at(rig, -1)
        refused = refusal(rig, highlighted) if highlighted is not None else None
        if refused is not None:
            self.report({"ERROR"}, refused)
            return {"CANCELLED"}
        if highlighted is not None and rig.library is None:
            # Its marker parented or cleared by hand since: what it is now.
            references.sync(rig, highlighted)
        reference = (
            references.reference_of(rig, highlighted) if highlighted is not None else None
        )
        waypoint, error = new_waypoint(rig, self.name, reference)
        if error is not None:
            self.report({"ERROR"}, error)
            return {"CANCELLED"}
        frame = context.scene.frame_current
        previous = waypoint_by_uid(rig, rig.kinema_last_recorded)
        message = f"Recorded '{waypoint.name}'"
        if reference is not None:
            message += f" relative to '{reference.name}'"
        if self.motion and previous is not None and frame > rig.kinema_last_recorded_frame:
            # Joint by default: it is the move that always works. Linear is a
            # deliberate choice about a tool path, so it should be one.
            motion = new_motion(
                rig, previous, waypoint, rig.kinema_last_recorded_frame, frame
            )
            message += (
                f", with a motion from '{previous.name}' over frames "
                f"{motion.frame_start}-{frame}"
            )
        rig.kinema_last_recorded, rig.kinema_last_recorded_frame = waypoint.uid, frame
        self.report({"INFO"}, message)
        return {"FINISHED"}


class KINEMA_OT_update_waypoint(KinemaWaypointOperator):
    bl_idname = "kinema.update_waypoint"
    bl_label = "Update Waypoint"
    bl_description = "Re-record this waypoint from the robot's current pose"

    index: IntProperty(name="Index", default=-1, options={"SKIP_SAVE"})

    def execute(self, context: bpy.types.Context) -> set[str]:
        rig = active_rig(context)
        convert_saved_before_motions(rig)
        waypoint = _waypoint_at(rig, self.index)
        if waypoint is None:
            self.report({"ERROR"}, "No waypoint to update")
            return {"CANCELLED"}

        context.view_layer.update()
        refused = refusal(rig, waypoint)
        if refused is not None:
            self.report({"ERROR"}, refused)
            return {"CANCELLED"}
        references.sync(rig, waypoint)
        reference = references.reference_of(rig, waypoint)
        error = _record(rig, waypoint, reference)
        if error is not None:
            self.report({"ERROR"}, error)
            return {"CANCELLED"}
        references.place_marker(rig, waypoint, reference, _unflatten(waypoint.pose))
        self.report({"INFO"}, f"'{waypoint.name}' updated from the current pose")
        return {"FINISHED"}


class KINEMA_OT_goto_waypoint(KinemaWaypointOperator):
    bl_idname = "kinema.goto_waypoint"
    bl_label = "Go To Waypoint"
    bl_description = (
        "Jump to the first frame the job is at this waypoint and restore the "
        "exact configuration it was taught in, rather than whichever solution "
        "the solver picks today"
    )

    index: IntProperty(name="Index", default=-1, options={"SKIP_SAVE"})

    def execute(self, context: bpy.types.Context) -> set[str]:
        rig = active_rig(context)
        convert_saved_before_motions(rig)
        waypoint = _waypoint_at(rig, self.index)
        if waypoint is None:
            self.report({"ERROR"}, "No waypoint to go to")
            return {"CANCELLED"}
        refused = refusal(rig, waypoint)
        if refused is not None:
            self.report({"ERROR"}, refused)
            return {"CANCELLED"}
        # A marker parented by hand, or an object deleted, caught up first: where
        # the waypoint is comes from its frame of reference.
        lost = references.sync(rig, waypoint) if rig.library is None else None

        # The first frame a motion starts or ends there; where none does, the
        # robot is posed on the frame it is on.
        frames = [
            int(frame)
            for motion in rig.kinema_motions
            for uid, frame in ((motion.start, motion.frame_start), (motion.end, motion.frame_end))
            if uid == waypoint.uid
        ]
        if frames:
            context.scene.frame_set(min(frames))
        # The stored joint vector, not the pose: a pose has up to eight
        # solutions and re-solving would be free to pick a different one. On
        # an object that has moved since, walked there from it.
        context.view_layer.update()
        values = list(waypoint.q)
        here, taught = pose_of(waypoint), references.taught_pose(waypoint)
        if references.reference_of(rig, waypoint) is not None and _differs(here, taught):
            walked = _walk_here(rig, waypoint, taught, here)
            if walked is None:
                self.report(
                    {"WARNING"},
                    f"The robot can't reach '{waypoint.name}' where its object is now "
                    f"from the configuration it was taught in; shown as taught",
                )
                # The goal with the joints, so live IK leaves them as taught.
                here = taught
            else:
                values = walked
        apply_joint_values(rig, values[: waypoint.dof])
        # And put the goal where the tool now is, so live IK does not
        # immediately drag the arm back off the configuration just restored.
        ik_name = rig.get(builder.PROP_IK_BONE)
        context.view_layer.update()
        if ik_name and ik_name in rig.pose.bones:
            rig.pose.bones[ik_name].matrix = here
        context.view_layer.update()
        frame = context.scene.frame_current
        if lost:
            self.report({"WARNING"}, f"At '{waypoint.name}', frame {frame}. {lost}")
        else:
            self.report({"INFO"}, f"At '{waypoint.name}', frame {frame}")
        return {"FINISHED"}


class KINEMA_OT_remove_waypoint(KinemaWaypointOperator):
    bl_idname = "kinema.remove_waypoint"
    bl_label = "Remove Waypoint"
    bl_description = "Delete this waypoint, its marker, and the motions to and from it"

    index: IntProperty(name="Index", default=-1, options={"SKIP_SAVE"})

    def execute(self, context: bpy.types.Context) -> set[str]:
        rig = active_rig(context)
        convert_saved_before_motions(rig)
        index = self.index if self.index >= 0 else rig.kinema_active_waypoint
        if not 0 <= index < len(rig.kinema_waypoints):
            self.report({"ERROR"}, "No waypoint to remove")
            return {"CANCELLED"}

        return reported(self, delete_waypoint(rig, index))


def new_waypoint(rig, name: str = "", reference=None, values=None):
    """A waypoint on ``rig`` where its tool is -- or would be at ``values``, a joint
    vector, the rig left where it stands -- relative to ``reference``, an object, or
    the robot's base; with its marker. Returns it and None, or None and why not."""
    waypoint = rig.kinema_waypoints.add()
    waypoint.uid = _new_uid()
    waypoint.name = name or f"point{len(rig.kinema_waypoints)}"
    error = _record(rig, waypoint, reference, values)
    if error is not None:
        rig.kinema_waypoints.remove(len(rig.kinema_waypoints) - 1)
        return None, error
    waypoint.marker = _make_marker(rig, waypoint, _unflatten(waypoint.pose), reference)
    rig.kinema_active_waypoint = len(rig.kinema_waypoints) - 1
    from . import ik as ik_ops

    # Taught in the configuration the arm is in: a switch before it is behind it.
    ik_ops.forget_switch(rig.name)
    return waypoint, None


def delete_waypoint(rig, index: int) -> Outcome:
    """Remove ``rig``'s waypoint at ``index``, its marker, and the motions to and from it."""
    waypoint = rig.kinema_waypoints[index]
    name = waypoint.name
    # The row owns its marker, the way an attachment row owns its copy.
    marker = waypoint.marker
    if marker is not None and PROP_WAYPOINT in marker:
        bpy.data.objects.remove(marker, do_unlink=True)
    uid = waypoint.uid
    gone = [
        i for i, motion in enumerate(rig.kinema_motions) if uid in (motion.start, motion.end)
    ]
    for i in reversed(gone):
        rig.kinema_motions.remove(i)
    left = len(rig.kinema_motions) - 1
    rig.kinema_active_motion = max(0, min(rig.kinema_active_motion, left))
    if rig.kinema_last_recorded == uid:
        # Recording goes on from where the job now ends, or from nothing.
        last = max(rig.kinema_motions, key=lambda m: int(m.frame_end), default=None)
        rig.kinema_last_recorded = last.end if last is not None else ""
        rig.kinema_last_recorded_frame = int(last.frame_end) if last is not None else 0
    rig.kinema_waypoints.remove(index)
    # max(0, ...): removing the last row leaves an empty list, and the
    # index would otherwise go to -1 and rely on the property's own min to
    # catch it.
    rig.kinema_active_waypoint = max(0, min(index, len(rig.kinema_waypoints) - 1))
    said = ""
    if gone:
        said = f", and {len(gone)} motion{'s' if len(gone) != 1 else ''} to and from it"
    return Outcome(True, "INFO", f"Removed '{name}'{said}")


def new_motion(
    rig, start, end, frame_start: int, frame_end: int, move: str = MOVE_JOINT, name: str = ""
):
    """A motion on ``rig`` from waypoint ``start`` to ``end`` over those frames, named
    ``name`` or after the waypoint it ends at; made the active one."""
    motion = rig.kinema_motions.add()
    motion.name = name or end.name
    motion.start, motion.end = start.uid, end.uid
    motion.move = move
    motion.frame_start, motion.frame_end = int(frame_start), int(frame_end)
    rig.kinema_active_motion = len(rig.kinema_motions) - 1
    rig.kinema_unconverted = ""
    return motion


def delete_motion(rig, index: int) -> Outcome:
    """Remove ``rig``'s motion at ``index``. Its waypoints stay."""
    name = rig.kinema_motions[index].name
    rig.kinema_motions.remove(index)
    rig.kinema_active_motion = max(0, min(index, len(rig.kinema_motions) - 1))
    return Outcome(True, "INFO", f"Removed '{name}'")


class KINEMA_OT_add_motion(KinemaWaypointOperator):
    bl_idname = "kinema.add_motion"
    bl_label = "Add Motion"
    bl_description = (
        "Add a joint motion after the last one, from where it ends to the "
        "highlighted waypoint, a second long. With no motion yet, from the "
        "highlighted waypoint to the next one, from the current frame"
    )

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        rig = active_rig(context)
        return rig is not None and len(rig.kinema_waypoints) > 1

    def execute(self, context: bpy.types.Context) -> set[str]:
        rig = active_rig(context)
        convert_saved_before_motions(rig)
        target = _waypoint_at(rig, -1) or rig.kinema_waypoints[-1]
        last = max(rig.kinema_motions, key=lambda m: int(m.frame_end), default=None)
        start = waypoint_by_uid(rig, last.end) if last is not None else None
        if last is not None and start is None:
            self.report(
                {"ERROR"},
                f"'{last.name}', the last motion, ends at a waypoint that no longer "
                f"exists: point it at one, or remove it, then Add",
            )
            return {"CANCELLED"}
        if start is None:
            # Nothing to go on from: the highlighted waypoint to the one after it.
            if len(rig.kinema_waypoints) < 2:
                self.report({"ERROR"}, "A motion goes between two waypoints: record another")
                return {"CANCELLED"}
            start = target
            index = list(rig.kinema_waypoints).index(target)
            target = rig.kinema_waypoints[(index + 1) % len(rig.kinema_waypoints)]
        first = int(last.frame_end) if last is not None else context.scene.frame_current
        motion = new_motion(
            rig, start, target, first, first + max(1, round(scene_fps(context.scene)))
        )
        self.report(
            {"INFO"},
            f"Added '{motion.name}', '{start.name}' to '{target.name}' over frames "
            f"{motion.frame_start}-{motion.frame_end}",
        )
        return {"FINISHED"}


class KINEMA_OT_remove_motion(KinemaWaypointOperator):
    bl_idname = "kinema.remove_motion"
    bl_label = "Remove Motion"
    bl_description = "Delete this motion. Its waypoints stay"

    index: IntProperty(name="Index", default=-1, options={"SKIP_SAVE"})

    def execute(self, context: bpy.types.Context) -> set[str]:
        rig = active_rig(context)
        convert_saved_before_motions(rig)
        index = self.index if self.index >= 0 else rig.kinema_active_motion
        if not 0 <= index < len(rig.kinema_motions):
            self.report({"ERROR"}, "No motion to remove")
            return {"CANCELLED"}
        return reported(self, delete_motion(rig, index))


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
        "Key the robot through its motions in frame order: joint channels "
        "for joint moves, the IK target for linear ones"
    )

    def execute(self, context: bpy.types.Context) -> set[str]:
        return reported(self, generate_job(context, active_rig(context)))


def generate_job(context, rig) -> Outcome:
    """Key ``rig`` through its motions in frame order, and check the job: what
    Generate Motion does. No keys are written where it is refused; the job is
    caught up first all the same (see :func:`job_plan`)."""
    job, failed = _plan(rig)
    if job is None:
        return failed
    plan = job.spans
    # Where every waypoint is on every frame of the job, for those taught
    # relative to an object that moves: see ops/references.py.
    placement = place_job(context, rig, plan)

    needs_ik = any(keyed_as(span, placement) == MOVE_LINEAR for span in plan)
    ik_name = ik_bone_name(rig)
    if needs_ik and ik_name is None:
        moving = [
            span.label
            for span in plan
            if span.move != MOVE_LINEAR and needs_tracking(span, placement)
        ]
        if moving:
            names = ", ".join(f"'{name}'" for name in moving)
            return Outcome(
                False,
                "ERROR",
                f"{names}: held on an object that moves, which the tool can only "
                f"follow with an IK target; add one",
            )
        return Outcome(
            False,
            "ERROR",
            "Linear moves need an IK target; add one, or set those motions to Joint",
        )
    # The marker is the waypoint: one dragged since teaching takes every
    # move to and from it along, and one relative to an object goes where
    # the object is on each frame the job is there. Its joints are found
    # there before anything is checked or written, and the checks below see
    # them.
    unreachable, followed, visits = follow_markers(rig, plan, placement)
    if unreachable:
        names = ", ".join(f"'{name}'" for name in unreachable)
        return Outcome(
            False,
            "ERROR",
            f"{names}: the robot can't follow the marker there from the "
            f"configuration it was taught in. Move the marker back, or pose "
            f"the robot there and press Update",
        )
    changed = {waypoint.as_pointer(): values for waypoint, values in followed}
    changed.update(visits)
    refused, turned, shared, changed = (
        settle_configurations(rig, plan, changed, placement)
        if needs_ik
        else ([], [], [], changed)
    )
    if shared:
        names = ", ".join(f"'{name}'" for name in shared)
        return Outcome(
            False,
            "ERROR",
            f"{names}: linear moves arrive there a whole turn apart, and a "
            f"waypoint keeps one turn for every move that uses it. Teach a "
            f"waypoint for each, or make one of the moves a joint move",
        )
    if refused:
        names = ", ".join(f"'{name}'" for name in refused)
        return Outcome(
            False,
            "ERROR",
            f"{names}: taught in a different configuration from the waypoint "
            f"before, and a linear move can't change configuration. Make it a "
            f"joint move, or re-teach one of the two",
        )
    # Only now, with nothing refused: a cancelled operator leaves no undo
    # step, so a change made before a refusal could not be taken back.
    strips.retime(rig, job.retimes)
    # Stored as Update stores a pose, so the waypoint *is* where its marker
    # is: Go To restores it there, and the next generation starts from it.
    for waypoint, values in followed:
        waypoint.q = values
        waypoint.pose = _flatten(references.local_pose(waypoint))
    for span in plan:
        for waypoint in (span.start, span.end):
            if references.reference_of(rig, waypoint) is not None:
                # A marker dragged on its object moves the waypoint on it; the
                # joints stay where they were taught, and each visit walks
                # from there.
                waypoint.pose = _flatten(references.local_pose(waypoint))
    for waypoint, values, _, _ in turned:
        if references.reference_of(rig, waypoint) is None:
            waypoint.q = values

    first, last = plan[0].start_frame, plan[-1].end_frame
    # A linear finish leaves a switch-off key one frame past the end, so
    # the job owns that frame too -- but only then. Claiming it after a
    # joint finish, where nothing is written there, would have the next
    # regeneration delete a key the user had put on the first free frame.
    ends_linear = keyed_as(plan[-1], placement) == MOVE_LINEAR
    owned = (first, last + 1 if ends_linear else last)
    # Union with what the last generation covered, which the new plan need
    # not still reach: shortening a job by moving its final waypoint
    # earlier has to take the old keys with it, or the robot goes on
    # visiting a frame no waypoint claims any more.
    clear_from, clear_to = _union(_generated_range(rig), owned)

    original = context.scene.frame_current
    # Suspended for the same reason baking suspends: writing keys walks the
    # frame, and the live handler would solve each one on the way past and
    # fight the values being written. Written into the job's action, which
    # is then laid out as one NLA strip per motion: see ops/strips.py.
    with _handlers_suspended(), strips.editing(rig, context):
        _clear_generated(rig, ik_name, clear_from, clear_to)
        # The target's last keyed rotation, so the next is keyed in the same
        # hemisphere: see _key_linear_span.
        previous = None
        for span in plan:
            if keyed_as(span, placement) == MOVE_JOINT:
                _key_joint_span(rig, span, changed)
            else:
                previous = _key_linear_span(
                    rig, span, ik_name, previous, placement, changed
                )
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
        muted = strips.lay_out(rig, plan)
    rig.kinema_generated_range = owned
    # What was generated, so Optimize Motion can tell when the keys no
    # longer follow the waypoints.
    rig.kinema_generated_job = job_signature(rig)
    rig.kinema_optimized_job = ""
    context.scene.frame_set(original)
    checks = check_job(context, rig, plan, placement=placement)
    # Kept on the move's row too, so it is still there once the report has
    # gone: anyone who meant the taught turn has to re-teach that waypoint.
    notes = {
        frame: f"{description}: kept as the line arrives"
        for _, _, description, frame in turned
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
    for note in job.notes:
        message += f". {note}"
    if muted:
        message += f". {strips.muted_text(muted)}"
    for waypoint, _, description, _ in turned:
        message += (
            f". '{waypoint.name}' was taught a whole turn from where the line "
            f"arrives, and keeps the turn it arrives with ({description}). "
            f"Re-teach it to change that"
        )
    return Outcome(True, "WARNING" if flagged or job.notes else "INFO", message, flagged)


def _key_joint_span(rig, span, changed) -> None:
    """Key the joints at both ends and switch live IK off for the span.

    The joints are what carry a MoveJ, so the solver must not be writing
    over them while it runs. Each end's are the waypoint's as the job meets
    it there (``changed``: see :func:`values_at`).
    """
    rig.kinema_ik_enabled = False
    rig.keyframe_insert(data_path="kinema_ik_enabled", frame=span.start_frame)
    for waypoint, frame in (
        (span.start, span.start_frame),
        (span.end, span.end_frame),
    ):
        apply_joint_values(rig, values_at(rig, waypoint, frame, changed)[: waypoint.dof])
        for pose_bone in builder.joint_bones(rig):
            key_joint_value(pose_bone, frame)

def _key_linear_span(rig, span, ik_name: str, previous, placement, changed):
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

    On an object that moves, the line moves with it: the goal is keyed on
    every frame, where the line is on that frame. So is a hold there, which
    follows the object as the tool would follow a conveyor.
    """
    rig.kinema_ik_enabled = True
    rig.keyframe_insert(data_path="kinema_ik_enabled", frame=span.start_frame)
    goal = rig.pose.bones[ik_name]
    goal.rotation_mode = "QUATERNION"
    for frame, pose in line_of(span, placement):
        goal.matrix = pose
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
        apply_joint_values(rig, values_at(rig, waypoint, frame, changed)[: waypoint.dof])
        for pose_bone in builder.joint_bones(rig):
            key_joint_value(pose_bone, frame)
    return previous


# --------------------------------------------------------------------------
# where the job is, frame by frame
# --------------------------------------------------------------------------
def place_job(context, rig, plan) -> references.Placement:
    """Where ``plan``'s waypoints are on each of its frames: see ops/references.py."""
    waypoints = [w for span in plan for w in (span.start, span.end)]
    frames = range(plan[0].start_frame, plan[-1].end_frame + 1) if plan else ()
    return references.Placement(rig, waypoints, frames, context.scene)


def is_still(span) -> bool:
    """Whether the robot stays at one waypoint over ``span``: a gap held there, or
    a motion from a waypoint to itself, which is a pause."""
    return span.made == MADE_HOLD or span.start.as_pointer() == span.end.as_pointer()


def needs_tracking(span, placement) -> bool:
    """Whether ``span`` has to follow a moving frame of reference on every frame:
    a linear move or a hold on an object that moves during it. A joint move
    ends where the object is; between, the tool's path is free, as it always is."""
    if placement is None or not placement.relative:
        return False
    if span.move != MOVE_LINEAR and not is_still(span):
        return False
    return placement.tracks(span)


def keyed_as(span, placement) -> str:
    """How ``span`` is keyed: a hold on a moving object is followed as a line is,
    by the IK goal; everything else as its move says."""
    if span.move == MOVE_LINEAR or needs_tracking(span, placement):
        return MOVE_LINEAR
    return MOVE_JOINT


def line_of(span, placement) -> list[tuple[int, Matrix]]:
    """The frames a linear move keys its goal on, each with where the goal is then.

    Still frames of reference: :func:`_linear_key_frames` along the one line.
    One that moves: every frame, the line between where the two ends are on
    that frame -- on the same object, its line carried with it.
    """
    first, last = span.start_frame, span.end_frame
    if needs_tracking(span, placement):
        out = []
        for frame in range(first, last + 1):
            fraction = (frame - first) / span.frames if span.frames else 0.0
            start = pose_of(span.start, placement, frame)
            end = pose_of(span.end, placement, frame)
            out.append((frame, _between(start, end, fraction)))
        return out
    start, end = pose_of(span.start, placement, first), pose_of(span.end, placement, last)
    return [
        (frame, _between(start, end, (frame - first) / span.frames))
        for frame in _linear_key_frames(span, start, end)
    ]


def _visit_key(rig, waypoint, frame: int):
    """What a waypoint's joint values are kept under as the job meets it: the
    waypoint, at the base, where it holds one set; each visit, on an object."""
    if references.reference_of(rig, waypoint) is None:
        return waypoint.as_pointer()
    return (waypoint.as_pointer(), int(frame))


def values_at(rig, waypoint, frame: int, changed=None) -> list[float]:
    """The joint values the job has at ``waypoint`` on ``frame``: as found for that
    visit, or for the waypoint, in ``changed``; as taught otherwise."""
    changed = changed or {}
    key = _visit_key(rig, waypoint, frame)
    if key in changed:
        return list(changed[key])
    if waypoint.as_pointer() in changed:
        return list(changed[waypoint.as_pointer()])
    return list(waypoint.q)


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
def follow_markers(rig, plan, placement=None):
    """The joint values each waypoint whose marker has been dragged takes there.

    Returns ``(refused, followed, visits)`` and writes nothing, so a refusal
    leaves the waypoints exactly as they were:

    - ``followed``: ``(waypoint, joint values)`` for every waypoint in the job
      whose marker has moved since it was taught. The values are walked there
      from the taught ones, a few millimetres and degrees at a time, so they
      keep the configuration taught: the same elbow, wrist and turns.
    - ``visits``: for a waypoint taught relative to an object, the joint values
      on each frame the job meets it somewhere other than where it was taught,
      by ``(waypoint.as_pointer(), frame)``: walked there from the taught ones,
      and not stored, since the object can be elsewhere on each.
    - ``refused``: the names of those the walk cannot take there -- out of
      reach, or taught with a different set of joints than the rig has now.

    The marker is the waypoint (:func:`pose_of`), so once these are stored
    every move to and from it follows: a joint move replays them, and a linear
    move keys them as IK's seed. Keying the values taught at the marker's old
    place instead left a joint move there, and had the robot jump back wherever
    live IK hands a linear move over to the joint keys.
    """
    moved, seen, met = [], set(), []
    for span in plan:
        for waypoint, frame in ((span.start, span.start_frame), (span.end, span.end_frame)):
            key = (waypoint.as_pointer(), int(frame))
            if references.reference_of(rig, waypoint) is not None:
                if key in seen:
                    continue
                seen.add(key)
                target = pose_of(waypoint, placement, frame)
                if _differs(target, references.taught_pose(waypoint)):
                    met.append((waypoint, int(frame), target))
            elif waypoint.as_pointer() not in seen and marker_has_moved(waypoint):
                seen.add(waypoint.as_pointer())
                moved.append(waypoint)
    if not moved and not met:
        return [], [], {}
    # Built here rather than asked of the solver: a job of joint moves needs no
    # IK target, and the solver is only made for one.
    chain = chain_from_rig(rig, _tool_bone(rig))
    if chain is None:
        names = [w.name for w in moved] + [_on_frame(w, f) for w, f, _ in met]
        return names, [], {}

    names = [pose_bone.name for pose_bone in builder.joint_bones(rig)]
    columns = [names.index(name) for name in chain.bone_names]
    held = manager.held_mask(rig, chain)
    to_solver = np.linalg.inv(manager.root_pose(rig))

    refused, followed, visits = [], [], {}
    for waypoint in moved:
        values = None
        if waypoint.dof == len(names):
            values = _walk(chain, held, to_solver, columns, waypoint.q,
                           _unflatten(waypoint.pose), pose_of(waypoint))
        if values is None:
            refused.append(waypoint.name)
        else:
            followed.append((waypoint, values))
    # A still object meets the job in one place however often: walked once.
    walked = {}
    for waypoint, frame, target in met:
        place = (waypoint.as_pointer(), tuple(round(v, 9) for v in _flatten(target)))
        if place not in walked:
            walked[place] = None
            if waypoint.dof == len(names):
                walked[place] = _walk(chain, held, to_solver, columns, waypoint.q,
                                      references.taught_pose(waypoint), target)
        values = walked[place]
        if values is None:
            refused.append(_on_frame(waypoint, frame))
        else:
            visits[(waypoint.as_pointer(), frame)] = values
    return refused, followed, visits


def _on_frame(waypoint, frame: int) -> str:
    return f"{waypoint.name} (frame {frame})"


def _walk_here(rig, waypoint, start: Matrix, end: Matrix):
    """``waypoint``'s taught joints walked from ``start`` to ``end``, or None."""
    chain = chain_from_rig(rig, _tool_bone(rig))
    names = [pose_bone.name for pose_bone in builder.joint_bones(rig)]
    if chain is None or waypoint.dof != len(names):
        return None
    columns = [names.index(name) for name in chain.bone_names]
    held = manager.held_mask(rig, chain)
    to_solver = np.linalg.inv(manager.root_pose(rig))
    return _walk(chain, held, to_solver, columns, waypoint.q, start, end)


def _differs(a: Matrix, b: Matrix, tolerance: float = 1e-6) -> bool:
    return any(abs(a[r][c] - b[r][c]) > tolerance for r in range(4) for c in range(4))


def _walk(chain, held, to_solver, columns, taught, start: Matrix, end: Matrix):
    """Joint values ``taught`` at ``start``, walked to ``end``.

    Each step is seeded from the last, as a linear move's configuration check
    walks its line: one solve from the taught values straight to a marker
    dragged a long way could land in any configuration that reaches it. Held
    joints keep their taught values. None if a step cannot be solved.
    """
    values = np.array(list(taught), dtype=float)
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


def settle_configurations(rig, plan, overrides=None, placement=None):
    """Check every linear move's end against the configuration it was taught in.

    ``overrides`` maps a waypoint's ``as_pointer()`` to joint values to use in
    place of its stored ones: those :func:`follow_markers` found for it, which
    are only written once nothing has been refused. A visit to one taught
    relative to an object is under ``(as_pointer(), frame)``, as
    :func:`values_at` reads them, and so is what it arrives with.

    Returns ``(refused, turned, shared, settled)`` and writes nothing, so a
    refusal leaves the waypoints exactly as they were:

    - ``refused``: the linear moves whose two ends were taught in different
      configurations, named by the waypoint each arrives at. A straight line
      cannot change configuration, so these are not generated.
    - ``turned``: ``(waypoint, joint values, description, frame)`` for each linear
      move whose end was taught with a joint a whole turn from where the line
      arrives. That is the same pose, and a linear move keeps the turns it
      starts with -- a KUKA LIN ignores its target's Turn the same way -- so
      the waypoint takes the turns the line arrives with. Seeding IK from the
      taught turn instead would pull it over to that turn part-way along the
      line, in a single frame.

    - ``shared``: the waypoints whose turns a linear move would change after
      an earlier linear move started or ended there with the turns as they
      were. A waypoint holds one set of joint values for every motion that
      uses it, so changing it would leave the earlier line a whole turn off.
    - ``settled``: ``overrides`` with what this found: each end's turns as its
      line arrives, and a pause ending as it began.

    Settled in time order, so a move that starts where an earlier one's turns
    were changed walks from the changed values. A hold on an object that moves
    is walked as a line is, since its goal is keyed on every frame as a line's
    is; one on an object standing still ends with the joints it began with,
    which a visit walked afresh from the taught joints need not have.
    """
    solver = manager.get_solver(rig, ik_bone_name(rig))
    # A waypoint's pose is the TCP's. With the tip handed to another bone the
    # solver is aiming something else at it, and the walk would prove nothing.
    changed = dict(overrides or {})
    if solver is None or solver.tip_bone != _tool_bone(rig):
        return [], [], [], changed
    chain = solver.chain
    refused, turned, shared = [], [], []
    lined = set()  # waypoints an earlier linear move started or ended at
    for span in plan:
        start_key = _visit_key(rig, span.start, span.start_frame)
        end_key = _visit_key(rig, span.end, span.end_frame)
        if keyed_as(span, placement) != MOVE_LINEAR:
            if is_still(span) and start_key != end_key:
                changed[end_key] = values_at(rig, span.start, span.start_frame, changed)
            continue
        start = values_at(rig, span.start, span.start_frame, changed)
        end = values_at(rig, span.end, span.end_frame, changed)
        arrival = _arrival(rig, span, chain, start, end, placement)
        if arrival is None:
            # Not judged, but its waypoints' turns are what it keys all the same.
            lined.update((start_key, end_key))
            continue
        kind, values = arrival
        if kind == BRANCH:
            refused.append(span.end.name)
        elif kind == TURNS and end_key in lined:
            if span.end.name not in shared:
                shared.append(span.end.name)
        elif kind == TURNS:
            changed[end_key] = values
            turned.append(
                (span.end, values, _turn_description(rig, chain, span, end, values), span.end_frame)
            )
        lined.update((start_key, end_key))
    return refused, turned, shared, changed


def _arrival(rig, span, chain, start_values, end_values, placement=None):
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

    def solve(seed, pose: Matrix):
        goal = to_solver @ np.array(pose, dtype=float)
        return numpy_backend.solve(chain, seed, goal, held=held)

    steps = _line_steps(span, placement)
    # The end's configuration solved for where its marker is: the same thing
    # as the values given, which for a moved marker follow_markers has already
    # walked there.
    reference = solve(q_end, steps[-1][1])
    if not reference.converged:
        return None

    q = q_start
    for fraction, pose in steps:
        seed = q.copy()
        # Held joints are not solved for; they go where their keys take them.
        seed[held] = q_start[held] + (q_end[held] - q_start[held]) * fraction
        result = solve(seed, pose)
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


def _line_steps(span, placement) -> list[tuple[float, Matrix]]:
    """Where a walk along a linear move solves: ``(fraction of the move, pose)``.

    Along the one line, finely enough (:func:`_walk_fractions`); on an object
    that moves, frame by frame where the line is then, and as finely between.
    """
    if not needs_tracking(span, placement):
        start = pose_of(span.start, placement, span.start_frame)
        end = pose_of(span.end, placement, span.end_frame)
        return [
            (fraction, _between(start, end, fraction))
            for fraction in _walk_fractions(start, end, span.frames)
        ]
    keyed = line_of(span, placement)
    steps = []
    for (frame, before), (_, after) in zip(keyed, keyed[1:], strict=False):
        for fraction in _walk_fractions(before, after):
            overall = (frame - span.start_frame + fraction) / span.frames
            steps.append((overall, _between(before, after, fraction)))
    return steps or [(1.0, keyed[-1][1])]


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
    frames_needed: IntProperty(name="Frames Needed")
    clearance: FloatProperty(name="Clearance", unit="LENGTH")
    clearance_bone: StringProperty(name="Closest Link")
    clearance_obstacle: StringProperty(name="Closest Obstacle")
    clearance_frame: IntProperty(name="Closest Frame")
    clearance_between: BoolProperty(
        name="Between Frames", description="The closest approach was just after that frame"
    )


def check_job(
    context, rig, plan, samples=None, placement=None
) -> list[motion_check.MoveCheck] | None:
    """Play the job once, the way playback does, and measure every move.

    ``samples`` is the job already played by :func:`play_job`, to measure that
    instead of playing it again; ``placement``, where its waypoints are, as
    :func:`place_job` read it.
    """
    if samples is None:
        samples = play_job(context, rig, plan)
    if samples is None:
        # Taught with a TCP that has since been deleted: there is nothing to
        # measure a line against. None, not an empty list, so that no caller
        # can mistake a job it could not measure for one with nothing wrong.
        return None
    if placement is None:
        placement = place_job(context, rig, plan)
    joints = check_joints(rig)
    fps = scene_fps(context.scene)
    capsules = tuple(collision_ops.capsules_of(rig))
    return [
        motion_check.check_move(
            span.label,
            keyed_as(span, placement),
            span.start_frame,
            span.end_frame,
            np.array(pose_of(span.start, placement, span.start_frame), dtype=float),
            np.array(pose_of(span.end, placement, span.end_frame), dtype=float),
            samples,
            joints,
            fps,
            capsules,
            along=_along(span, placement),
        )
        for span in plan
    ]


def _along(span, placement):
    """Where a move's two ends are on each of its frames, for a line on an object
    that moves: what its line is measured against there. None for a still one."""
    if not needs_tracking(span, placement):
        return None
    return {
        frame: (
            np.array(pose_of(span.start, placement, frame), dtype=float),
            np.array(pose_of(span.end, placement, frame), dtype=float),
        )
        for frame in range(span.start_frame, span.end_frame + 1)
    }


def check_joints(rig) -> list[motion_check.Joint]:
    """The rig's joints as the check measures them, in rig order, with their limits."""
    return [
        motion_check.Joint(
            name=pose_bone.name,
            prismatic=pose_bone.bone.get(builder.PROP_JOINT_TYPE, "revolute")
            == "prismatic",
            lower=_float_or_none(pose_bone.bone.get(builder.PROP_LOWER)),
            upper=_float_or_none(pose_bone.bone.get(builder.PROP_UPPER)),
            velocity=limit_of(pose_bone),
            acceleration=acceleration_limit_of(pose_bone),
        )
        for pose_bone in builder.joint_bones(rig)
    ]


def scene_fps(scene) -> float:
    return scene.render.fps / scene.render.fps_base


def play_job(context, rig, plan) -> list[motion_check.Sample] | None:
    """Every frame of the job as playback shows it, and one either side.

    Each frame is set with the handlers live, so live IK solves it with the
    rig's own backend, from the rig's own keys -- whatever playback would
    show, including a compile if the solver needs one. None for a rig with no
    TCP to read the tool from.
    """
    scene = context.scene
    bones = builder.joint_bones(rig)
    tool = _tool_bone(rig)
    if tool not in rig.pose.bones:
        return None
    first, last = plan[0].start_frame, plan[-1].end_frame

    # A frame either side of the job as well. An acceleration is measured
    # from a frame's neighbours, and the job's own ends are where one that
    # sets off or stops at speed changes it all at once.
    samples = []
    original = scene.frame_current
    # Where the robot is against the obstacles is read as it plays too, where
    # there are both capsules and obstacles to measure.
    capsules = collision_ops.capsules_of(rig)
    measured = bool(capsules) and bool(collision_ops.obstacle_objects(scene))
    window = context.window_manager
    window.progress_begin(first - 1, last + 1)
    try:
        for frame in range(first - 1, last + 2):
            scene.frame_set(frame)
            context.view_layer.update()
            segments, radii, obstacles, placement = None, None, (), None
            if measured:
                a, b, radii = collision_ops.capsule_ends(rig, capsules)
                segments = np.stack([a, b], axis=1)
                obstacles = tuple(collision_ops.obstacles(context))
                placement = np.array(rig.matrix_world) @ manager.root_pose(rig)
            samples.append(
                motion_check.Sample(
                    frame=frame,
                    q=np.array([joint_value(pose_bone) for pose_bone in bones]),
                    tool=np.array(rig.pose.bones[tool].matrix, dtype=float),
                    segments=segments,
                    radii=radii,
                    obstacles=obstacles,
                    placement=placement,
                )
            )
            window.progress_update(frame)
        if measured:
            samples = _measured_between(context, rig, capsules, samples, first, last, window)
    finally:
        window.progress_end()
        scene.frame_set(original)
    return samples


def _measured_between(
    context, rig, capsules, samples, first: int, last: int, window=None
) -> list:
    """``samples`` with the robot against the obstacles measured between frames too.

    Only where a capsule moved further than its radius against an obstacle
    from one frame of the job to the next, either of them moving
    (``motion_check.between_frames``). Each part is set as a
    subframe, so the joints come as playback interpolates them, and live IK
    solves a line from its goal there. Nothing is recorded for the limit
    check, which reads whole frames. ``window``'s progress goes over the job's
    frames again, part by part.
    """
    scene = context.scene
    out = list(samples)
    with velocity.unobserved():
        for index, (now, then) in enumerate(zip(samples, samples[1:], strict=False)):
            if not first <= now.frame < last:
                continue
            parts = motion_check.between_frames(
                now.segments, then.segments, now.radii, now.obstacles, then.obstacles,
                radii_after=then.radii,
            )
            measured = []
            for part in range(1, parts):
                scene.frame_set(now.frame, subframe=part / parts)
                context.view_layer.update()
                if window is not None:
                    window.progress_update(now.frame + part / parts)
                a, b, radii = collision_ops.capsule_ends(rig, capsules)
                measured.append(
                    replace(
                        now,
                        segments=np.stack([a, b], axis=1),
                        radii=radii,
                        obstacles=tuple(collision_ops.obstacles(context)),
                        subframe=part / parts,
                    )
                )
            if measured:
                out[index] = replace(now, between=tuple(measured))
    return out


def estimate_frames_needed(context, rig, checks, samples) -> None:
    """Give each move still over a limit the frames it needs, on a job Optimize Motion wrote.

    Only there: on a job straight from Generate Motion, a linear move changes
    speed at its waypoints all at once however long it takes, and the estimate
    would mean nothing (``motion_check.frames_needed``).
    """
    if checks is None or rig.kinema_optimized_job != job_signature(rig):
        return
    joints, fps = check_joints(rig), scene_fps(context.scene)
    for index, check in enumerate(checks):
        check.frames_needed = motion_check.frames_needed(
            check, samples, joints, fps, index == 0, index == len(checks) - 1
        )


def store_check(rig, checks) -> int:
    """Keep the findings on the rig. Returns how many moves have a problem.

    ``checks`` is None for a job that could not be measured, which leaves no
    rows rather than rows that look clean. The waypoints the rows describe
    are fingerprinted alongside them, so the panel can tell when they no
    longer match the job: see :func:`job_signature`.
    """
    rig.kinema_motion_check.clear()
    rig.kinema_motion_check_job = job_signature(rig) if checks is not None else ""
    rig.kinema_motion_check_collision = (
        collision_ops.collision_state(rig, bpy.context.scene) if checks is not None else ""
    )
    for check in checks or ():
        item = rig.kinema_motion_check.add()
        for field in motion_check.MoveCheck.__dataclass_fields__:
            setattr(item, field, getattr(check, field))
    return sum(1 for check in checks or () if motion_check.problems(check))


def job_signature(rig) -> str:
    """A fingerprint of the job as it stands: its motions, and the waypoints' poses.

    Each motion's name, waypoints, move and frames; each waypoint's name, pose
    and joints. Stored with a motion check and compared by the panel, so
    findings about a job that has since been edited are marked as such instead
    of being shown as current. The keys are not in it -- editing a curve by
    hand is only caught by Check Again -- but everything Generate Motion reads
    is. A rig saved before motions, not converted yet, is fingerprinted as that
    version did it: :func:`_signature_before_motions`.
    """
    if rig.kinema_job_version < JOB_VERSION:
        return _signature_before_motions(rig)
    parts = []
    relative = False
    # Once per object, however many waypoints are on it: a workpiece keyed on
    # every frame of a long job takes milliseconds, and this runs per redraw.
    placed = {}
    for waypoint in rig.kinema_waypoints:
        reference = references.reference_of(rig, waypoint)
        relative = relative or reference is not None
        # In its frame of reference: a waypoint on a workpiece is where it is on
        # the workpiece, wherever the workpiece is on this frame.
        pose = references.local_pose(waypoint)
        part = (
            waypoint.uid,
            waypoint.name,
            tuple(round(float(pose[r][c]), 6) for r in range(4) for c in range(4)),
            tuple(round(float(v), 6) for v in list(waypoint.q)[: waypoint.dof]),
        )
        if reference is not None:
            key = reference.as_pointer()
            if key not in placed:
                placed[key] = references.fingerprint(reference)
            # Only then, so a job with none keeps the fingerprint it always had.
            part += (tuple(round(float(v), 6) for v in waypoint.taught), placed[key])
        parts.append(part)
    if relative:
        # Where the robot stands decides where an object is from it; at the
        # base, a waypoint goes with the robot wherever that is.
        parts.append(("robot", references.fingerprint(rig)))
    for motion in sorted(rig.kinema_motions, key=lambda m: int(m.frame_start)):
        parts.append(
            (motion.name, motion.start, motion.end, motion.move,
             int(motion.frame_start), int(motion.frame_end))
        )
    return hashlib.sha1(repr(parts).encode("utf-8")).hexdigest()


def _signature_before_motions(rig) -> str:
    """The fingerprint a job saved before motions was stored with, taken as that
    version took it: the waypoints in frame order, each by name, frame, move,
    pose and joints.

    What a file saved then holds in its Motion Check, and in what it generated
    and optimized, so conversion can tell which of those still describe the job
    (:func:`convert_saved_before_motions`). It has to stay exactly as it was.
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
        return reported(self, check_motion(context, active_rig(context)))


def check_motion(context, rig) -> Outcome:
    """Play ``rig``'s job through once, solving as playback does, and check each
    move: what Check Motion does."""
    job, failed = _plan(rig)
    if job is None:
        return failed
    plan = job.spans
    samples = play_job(context, rig, plan)
    checks = check_job(context, rig, plan, samples=samples)
    if checks is not None:
        # Written before anything reads the job's fingerprint: the estimates
        # of frames needed, and the check stored with it.
        strips.retime(rig, job.retimes)
    estimate_frames_needed(context, rig, checks, samples)
    flagged = store_check(rig, checks)
    if checks is None:
        message = "Nothing checked: this rig has no TCP to measure the job by"
        if job.retimes:
            # Not written: a cancelled operator leaves no undo step.
            moved = ", ".join(f"'{m.name}'" for m, _, _ in job.retimes)
            message += (
                f". The NLA editor's retime of {moved} is taken into the Motions "
                f"list by the next Generate Motion"
            )
        return Outcome(False, "WARNING", message)
    message = f"Checked {len(plan)} move{'s' if len(plan) != 1 else ''}"
    message += f"; {_flagged_text(flagged)}" if flagged else "; nothing to report"
    for note in job.notes:
        message += f". {note}"
    return Outcome(True, "WARNING" if flagged or job.notes else "INFO", message, flagged)


def _plan(rig):
    """The rig's job, and None; or None, and why there is none: a warning for an
    overlap, which only the user can resolve, and an error for the rest."""
    try:
        return job_plan(rig), None
    except MotionsOverlap as exc:
        return None, Outcome(False, "WARNING", f"{exc}. Nothing was generated or checked")
    except WaypointError as exc:
        return None, Outcome(False, "ERROR", str(exc))


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
    KinemaMotion,
    KinemaMoveCheck,
    KINEMA_OT_add_waypoint,
    KINEMA_OT_update_waypoint,
    KINEMA_OT_goto_waypoint,
    KINEMA_OT_remove_waypoint,
    KINEMA_OT_add_motion,
    KINEMA_OT_remove_motion,
    KINEMA_OT_generate_motion,
    KINEMA_OT_check_motion,
)


def register_props() -> None:
    bpy.types.Object.kinema_waypoints = CollectionProperty(type=KinemaWaypoint)
    bpy.types.Object.kinema_motions = CollectionProperty(type=KinemaMotion)
    bpy.types.Object.kinema_active_motion = IntProperty(
        name="Active Motion", description="Row highlighted in the Motions list",
        default=0, min=0,
    )
    # 0 in a file saved before motions: see convert_saved_before_motions.
    bpy.types.Object.kinema_job_version = IntProperty(name="Job Version", default=0)
    # Where recording goes on from: the last waypoint recorded, and its frame.
    bpy.types.Object.kinema_last_recorded = StringProperty(name="Last Recorded", default="")
    bpy.types.Object.kinema_last_recorded_frame = IntProperty(
        name="Last Recorded Frame", default=0
    )
    # Why a job saved before motions made none: see convert_saved_before_motions.
    bpy.types.Object.kinema_unconverted = StringProperty(name="Not Converted", default="")
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
    # The action holding the job's keys, once laid out as NLA strips: see
    # ops/strips.py. None while the job is in the active action.
    bpy.types.Object.kinema_job_action = PointerProperty(
        name="Job Action",
        type=bpy.types.Action,
        description="The action Kinema's job strips show",
    )
    bpy.types.Object.kinema_generated_range = IntVectorProperty(
        name="Generated Range",
        description="First and last frame the last Generate Motion wrote",
        size=2,
        default=(0, -1),
    )
    # The waypoints that range was generated from, as job_signature() saw them.
    bpy.types.Object.kinema_generated_job = StringProperty(
        name="Generated Job",
        description="Fingerprint of the waypoints the last Generate Motion keyed",
        default="",
    )
    # The same, for the job as Optimize Motion last wrote it: Check Again
    # estimates the frames a move needs only there.
    bpy.types.Object.kinema_optimized_job = StringProperty(
        name="Optimized Job",
        description="Fingerprint of the waypoints the last Optimize Motion keyed",
        default="",
    )
    bpy.types.Object.kinema_active_waypoint = IntProperty(
        name="Active Waypoint",
        description="Row highlighted in the Waypoints list",
        default=0,
        min=0,
    )


def unregister_props() -> None:
    del bpy.types.Object.kinema_waypoints
    del bpy.types.Object.kinema_motions
    del bpy.types.Object.kinema_active_motion
    del bpy.types.Object.kinema_job_version
    del bpy.types.Object.kinema_last_recorded
    del bpy.types.Object.kinema_last_recorded_frame
    del bpy.types.Object.kinema_unconverted
    del bpy.types.Object.kinema_motion_check
    del bpy.types.Object.kinema_motion_check_job
    del bpy.types.Object.kinema_generated_range
    del bpy.types.Object.kinema_generated_job
    del bpy.types.Object.kinema_job_action
    del bpy.types.Object.kinema_optimized_job
    del bpy.types.Object.kinema_active_waypoint
