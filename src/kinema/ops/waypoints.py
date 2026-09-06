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

*LINEAR* keys the IK target's transform at both ends with LINEAR
interpolation and switches live IK on. The solver then tracks the target every
frame, and the tool travels the straight line between the two taught poses.
Measured on the 6-DoF fixture, the tool stays on that line to 0.000 mm.

Both taught values are stored: the tool pose *and* the joint vector. The pose
is the portable truth -- it can be replayed on a different robot -- while the
joint vector pins the configuration, which a pose alone cannot, since a pose
has up to eight solutions and nothing else records which one was taught.
"""

from __future__ import annotations

import bpy
from bpy.props import (
    CollectionProperty,
    EnumProperty,
    FloatVectorProperty,
    IntProperty,
    IntVectorProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Operator, PropertyGroup
from mathutils import Matrix

from ..rig import builder
from ..rig.waypoints import MOVE_JOINT, MOVE_LINEAR, WaypointError, spans
from ..ui.panel import active_rig
from .ik import key_joint_value, own_fcurve_containers

#: Empty display size, as a fraction of the robot's reach. Waypoints are read
#: at the scale of the robot, not the scene.
MARKER_SCALE = 0.08

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
    the viewport re-aims the move.

    Its ``matrix_basis`` *is* the pose in rig space: the marker is parented to
    the rig with an identity parent inverse, exactly so these two are the same
    number and no conversion can drift between them.
    """
    marker = waypoint.marker
    if marker is not None and marker.parent is not None:
        return marker.matrix_basis.copy()
    return _unflatten(waypoint.pose)


def marker_has_moved(waypoint, tolerance: float = 1e-6) -> bool:
    """True if the marker no longer agrees with the configuration taught for it.

    Matters only for a joint move, which replays the stored joint vector and so
    cannot follow a marker anywhere. A linear move solves for the pose and
    follows it without needing to be told.
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

        first, last = plan[0].start_frame, plan[-1].end_frame
        # One past the end covers the switch-off key a linear finish leaves.
        owned = (first, last + 1)
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
            for span in plan:
                if span.move == MOVE_JOINT:
                    self._key_joint_span(rig, span)
                else:
                    self._key_linear_span(rig, span, ik_name)
            if plan[-1].move == MOVE_LINEAR:
                # Leave the switch off past the end. Without this the last
                # linear span's "on" key is the final one on the channel, so
                # the solver goes on running for every frame after the job --
                # overwriting any joint animation out there, and doing the work
                # to no purpose while the user scrubs.
                rig.kinema_ik_enabled = False
                rig.keyframe_insert(data_path="kinema_ik_enabled", frame=last + 1)
            _set_linear_interpolation(rig, ik_name)
        rig.kinema_generated_range = owned
        context.scene.frame_set(original)

        # A joint move replays the joint vector it was taught with, so it
        # cannot follow a marker that has since been dragged somewhere else.
        # Silently ignoring the marker would be the worst of both: it moved on
        # screen and changed nothing.
        stranded = [
            span.end.name
            for span in plan
            if span.move == MOVE_JOINT and marker_has_moved(span.end)
        ]
        message = (
            f"Generated {len(plan)} move{'s' if len(plan) != 1 else ''} "
            f"over frames {first}-{last}"
        )
        if stranded:
            self.report(
                {"WARNING"},
                f"{message}. {', '.join(stranded)}: the marker has moved but a "
                f"joint move replays its taught configuration -- press Update, "
                f"or set it to Linear to follow the marker",
            )
            return {"FINISHED"}
        self.report({"INFO"}, message)
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
    def _key_linear_span(rig, span, ik_name: str) -> None:
        """Key the IK goal at both ends and switch live IK on for the span.

        The solver tracks the goal every frame, so keying the goal's own
        transform LINEAR is what makes the tool travel a straight line: the
        goal moves along it, and the arm follows.
        """
        rig.kinema_ik_enabled = True
        rig.keyframe_insert(data_path="kinema_ik_enabled", frame=span.start_frame)
        goal = rig.pose.bones[ik_name]
        goal.rotation_mode = "QUATERNION"
        for waypoint, frame in (
            (span.start, span.start_frame),
            (span.end, span.end_frame),
        ):
            goal.matrix = pose_of(waypoint)
            goal.keyframe_insert(data_path="location", frame=frame)
            goal.keyframe_insert(data_path="rotation_quaternion", frame=frame)


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
    passing the new one alone.
    """
    wanted = _generated_paths(rig, ik_name)
    for container in own_fcurve_containers(rig):
        for curve in list(container):
            if curve.data_path not in wanted:
                continue
            for point in reversed(list(curve.keyframe_points)):
                if first <= point.co[0] <= last + 1:
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
    KINEMA_OT_add_waypoint,
    KINEMA_OT_update_waypoint,
    KINEMA_OT_goto_waypoint,
    KINEMA_OT_remove_waypoint,
    KINEMA_OT_generate_motion,
)


def register_props() -> None:
    bpy.types.Object.kinema_waypoints = CollectionProperty(type=KinemaWaypoint)
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
    del bpy.types.Object.kinema_generated_range
    del bpy.types.Object.kinema_active_waypoint
