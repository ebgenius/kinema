"""Viewport handles on a rig: grab a joint, the IK control or the swivel directly.

Without these, moving anything by hand means Pose mode and picking a bone first, or
the sidebar's sliders. These are Blender's own gizmos -- the dial and arrow its
transform tools draw -- shown whenever the rig, or an object parented to it, is
selected, in Object or Pose mode.

**One source of truth.** A joint's dial is bound to the joint bone's own rotation
channel, a slide's arrow to its location, the swivel's dial to the ring's one channel.
Blender writes that channel, keys it on release when auto-keying is on, and pushes an
undo step, exactly as dragging the bone would. Nothing downstream can tell the
difference, so held joints, live IK, waypoints and baking need no change.

The IK control's axis handles are the exception. They work along the *tool's* axes,
and no single channel does that, so they write the control's whole matrix
(:mod:`..rig.gizmo_math`) and key it themselves.

Geometry-node gizmos looked like the obvious route and are not one: they only draw
for the active object's active modifier, an armature cannot carry a node modifier,
and a value living in a modifier input is not a bone channel anything else reads.
"""

from __future__ import annotations

from dataclasses import dataclass

import bpy
import numpy as np
from bpy.props import EnumProperty
from bpy.types import GizmoGroup
from mathutils import Matrix

from ..rig import builder, gizmo_math
from . import panel

SHOW_JOINTS = "JOINTS"
SHOW_IK = "IK"

DIAL = "GIZMO_GT_dial_3d"
ARROW = "GIZMO_GT_arrow_3d"
MOVE = "GIZMO_GT_move_3d"

ROLE_JOINT = "joint"
ROLE_SWIVEL = "swivel"
ROLE_IK_MOVE = "ik_move"
ROLE_IK_SLIDE = "ik_slide"
ROLE_IK_TURN = "ik_turn"

#: The tool's X, Y and Z, in the red, green and blue Blender uses for axes.
AXIS_COLOURS = ((0.9, 0.2, 0.2), (0.3, 0.8, 0.2), (0.2, 0.45, 0.95))
JOINT_COLOUR = (0.95, 0.65, 0.15)
SWIVEL_COLOUR = (0.2, 0.8, 0.85)
MOVE_COLOUR = (0.9, 0.9, 0.9)
ALPHA = 0.5
#: A swivel with no joint left to swing still turns, and does nothing; the panel
#: greys its slider, and the dial fades to say the same thing.
STALLED_ALPHA = 0.15


@dataclass(frozen=True)
class Handle:
    """One gizmo's job: its type, the bone it moves, and how dragging it moves it."""

    gizmo_type: str
    bone: str
    role: str
    #: Which of the tool's axes an IK slide or turn works along; -1 otherwise.
    axis: int = -1


def gizmo_rig(context: bpy.types.Context) -> bpy.types.Object | None:
    """The rig whose handles should be showing, or None.

    The same rig the sidebar acts on, so selecting a link mesh shows its robot's
    handles too. Object and Pose mode only: in Edit mode the pose is not what is
    being edited, and a dial turning a joint there would be a surprise.
    """
    if context.mode not in {"OBJECT", "POSE"}:
        return None
    rig = panel.active_rig(context)
    if rig is None or not getattr(rig, "kinema_gizmos", set()):
        return None
    return rig


def handles(rig: bpy.types.Object) -> list[Handle]:
    """Every handle this rig should have, in a stable order."""
    shown = getattr(rig, "kinema_gizmos", set())
    pose = rig.pose.bones
    found: list[Handle] = []
    if SHOW_JOINTS in shown:
        for pose_bone in builder.joint_bones(rig):
            prismatic = pose_bone.bone.get(builder.PROP_JOINT_TYPE) == "prismatic"
            found.append(Handle(ARROW if prismatic else DIAL, pose_bone.name, ROLE_JOINT))
    if SHOW_IK in shown:
        ik = rig.get(builder.PROP_IK_BONE)
        if ik and ik in pose:
            found.append(Handle(MOVE, ik, ROLE_IK_MOVE))
            found += [Handle(ARROW, ik, ROLE_IK_SLIDE, axis) for axis in range(3)]
            found += [Handle(DIAL, ik, ROLE_IK_TURN, axis) for axis in range(3)]
        swivel = rig.get(builder.PROP_SWIVEL_BONE)
        if swivel and swivel in pose:
            found.append(Handle(DIAL, swivel, ROLE_SWIVEL))
    return found


def bone_shown(pose_bone: bpy.types.PoseBone) -> bool:
    """Whether the viewport is showing this bone, so a handle on it should show."""
    bone = pose_bone.bone
    if bone.hide:
        return False
    collections = bone.collections
    return len(collections) == 0 or any(c.is_visible_effectively for c in collections)


def rest_frame(rig: bpy.types.Object, pose_bone: bpy.types.PoseBone) -> Matrix:
    """The frame a bone's own channels act in, in world space.

    Where the bone would be with its channels at zero, carried by wherever its
    parent actually is. Built from the parent rather than from the bone's own
    matrix, for two reasons:

    * a handle drawn in the bone's own frame turns with the joint it turns, so
      the dial would spin under the mouse while it was dragged, and
    * a joint dragged past its limit shows the limit, not the channel, and a
      frame recovered from what it shows would no longer be where the channel
      counts from.

    Deliberately not normalised: on a scaled rig, an arrow's offset measured in
    this frame is then in the channel's own units.
    """
    bone = pose_bone.bone
    parent = pose_bone.parent
    if parent is None:
        local = bone.matrix_local
    else:
        local = parent.matrix @ parent.bone.matrix_local.inverted() @ bone.matrix_local
    return rig.matrix_world @ local


def axis_frame(rig: bpy.types.Object, pose_bone: bpy.types.PoseBone) -> Matrix:
    """:func:`rest_frame` with the bone's Y -- a joint's axis -- as its Z.

    A dial turns about its own Z and an arrow points along it. ``BONE_TO_TOOL``
    already names the change of basis that puts bone Y on Z, and it keeps the frame
    right-handed, so a positive turn of the dial is a positive turn of the joint.
    """
    return rest_frame(rig, pose_bone) @ builder.BONE_TO_TOOL


def tool_frame(rig: bpy.types.Object, pose_bone: bpy.types.PoseBone) -> Matrix:
    """The tool frame a pose bone carries, in world space, as the TCP panel reports it."""
    return rig.matrix_world @ pose_bone.matrix @ builder.BONE_TO_TOOL


def handle_matrix(rig: bpy.types.Object, handle: Handle) -> Matrix | None:
    """Where ``handle`` is drawn, or None if its bone has gone."""
    pose_bone = rig.pose.bones.get(handle.bone)
    if pose_bone is None:
        return None
    if handle.role in (ROLE_JOINT, ROLE_SWIVEL):
        return axis_frame(rig, pose_bone)
    if handle.role == ROLE_IK_MOVE:
        # The move handle is bound to the control's location, which Blender
        # measures in the rest frame -- so that is where the handle has to sit.
        return rest_frame(rig, pose_bone)
    frame = np.array(tool_frame(rig, pose_bone))
    return Matrix(gizmo_math.axis_basis(frame, handle.axis).tolist())


def apply_ik_handle(
    rig: bpy.types.Object,
    handle: Handle,
    start,
    value: float,
    scene: bpy.types.Scene,
) -> None:
    """Put the IK control where dragging ``handle`` by ``value`` takes it.

    ``start`` is the control's armature-space matrix when the drag began, and every
    step is measured from it, as Blender's own handles do: accumulating per mouse
    event would compound rounding, and cancelling a drag would have nothing to
    return to.

    A slide's ``value`` is a distance in armature units -- the handle's frame
    carries the rig's scale, so a drag of one unit on screen is one here -- and a
    turn's is an angle about the tool axis through the tool point.
    """
    pose_bone = rig.pose.bones.get(handle.bone)
    if pose_bone is None:
        return
    start = np.asarray(start, dtype=float)
    axis = start[:3, :3] @ np.asarray(builder.BONE_TO_TOOL)[:3, handle.axis]
    if handle.role == ROLE_IK_SLIDE:
        goal = gizmo_math.slide(start, axis, value)
    else:
        goal = gizmo_math.turn(start, axis, value)
    pose_bone.matrix = Matrix(goal.tolist())

    # A handler-driven gizmo has no RNA property for Blender to key on release,
    # so it keys itself -- only when the user has asked for keys, like the rest.
    if scene.tool_settings.use_keyframe_insert_auto:
        frame = scene.frame_current
        for path in ("location", panel._rotation_channel(pose_bone)):
            pose_bone.keyframe_insert(path, frame=frame, group=pose_bone.name)


def _signature(rig: bpy.types.Object) -> tuple:
    return rig.name, tuple(handles(rig))


def _style(gizmo, handle: Handle) -> None:
    gizmo.use_undo = True
    gizmo.alpha = ALPHA
    gizmo.alpha_highlight = 1.0
    if handle.role in (ROLE_IK_SLIDE, ROLE_IK_TURN):
        colour = AXIS_COLOURS[handle.axis]
    elif handle.role == ROLE_SWIVEL:
        colour = SWIVEL_COLOUR
    elif handle.role == ROLE_IK_MOVE:
        colour = MOVE_COLOUR
    else:
        colour = JOINT_COLOUR
    gizmo.color = colour
    gizmo.color_highlight = tuple(min(1.0, c + 0.3) for c in colour)

    if handle.gizmo_type == DIAL:
        gizmo.use_draw_value = True
        gizmo.line_width = 2.0
        if handle.role == ROLE_IK_TURN:
            # Clipped to the half facing the viewer: three full rings on one
            # point hide each other and the robot under them.
            gizmo.draw_options = {"CLIP", "ANGLE_VALUE"}
        else:
            # The arc from the frame's Y is the joint's value, drawn as an angle.
            gizmo.draw_options = {"ANGLE_START_Y", "ANGLE_VALUE"}
    elif handle.gizmo_type == ARROW:
        gizmo.use_draw_value = True
        gizmo.draw_style = "NORMAL"
    elif handle.gizmo_type == MOVE:
        gizmo.draw_style = "RING_2D"
        gizmo.draw_options = {"FILL_SELECT", "ALIGN_VIEW"}
        gizmo.scale_basis = 0.2


class KINEMA_GGT_rig(GizmoGroup):
    """A selected rig's handles: its joints, its IK control and its swivel."""

    bl_idname = "KINEMA_GGT_rig"
    bl_label = "Kinema Rig Handles"
    bl_space_type = "VIEW_3D"
    bl_region_type = "WINDOW"
    bl_options = {"3D", "PERSISTENT"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return gizmo_rig(context) is not None

    def setup(self, context: bpy.types.Context) -> None:
        self._signature = None
        #: gizmo pointer -> its Handle. Pointers, because RNA wrappers are not
        #: identity-stable: the same gizmo can come back as a new Python object.
        self._handles: dict[int, Handle] = {}
        #: The IK axis handle being dragged: its gizmo, the control's matrix when
        #: the drag began, and how far it has gone.
        self._drag: dict | None = None
        rig = gizmo_rig(context)
        if rig is not None:
            self._build(rig)
            self._bind(rig)

    def refresh(self, context: bpy.types.Context) -> None:
        rig = gizmo_rig(context)
        if rig is None:
            return
        if self._signature != _signature(rig):
            self._build(rig)
        # Re-bound on every refresh, not once. A binding points into the pose,
        # and undo or a trip through Edit mode rebuilds the pose underneath it --
        # a dangling binding is a crash, not a wrong value.
        self._bind(rig)

    def draw_prepare(self, context: bpy.types.Context) -> None:
        rig = gizmo_rig(context)
        current = rig is not None and self._signature == _signature(rig)
        stalled = current and panel._swivel_section(rig) == "stalled"
        for gizmo in self.gizmos:
            pointer = gizmo.as_pointer()
            handle = self._handles.get(pointer)
            pose_bone = rig.pose.bones.get(handle.bone) if current and handle else None
            if pose_bone is None:
                # Out of date until the next refresh rebuilds it: show nothing
                # rather than a handle bound to the wrong bone.
                gizmo.hide = True
                continue
            gizmo.hide = not bone_shown(pose_bone)
            if handle.role == ROLE_SWIVEL:
                gizmo.alpha = STALLED_ALPHA if stalled else ALPHA
            if gizmo.is_modal:
                # Held where the drag began. An axis handle's offset is measured
                # from there, and moving its frame with the control as well would
                # count the drag twice.
                continue
            if self._drag is not None and self._drag["gizmo"] == pointer:
                self._drag = None
            gizmo.matrix_basis = handle_matrix(rig, handle)

    def invoke_prepare(self, context: bpy.types.Context, gizmo) -> None:
        handle = self._handles.get(gizmo.as_pointer())
        rig = gizmo_rig(context)
        if rig is None or handle is None or handle.role not in (ROLE_IK_SLIDE, ROLE_IK_TURN):
            return
        pose_bone = rig.pose.bones.get(handle.bone)
        if pose_bone is None:
            return
        self._drag = {
            "gizmo": gizmo.as_pointer(),
            "start": np.array(pose_bone.matrix),
            "value": 0.0,
        }

    def _build(self, rig: bpy.types.Object) -> None:
        self.gizmos.clear()
        self._handles = {}
        self._drag = None
        for handle in handles(rig):
            gizmo = self.gizmos.new(handle.gizmo_type)
            _style(gizmo, handle)
            self._handles[gizmo.as_pointer()] = handle
        self._signature = _signature(rig)

    def _bind(self, rig: bpy.types.Object) -> None:
        pose = rig.pose.bones
        for gizmo in self.gizmos:
            handle = self._handles.get(gizmo.as_pointer())
            if handle is None or gizmo.is_modal:
                continue
            pose_bone = pose.get(handle.bone)
            if pose_bone is None:
                continue
            if handle.role in (ROLE_JOINT, ROLE_SWIVEL):
                # index 1: the Y channel, the one aligned to the joint axis.
                channel = "location" if handle.gizmo_type == ARROW else "rotation_euler"
                gizmo.target_set_prop("offset", pose_bone, channel, index=1)
            elif handle.role == ROLE_IK_MOVE:
                gizmo.target_set_prop("offset", pose_bone, "location")
            else:
                self._bind_axis_handle(gizmo, rig.name, handle)

    def _bind_axis_handle(self, gizmo, rig_name: str, handle: Handle) -> None:
        pointer = gizmo.as_pointer()

        def get_value() -> float:
            drag = self._drag
            return drag["value"] if drag is not None and drag["gizmo"] == pointer else 0.0

        def set_value(value: float) -> None:
            drag = self._drag
            # Looked up by name on every call: holding the object across calls
            # would outlive an undo that replaced it.
            rig = bpy.data.objects.get(rig_name)
            if rig is None or drag is None or drag["gizmo"] != pointer:
                return
            drag["value"] = value
            apply_ik_handle(rig, handle, drag["start"], value, bpy.context.scene)

        gizmo.target_set_handler("offset", get=get_value, set=set_value)


def register_props() -> None:
    bpy.types.Object.kinema_gizmos = EnumProperty(
        name="Viewport Handles",
        description="Which of this rig's controls get handles in the viewport while it is selected",
        items=[
            (SHOW_JOINTS, "Joints", "A dial on each rotary joint and an arrow on each slide",
             "ORIENTATION_GIMBAL", 1),
            (SHOW_IK, "IK", "Move and turn handles on the IK target, and a dial on the swivel",
             "CON_KINEMATIC", 2),
        ],
        # Not animatable: which handles show is a working preference, not part of
        # the shot.
        options={"ENUM_FLAG"},
        default={SHOW_JOINTS, SHOW_IK},
    )


def unregister_props() -> None:
    del bpy.types.Object.kinema_gizmos


classes = (KINEMA_GGT_rig,)
