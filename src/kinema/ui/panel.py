"""Kinema's 3D viewport sidebar (N-panel).

Layout intent: an animator should see controls, not machinery. The everyday
case -- pick a robot, pose the TCP, tweak a joint -- is always visible, and
diagnostics stay collapsed.

The joint sliders drive the pose bones' real transform channels rather than
proxy properties, so they are keyframable exactly like any other bone channel:
click the dot beside a slider and you get a keyframe on the actual rig.
"""

from __future__ import annotations

import math

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Panel, PropertyGroup, UIList

from .. import runtime
from ..ops.import_robot import SETTING_NAMES, import_settings
from ..rig import builder
from ..rig.waypoints import MOVE_LINEAR

CATEGORY = "Kinema"


def active_rig(context: bpy.types.Context) -> bpy.types.Object | None:
    """The Kinema rig the panel should act on.

    Accepts the active object if it is a rig, otherwise the armature that owns
    the active object -- so selecting a link mesh still shows its robot's
    controls rather than an empty panel.
    """
    obj = context.object
    if builder.is_kinema_rig(obj):
        return obj
    if obj is not None and builder.is_kinema_rig(obj.parent):
        return obj.parent
    for candidate in context.selected_objects:
        if builder.is_kinema_rig(candidate):
            return candidate
    return None


class KinemaSceneProps(PropertyGroup):
    """Scene-level state. Per-rig state lives on the armature object itself.

    The import settings live here rather than on the catalog operator because
    that operator no longer carries REGISTER -- its redo panel used to re-run
    the entire import on every slider drag. Chosen before the import instead of
    adjusted after it.
    """

    last_import: StringProperty(
        name="Last Import",
        description="Most recently imported robot description",
        default="",
    )
    catalog_tag: EnumProperty(
        name="Category",
        description="Filter the catalog by robot type",
        items=lambda self, context: _tag_items(),
    )
    catalog_pick: StringProperty(
        name="Catalog Pick",
        description="Robot most recently looked up in the catalog",
        default="",
    )
    catalog_show_all: BoolProperty(
        name="Show All Variants",
        description=(
            "Include entries reviewed as duplicates, broken or partially "
            "supported. Off, only the recommended entry for each robot shows"
        ),
        default=False,
    )
    # Blender reads properties from a class's own __annotations__ and does not
    # walk base classes, so the shared definitions are merged in rather than
    # inherited. Same trick the import operators use.
    __annotations__.update(import_settings())


def _tag_items():
    from ..catalog import index as catalog

    try:
        tags = catalog.available_tags()
    except Exception:  # noqa: BLE001
        tags = []
    return [("", "All", "Every robot in the catalog")] + [
        (tag, tag.replace("_", " ").title(), f"Robots tagged '{tag}'") for tag in tags
    ]


def _draw_catalog(layout, context: bpy.types.Context) -> None:
    """The catalogue lookup: search, then what to do with the answer.

    Kinema cannot download the description, so the useful output is the clone
    command and -- above all -- *which file in the repository to open*. A user
    who clones mujoco_menagerie and is left to find one robot among 2466 files
    has not been helped.
    """
    from ..catalog import index as catalog

    props = context.scene.kinema

    layout.label(text="Kinema does not download robots.", icon="INFO")
    layout.operator("kinema.browse_catalog", text="Search Catalog…", icon="VIEWZOOM")
    layout.prop(props, "catalog_show_all")

    entry = catalog.get(props.catalog_pick) if props.catalog_pick else None
    if entry is None:
        return

    box = layout.box()
    box.label(text=entry.label, icon="OUTLINER_OB_ARMATURE")
    if entry.status:
        box.label(text=entry.note or entry.status, icon="ERROR")

    column = box.column(align=True)
    column.label(text="Clone command copied to clipboard:")
    column.label(text=f"{entry.clone_dir}", icon="FILE_FOLDER")
    if entry.file_path:
        column.label(text=f"then open: {entry.file_path}", icon="FILE_TICK")
    else:
        column.label(text="file unknown; look inside after cloning", icon="QUESTION")

    box.operator("kinema.open_catalog_repo", text="Open Repository", icon="URL")


class KinemaPanelBase:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = CATEGORY


class KINEMA_PT_main(KinemaPanelBase, Panel):
    bl_idname = "KINEMA_PT_main"
    bl_label = "Kinema"

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout

        column = layout.column(align=True)
        column.scale_y = 1.2
        column.operator("kinema.import_urdf", text="Import URDF File…", icon="FILE_FOLDER")

        header, body = layout.panel("kinema_find_robot", default_closed=True)
        header.label(text="Find a Robot")
        if body is not None:
            _draw_catalog(body, context)

        header, body = layout.panel("kinema_import_options", default_closed=True)
        header.label(text="Import Options")
        if body is not None:
            body.use_property_split = True
            for name in SETTING_NAMES:
                body.prop(context.scene.kinema, name)

        rig = active_rig(context)
        if rig is None:
            box = layout.box()
            box.label(text="No robot selected", icon="INFO")
            box.label(text="Import one, or select an existing rig.")
            return

        layout.separator()
        row = layout.row()
        row.label(text=rig.get(builder.PROP_ROBOT_NAME, rig.name), icon="ARMATURE_DATA")
        count = len(builder.joint_bones(rig))
        sub = row.row()
        sub.alignment = "RIGHT"
        sub.label(text=f"{count} DoF")


class KINEMA_PT_joints(KinemaPanelBase, Panel):
    bl_idname = "KINEMA_PT_joints"
    bl_parent_id = "KINEMA_PT_main"
    bl_label = "Joints (FK)"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return active_rig(context) is not None

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        rig = active_rig(context)
        joints = builder.joint_bones(rig)

        if not joints:
            layout.label(text="This rig has no movable joints", icon="INFO")
            return

        header = layout.row(align=True)
        header.operator("kinema.reset_pose", text="Rest Pose", icon="LOOP_BACK")
        header.operator("kinema.key_joints", text="Key All", icon="DECORATE_KEYFRAME")

        # Beside Rest Pose because it answers the same question -- "put the
        # robot back" -- for the half Rest Pose cannot reach. Only shown when
        # there is something to reset, so it does not imply the rig has meshes
        # when it has none.
        if builder.link_meshes(rig):
            layout.operator(
                "kinema.reset_link_meshes", text="Reset Meshes", icon="LOOP_BACK"
            )
            # Not a pose control, but it lives with the other "where is the
            # geometry" answers, and it warns loudly rather than sitting quiet
            # in a panel the user has scrolled past.
            layout.prop(rig, "kinema_meshes_at_file_origin", icon="EXPORT")
            if rig.kinema_meshes_at_file_origin:
                layout.label(text="Meshes are off the rig", icon="ERROR")

        from ..ops import velocity

        # Red on the slider of a joint moving faster than its limit here, so
        # the warning sits on the control that fixes it.
        too_fast = {reading.joint for reading in velocity.warnings(rig, context.scene)}

        column = layout.column(align=True)
        for pose_bone in joints:
            bone = pose_bone.bone
            is_revolute = bone.get(builder.PROP_JOINT_TYPE, "revolute") != "prismatic"
            row = column.row(align=True)
            row.alert = pose_bone.name in too_fast
            # index=1 is the Y channel: the one aligned to the joint axis.
            if is_revolute:
                row.prop(pose_bone, "rotation_euler", index=1, text=pose_bone.name)
            else:
                row.prop(pose_bone, "location", index=1, text=pose_bone.name)
            # Beside the slider it governs: a held joint is the one you move
            # here, or by dragging its bone, while IK solves the rest.
            held = getattr(pose_bone, "kinema_ik_hold", False)
            row.prop(
                pose_bone, "kinema_ik_hold", text="",
                icon="PINNED" if held else "UNPINNED", emboss=False,
            )

            limited = builder.PROP_LOWER in bone
            icon = "CON_ROTLIMIT" if limited else "BLANK1"
            sub = row.row(align=True)
            sub.enabled = limited
            sub.label(text="", icon=icon)


class KINEMA_PT_velocity(KinemaPanelBase, Panel):
    """Each joint's speed at this frame, against its velocity limit."""

    bl_idname = "KINEMA_PT_velocity"
    bl_parent_id = "KINEMA_PT_main"
    bl_label = "Velocity Limits"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return active_rig(context) is not None

    def draw_header(self, context: bpy.types.Context) -> None:
        from ..ops import velocity

        # Flagged in the header too, so a closed section still says so.
        if velocity.warnings(active_rig(context), context.scene):
            self.layout.label(text="", icon="ERROR")

    def draw(self, context: bpy.types.Context) -> None:
        from ..ops import velocity
        from ..rig.velocity import format_speed

        layout = self.layout
        rig = active_rig(context)
        scene = context.scene
        readings = velocity.readings(rig, scene)

        if not readings:
            layout.label(text="No velocity limits on this rig", icon="INFO")
            layout.operator(
                "kinema.load_joint_limits", text="Load Joint Limits…", icon="FILE_FOLDER"
            )
            return

        layout.prop(rig, "kinema_ignore_velocity")
        ignored = velocity.ignored(rig)
        degrees = scene.unit_settings.system_rotation == "DEGREES"

        column = layout.column(align=True)
        column.active = not ignored
        for reading in readings:
            row = column.row(align=True)
            row.alert = reading.over and not ignored
            row.label(text=reading.joint, icon="ERROR" if row.alert else "BLANK1")
            speed, limit = (
                format_speed(value, prismatic=reading.prismatic, degrees=degrees)
                for value in (reading.speed, reading.limit)
            )
            sub = row.row(align=True)
            sub.alignment = "RIGHT"
            sub.label(text=f"{speed}  of  {limit}")

        fps = scene.render.fps / scene.render.fps_base
        layout.label(text=f"Frame {scene.frame_current} at {fps:g} fps", icon="TIME")
        if any(reading.speed is None for reading in readings):
            # Only a joint live IK drives can read "—": its last frame is
            # remembered, not stored, and a jump leaves nothing to compare.
            layout.label(text="— : play or step a frame to measure", icon="INFO")
        layout.operator(
            "kinema.load_joint_limits", text="Load Joint Limits…", icon="FILE_FOLDER"
        )


def _joint_indices(rig: bpy.types.Object) -> dict[str, int]:
    """Bone name -> its index into ``builder.joint_bones``.

    That index, not the name, is what ``kinema_ik_tip`` stores: an integer is
    keyframable and a bone reference is not.
    """
    return {bone.name: index for index, bone in enumerate(builder.joint_bones(rig))}


def tcp_exists(rig: bpy.types.Object) -> bool:
    """Whether this rig has a TCP marker that is actually there.

    The property outlives the bone -- delete the marker in Edit mode and
    ``kinema_tcp_bone`` still names it. Testing the property's bare truthiness
    had the panel say "No TCP on this rig" and offer "Update TCP" in the same
    breath, so both halves ask this instead.
    """
    if rig is None:
        return False
    name = rig.get(builder.PROP_TCP_BONE)
    return bool(name) and name in rig.pose.bones


def _rotation_channel(obj: bpy.types.Object) -> str:
    """The rotation property ``obj.rotation_mode`` actually evaluates."""
    if obj.rotation_mode == "QUATERNION":
        return "rotation_quaternion"
    if obj.rotation_mode == "AXIS_ANGLE":
        return "rotation_axis_angle"
    return "rotation_euler"


def tip_index_of(rig: bpy.types.Object, pose_bone) -> int | None:
    """The ``kinema_ik_tip`` value that aims at ``pose_bone``, or None.

    Joint bones map to their index; the TCP marker maps to -1, the default.
    Everything else -- Root, the IK control -- is not a thing the solver can
    aim at and gets no radio button.
    """
    if builder.PROP_JOINT_NAME in pose_bone.bone:
        return _joint_indices(rig).get(pose_bone.name)
    tcp_name = rig.get(builder.PROP_TCP_BONE) or builder.TCP_BONE
    return -1 if pose_bone.name == tcp_name else None


def active_bone(rig: bpy.types.Object) -> bpy.types.PoseBone | None:
    """The bone the Bones list has highlighted, if it still exists."""
    if rig is None:
        return None
    bones = rig.pose.bones
    index = getattr(rig, "kinema_active_bone_index", 0)
    return bones[index] if 0 <= index < len(bones) else None


class KINEMA_UL_bones(UIList):
    """One row per bone: aim IK at it, and hang something off it.

    Both controls belong on the same row because they answer the same
    question -- what is this bone for? -- and because dressing a robot means
    walking the whole list once, not opening a dialog per link.
    """

    def draw_item(
        self, context, layout, data, item, icon, active_data, active_propname, index
    ):
        rig = item.id_data
        attachment = builder.bone_attachment(rig, item.name)
        index = tip_index_of(rig, item)

        split = layout.split(factor=0.42, align=True)

        left = split.row(align=True)
        target = left.row(align=True)
        if index is not None:
            selected = index == getattr(rig, "kinema_ik_tip", -1)
            operator = target.operator(
                "kinema.set_ik_tip",
                text="",
                icon="RADIOBUT_ON" if selected else "RADIOBUT_OFF",
                emboss=False,
            )
            operator.index = index
        else:
            # Root and the IK control cannot be solved to, so they get a spacer
            # rather than a dead button. The TCP marker is not among them: it is
            # the default target, and without a row of its own there would be no
            # way back to it from the list once a joint had been picked.
            target.label(text="", icon="BLANK1")
        left.label(text=item.name)

        right = split.row(align=True)
        right.prop(item, "kinema_attach_type", text="", icon_only=True, expand=True)
        if item.kinema_attach_type == "COLLECTION":
            right.prop(item, "kinema_attach_collection", text="")
        else:
            right.prop(item, "kinema_attach_object", text="")
        remove = right.row(align=True)
        remove.enabled = attachment is not None
        remove.operator(
            "kinema.detach_from_bone", text="", icon="X", emboss=False
        ).bone = item.name

    def filter_items(self, context, data, propname):
        """Hide the IK control bone; it is a goal, not a part of the robot."""
        bones = getattr(data, propname)
        rig = data.id_data
        ik_name = rig.get(builder.PROP_IK_BONE)

        flags = [
            0 if bone.name == ik_name else self.bitflag_filter_item for bone in bones
        ]
        if self.filter_name:
            matched = bpy.types.UI_UL_list.filter_items_by_name(
                self.filter_name, self.bitflag_filter_item, bones, "name"
            )
            flags = [f & m for f, m in zip(flags, matched, strict=True)]
        return flags, []


class KINEMA_PT_bones(KinemaPanelBase, Panel):
    bl_idname = "KINEMA_PT_bones"
    bl_parent_id = "KINEMA_PT_main"
    bl_label = "Bones"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return active_rig(context) is not None

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        rig = active_rig(context)

        layout.template_list(
            "KINEMA_UL_bones", "", rig.pose, "bones", rig, "kinema_active_bone_index",
            rows=6,
        )

        pose_bone = active_bone(rig)
        if pose_bone is None:
            return

        attachment = builder.bone_attachment(rig, pose_bone.name)
        if attachment is None:
            layout.label(text=f"Nothing attached to '{pose_bone.name}'", icon="INFO")
            return

        row = layout.row(align=True)
        row.operator(
            "kinema.select_attachment", text=attachment.name, icon="RESTRICT_SELECT_OFF"
        ).bone = pose_bone.name
        row.operator(
            "kinema.detach_from_bone", text="", icon="UNLINKED"
        ).bone = pose_bone.name

        # The offset *is* the attachment's own transform -- the bone parenting
        # is set up so that a zero transform sits it exactly on the joint --
        # which is why these are plain object channels and keyframe like any
        # other.
        header, body = layout.panel("kinema_attach_offset", default_closed=True)
        header.label(text="Offset from the Bone")
        if body is not None:
            body.use_property_split = True
            body.use_property_decorate = True
            body.prop(attachment, "location")
            # An attachment keeps its source's rotation_mode, so the Euler
            # channel is not necessarily the one Blender is reading -- drawing
            # it unconditionally would offer a field that silently does nothing
            # on a quaternion or axis-angle source.
            body.prop(attachment, "rotation_mode", text="Rotation Mode")
            body.prop(attachment, _rotation_channel(attachment), text="Rotation")
            body.prop(attachment, "scale")
            body.operator(
                "kinema.reset_attachment_offset", text="Reset", icon="LOOP_BACK"
            ).bone = pose_bone.name


class KINEMA_UL_waypoints(UIList):
    """One row per taught waypoint: name, how it is arrived at, and when.

    Sorted by frame rather than by list position, because the frame *is* the
    order -- see rig/waypoints.py. The list's own index is only ever "which row
    is highlighted".
    """

    def draw_item(
        self, context, layout, data, item, icon, active_data, active_propname, index
    ):
        row = layout.row(align=True)
        row.prop(item, "name", text="", emboss=False, icon="EMPTY_ARROWS")
        sub = row.row(align=True)
        sub.scale_x = 0.55
        sub.prop(item, "move", text="")
        sub.prop(item, "frame", text="")
        row.operator(
            "kinema.goto_waypoint", text="", icon="PLAY", emboss=False
        ).index = index

    def filter_items(self, context, data, propname):
        items = getattr(data, propname)
        order = bpy.types.UI_UL_list.sort_items_helper(
            list(enumerate(items)), lambda pair: pair[1].frame
        )
        flags = [self.bitflag_filter_item] * len(items)
        if self.filter_name:
            flags = bpy.types.UI_UL_list.filter_items_by_name(
                self.filter_name, self.bitflag_filter_item, items, "name"
            )
        return flags, order


class KINEMA_PT_waypoints(KinemaPanelBase, Panel):
    bl_idname = "KINEMA_PT_waypoints"
    bl_parent_id = "KINEMA_PT_main"
    bl_label = "Waypoints"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return active_rig(context) is not None

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        rig = active_rig(context)

        row = layout.row(align=True)
        row.operator("kinema.add_waypoint", text="Record", icon="KEYFRAME_HLT")
        row.operator("kinema.update_waypoint", text="Update", icon="FILE_REFRESH")
        row.operator("kinema.remove_waypoint", text="", icon="X")

        if not rig.kinema_waypoints:
            layout.label(text="Pose the robot, then Record", icon="INFO")
            return

        layout.template_list(
            "KINEMA_UL_waypoints", "", rig, "kinema_waypoints",
            rig, "kinema_active_waypoint", rows=4,
        )

        column = layout.column()
        column.enabled = len(rig.kinema_waypoints) > 1
        column.scale_y = 1.2
        column.operator("kinema.generate_motion", text="Generate Motion", icon="TRACKING")

        # The same test the operator makes, not a truthier one: a rig whose IK
        # bone was deleted keeps the property, and checking only that would
        # leave Generate Motion looking available until it was pressed.
        from ..ops.waypoints import ik_bone_name

        wants_ik = any(w.move == MOVE_LINEAR for w in rig.kinema_waypoints)
        if wants_ik and ik_bone_name(rig) is None:
            layout.label(text="Linear moves need an IK target", icon="ERROR")


class KINEMA_PT_external_axes(KinemaPanelBase, Panel):
    """Rails, turntables, positioners and spindles the description does not have."""

    bl_idname = "KINEMA_PT_external_axes"
    bl_parent_id = "KINEMA_PT_main"
    bl_label = "External Axes"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return active_rig(context) is not None

    def draw(self, context: bpy.types.Context) -> None:
        from ..ops.external_axes import MOUNT_ICONS, MOUNT_LABELS, axis_bones

        layout = self.layout
        rig = active_rig(context)
        axes = axis_bones(rig)
        if axes:
            column = layout.column(align=True)
            for pose_bone in axes:
                mount = pose_bone.bone.get(builder.PROP_EXTERNAL, "")
                row = column.row(align=True)
                row.label(text=pose_bone.name, icon=MOUNT_ICONS.get(mount, "BLANK1"))
                sub = row.row(align=True)
                sub.alignment = "RIGHT"
                sub.label(text=MOUNT_LABELS.get(mount, mount))
                row.operator(
                    "kinema.remove_external_axis", text="", icon="X", emboss=False
                ).bone = pose_bone.name
            # Their sliders and hold pins are with the other joints.
            layout.label(text="Move them in Joints (FK)", icon="INFO")
        else:
            layout.label(text="A track, turntable, positioner or spindle", icon="INFO")
        layout.operator("kinema.add_external_axis", text="Add External Axis…", icon="ADD")


class KINEMA_PT_tcp(KinemaPanelBase, Panel):
    bl_idname = "KINEMA_PT_tcp"
    bl_parent_id = "KINEMA_PT_main"
    bl_label = "Tool Centre Point"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return active_rig(context) is not None

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        rig = active_rig(context)
        tcp_name = rig.get(builder.PROP_TCP_BONE)

        exists = tcp_exists(rig)
        if exists:
            self._draw_readout(layout, rig, rig.pose.bones[tcp_name])
        else:
            layout.label(text="No TCP on this rig", icon="INFO")

        self._draw_placement(layout, rig, exists=exists)

    @staticmethod
    def _draw_readout(layout, rig, tcp) -> None:
        """Where the tool is and which way it faces.

        Reported in the *tool* frame, not the bone's: a bone's own Y is always
        head-to-tail, so its raw matrix would describe the marker rather than
        the thing the marker stands for.
        """
        world = rig.matrix_world @ tcp.matrix @ builder.BONE_TO_TOOL
        location = world.translation
        roll, pitch, yaw = (math.degrees(a) for a in world.to_euler("XYZ"))

        box = layout.box()
        column = box.column(align=True)
        column.label(
            text=f"Link: {rig.get(builder.PROP_TCP_LINK, '—')}", icon="EMPTY_AXIS"
        )
        column.label(
            text=f"X {location.x:+.4f}   Y {location.y:+.4f}   Z {location.z:+.4f}"
        )
        column.label(text=f"R {roll:+7.2f}   P {pitch:+7.2f}   Y {yaw:+7.2f}")

    @staticmethod
    def _draw_placement(layout, rig, *, exists: bool) -> None:
        # The dialog is the easy way: offsets, the 3D cursor or an object's origin,
        # previewed before anything moves. The fields below stay for typing.
        row = layout.row()
        row.scale_y = 1.2
        row.operator(
            "kinema.edit_tcp", text="Edit TCP…" if exists else "Place TCP…",
            icon="EMPTY_ARROWS",
        )
        column = layout.column(align=True)
        column.use_property_split = True
        column.prop_search(
            rig, "kinema_tcp_parent", rig.pose, "bones", text="Parent Bone",
            icon="BONE_DATA",
        )

        header, body = layout.panel("kinema_tcp_offset", default_closed=True)
        header.label(text="Tool Offset")
        if body is not None:
            body.use_property_split = True
            body.label(text="From the flange link frame", icon="INFO")
            body.prop(rig, "kinema_tcp_offset", text="Location")
            body.prop(rig, "kinema_tcp_rpy", text="Rotation")
            body.operator("kinema.reset_tcp_offset", text="Reset", icon="LOOP_BACK")

        parent = rig.kinema_tcp_parent
        chosen = rig.pose.bones.get(parent) if parent else None
        # The search offers every bone, but only joint bones carry a link frame
        # for the offset to be measured in -- so Root, the marker itself and the
        # IK control are all things the operator refuses. Checked here too, or
        # picking one leaves an enabled button that is guaranteed to cancel.
        usable = chosen is not None and builder.PROP_JOINT_NAME in chosen.bone

        row = layout.row(align=True)
        row.scale_y = 1.2
        # Disabled rather than hidden: the button is where the eye goes, and a
        # button that vanishes is harder to understand than one that explains
        # what it wants.
        row.enabled = usable
        label = "Update TCP" if exists else "Create TCP"
        row.operator(
            "kinema.set_tcp", text=label, icon="EMPTY_ARROWS"
        ).bone = parent

        if chosen is None:
            layout.label(text="Pick a parent bone first", icon="INFO")
        elif not usable:
            layout.label(text=f"'{parent}' is not a joint bone", icon="ERROR")
        layout.operator(
            "kinema.set_tcp", text="Move TCP to Active Bone", icon="BONE_DATA"
        ).bone = ""


class KINEMA_PT_ik(KinemaPanelBase, Panel):
    bl_idname = "KINEMA_PT_ik"
    bl_parent_id = "KINEMA_PT_main"
    bl_label = "Inverse Kinematics"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return active_rig(context) is not None

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        rig = active_rig(context)
        ik_bone = rig.get(builder.PROP_IK_BONE)

        if not ik_bone or ik_bone not in rig.pose.bones:
            layout.operator("kinema.add_ik", text="Add IK Target", icon="CON_KINEMATIC")
            layout.label(text="Adds a keyframable control at the tool.", icon="BLANK1")
            return

        column = layout.column(align=True)
        column.prop(rig, "kinema_ik_enabled", toggle=True,
                    icon="PLAY" if rig.kinema_ik_enabled else "PAUSE")
        column.prop(rig, "kinema_solver_mode", text="")

        # The friendly way to change the tip is the radio column in the Bones
        # list; this row shows where it landed and keys it.
        #
        # Deliberately no property decorator. The dot inserts a key with the
        # user's default interpolation, and an interpolated index ramps through
        # every value between two tips -- solving chains nobody asked for. The
        # button forces the channel to step instead, so the offered way to key
        # this is the one that behaves.
        from ..solver import manager

        tip = layout.column(align=True)
        tip.use_property_split = True
        tip.use_property_decorate = False
        row = tip.row(align=True)
        row.prop(rig, "kinema_ik_tip", text="Target Bone")
        row.operator("kinema.key_ik_tip", text="", icon="DECORATE_KEYFRAME")
        tip.label(text=f"Solving to '{manager.tip_bone(rig)}'", icon="BONE_DATA")

        row = layout.row(align=True)
        row.operator("kinema.snap_ik", text="Snap to Tool", icon="SNAP_ON")
        row.operator("kinema.remove_ik", text="", icon="X")

        self._draw_solutions(layout, rig)
        self._draw_swivel(layout, rig)

        layout.operator("kinema.bake_ik", text="Bake to Keyframes", icon="RENDER_ANIMATION")
        self._draw_readout(layout, rig)

    @staticmethod
    def _draw_solutions(layout, rig) -> None:
        """Cycle among the configurations that reach this pose."""
        from ..ops import ik as ik_ops

        header, body = layout.panel("kinema_solutions", default_closed=True)
        header.label(text="Configuration")
        if body is None:
            return

        count = ik_ops.solution_count(rig)
        body.operator(
            "kinema.find_solutions", text="Find Solutions", icon="VIEWZOOM"
        )
        if not count:
            body.label(text="An arm reaches a pose several ways.", icon="BLANK1")
            return

        row = body.row(align=True)
        row.operator("kinema.apply_solution", text="", icon="TRIA_LEFT").step = -1
        current = int(rig.kinema_active_solution)
        row.label(text=f"Solution {min(current + 1, count)} of {count}")
        row.operator("kinema.apply_solution", text="", icon="TRIA_RIGHT").step = 1

        if ik_ops.solutions_are_stale(rig):
            # They describe how to reach one pose. The target has moved, so
            # they describe nothing -- better said than silently offered.
            body.label(text="The target moved; search again", icon="ERROR")

    @staticmethod
    def _draw_swivel(layout, rig) -> None:
        """The elbow swivel: offered with a joint to spare, and kept once added."""
        state = _swivel_section(rig)
        if state is None:
            # Six unheld joints against a six-DoF pose leave nothing to swing.
            # Not an error, just absent -- and a held rail is why a seven-joint
            # rail robot does not offer one.
            return

        header, body = layout.panel("kinema_swivel", default_closed=True)
        # Flagged in the header too, so a collapsed section still says so.
        header.label(text="Elbow Swivel", icon="ERROR" if state == "stalled" else "NONE")
        if body is None:
            return

        free = _free_joint_count(rig)
        if state == "offer":
            body.operator("kinema.add_swivel", text="Add Elbow Swivel", icon="CON_ROTLIKE")
            body.label(text=f"{free} joints: the elbow can swing.", icon="BLANK1")
            return

        handle = rig.pose.bones[rig.get(builder.PROP_SWIVEL_BONE)]
        # The one channel that matters, with its own keyframe dot: the elbow's
        # angle round the shoulder-to-wrist line. The ring in the viewport turns
        # the same channel.
        row = body.row()
        row.enabled = state == "active"
        row.prop(handle, "rotation_euler", index=1, text="Swivel")
        if state == "stalled":
            # Said, not just greyed: the ring is still in the viewport and still
            # turns, and nothing else would explain why it no longer does anything.
            body.label(text=f"Only {free} joints left to IK: nothing to swing", icon="ERROR")
            body.label(text="Release a held joint to use it again", icon="BLANK1")
        body.label(
            text=(
                f"{rig.get(builder.PROP_SWIVEL_SHOULDER, '?')} - "
                f"{rig.get(builder.PROP_ELBOW_JOINT, '?')} - "
                f"{rig.get(builder.PROP_SWIVEL_WRIST, '?')}"
            ),
            icon="BONE_DATA",
        )
        body.prop(rig, "kinema_elbow_strength")
        if getattr(rig, "kinema_solver_mode", "PYROKI") != "PYROKI":
            body.label(text="Only the PyRoki solver swings it", icon="ERROR")
        body.operator("kinema.remove_swivel", text="Remove", icon="X")

    @staticmethod
    def _draw_readout(layout, rig) -> None:
        from .. import handlers

        elapsed = handlers.last_solve_ms(rig)
        solver = manager_state(rig)
        box = layout.box()
        info = box.column(align=True)
        if elapsed is not None:
            budget = _solve_budget_ms()
            icon = "CHECKMARK" if elapsed <= budget else "ERROR"
            info.label(text=f"Last solve: {elapsed:.1f} ms", icon=icon)
            if elapsed > budget:
                info.label(text="Over budget; live updates paused", icon="BLANK1")
        if solver is not None and solver.last_result is not None:
            result = solver.last_result
            info.label(text=result.summary, icon="BLANK1")
        if solver is not None and solver.pyroki_error:
            info.label(text="PyRoki unavailable for this rig:", icon="INFO")
            info.label(text=solver.pyroki_error[:46], icon="BLANK1")
        from ..solver import manager

        if manager.deferred(rig.name):
            info.label(text="PyRoki waits until the edit is done", icon="SORTTIME")
            info.label(text="Solving on NumPy meanwhile", icon="BLANK1")


def manager_state(rig):
    from ..solver import manager

    return manager._cache.get(rig.name)


def _chain_dof(rig) -> int:
    """How many joints the solver drives, without building one to find out.

    ``draw`` runs on every region redraw, and ``manager.get_solver`` builds and
    caches on a miss -- so calling it here put chain extraction on the UI thread
    on the first redraw after anything invalidated the cache, including simply
    reopening the file.

    The cached solver is exact when there is one. Otherwise the rig's own joint
    bones are the right answer for every rig whose IK aims at the tool, which is
    the default and the overwhelming case; a tip set part-way up the chain makes
    this an over-estimate, and the operator behind the button re-checks against
    the real chain and refuses with a reason.
    """
    solver = manager_state(rig)
    if solver is not None:
        return solver.chain.dof
    return len(builder.joint_bones(rig))


def _free_joint_count(rig) -> int:
    """Joints left to IK: the solver's chain, less the ones held by hand.

    Both halves have to describe the same joints. Taking the length from the
    solver's chain while counting holds across every joint bone miscounts a rig
    whose tip sits part-way up: a held joint past the tip is not in the chain,
    and subtracting it anyway can hide a swivel the chain has room for.

    No solver is built to find out, for the same reason as :func:`_chain_dof`.
    """
    solver = manager_state(rig)
    if solver is not None:
        names = list(solver.chain.bone_names)
    else:
        names = [pb.name for pb in builder.joint_bones(rig)]
    pose = rig.pose.bones
    return sum(
        1 for name in names if not getattr(pose.get(name), "kinema_ik_hold", False)
    )


def _swivel_section(rig) -> str | None:
    """What the Elbow Swivel section shows: offer, active, stalled -- or None.

    A swivel that exists always keeps its section, whatever the count. Holding a
    joint can take the chain down to six free joints, and the pin is keyframable,
    so hiding the section then took the Remove button with it and would make the
    whole thing blink in and out over a shot -- while the ring stayed in the
    viewport, turning and doing nothing. Only the offer to add one depends on
    there being a joint to spare.
    """
    name = rig.get(builder.PROP_SWIVEL_BONE)
    has_swivel = bool(name) and name in rig.pose.bones
    spare = _free_joint_count(rig) > 6
    if has_swivel:
        return "active" if spare else "stalled"
    return "offer" if spare else None


def _solve_budget_ms() -> float:
    from ..prefs import get_prefs

    prefs = get_prefs()
    return float(prefs.solve_timeout_ms) if prefs else 33.0


class KINEMA_PT_status(KinemaPanelBase, Panel):
    bl_idname = "KINEMA_PT_status"
    bl_parent_id = "KINEMA_PT_main"
    bl_label = "Solver"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        column = layout.column(align=True)

        if runtime.solver_available():
            column.label(text="PyRoki ready", icon="CHECKMARK")
        elif runtime.solver_error():
            column.label(text="PyRoki unavailable", icon="ERROR")
            column.label(text=runtime.solver_error()[:48], icon="BLANK1")
            column.label(text="Using NumPy fallback", icon="BLANK1")
        else:
            column.label(text="Solver loading…", icon="SORTTIME")
            column.label(text="Using NumPy fallback", icon="BLANK1")

        layout.operator("kinema.check_dependencies", text="Re-check", icon="FILE_REFRESH")


classes = (
    KinemaSceneProps,
    KINEMA_UL_bones,
    KINEMA_UL_waypoints,
    KINEMA_PT_main,
    KINEMA_PT_joints,
    KINEMA_PT_velocity,
    KINEMA_PT_bones,
    KINEMA_PT_waypoints,
    KINEMA_PT_external_axes,
    KINEMA_PT_tcp,
    KINEMA_PT_ik,
    KINEMA_PT_status,
)


def _on_ik_tip_changed(rig, context) -> None:
    """Forget the cached goal so the next update re-solves the new chain."""
    from .. import handlers

    handlers.reset(rig.name)


def _on_elbow_strength_changed(rig, context) -> None:
    """Re-solve when the pull changes, not only when the bone moves.

    The strength scales a cost, so changing it changes the answer. A property
    registered without an update callback tags nothing for re-evaluation, so
    the live handler would never run and the slider would look inert in exactly
    the way dragging the elbow bone used to.
    """
    from .. import handlers

    handlers.reset(rig.name)
    rig.update_tag()


def _on_ik_hold_changed(pose_bone, context) -> None:
    """Re-solve when a joint is held or released: the problem changed shape."""
    from .. import handlers

    rig = pose_bone.id_data
    handlers.reset(rig.name)
    rig.update_tag()


#: Guards the revert below against the callback it would otherwise re-enter,
#: the same shape as attach._attaching and handlers._solving.
_reverting_file_origin = False


def _on_meshes_at_file_origin(rig, context) -> None:
    """Run the move through the operator, so ticking the box is undoable.

    A property callback pushes no undo step of its own, and this one moves
    every link mesh in the rig. The rig is named rather than left to the
    context: the box belongs to one armature, and a scene with two robots in
    it would otherwise move whichever happened to be active.

    Two things this has to be careful about, both of them consequences of the
    property living on ``bpy.types.Object``:

    *Every* object in the file carries it, robot or not. Setting it on a cube
    has nothing to move, and must not become an instruction to move something
    else.

    And the property changes before the callback runs, so a move that cannot
    happen leaves the box ticked over meshes that never went anywhere -- the
    panel then reports "Meshes are off the rig" about a rig whose meshes are
    exactly where they were. A rig imported before the bake was recorded is
    precisely that case, so the tick comes back off with the cancellation.
    """
    global _reverting_file_origin

    if _reverting_file_origin or not builder.is_kinema_rig(rig):
        return

    enabled = rig.kinema_meshes_at_file_origin
    if "CANCELLED" not in bpy.ops.kinema.meshes_to_file_origin(
        rig=rig.name, enabled=enabled
    ):
        return

    _reverting_file_origin = True
    try:
        rig.kinema_meshes_at_file_origin = not enabled
    finally:
        _reverting_file_origin = False


def register_props() -> None:
    bpy.types.Scene.kinema = PointerProperty(type=KinemaSceneProps)
    # Registered on Object rather than kept as raw custom properties so the
    # panel can draw a real checkbox and dropdown, and so the values round-trip
    # through a saved .blend.
    bpy.types.Object.kinema_ik_enabled = BoolProperty(
        name="Live IK",
        description="Solve continuously as the IK target moves",
        default=False,
    )
    bpy.types.Object.kinema_solver_mode = EnumProperty(
        name="Solver",
        description="Which IK backend drives this rig",
        items=[
            ("PYROKI", "PyRoki", "Limit- and singularity-aware nonlinear solver"),
            ("NUMPY", "NumPy", "Lightweight damped least squares"),
            ("OFF", "Off", "No IK; pose the joints directly"),
        ],
        default="PYROKI",
    )
    # An index into builder.joint_bones(), not a bone name, because this has to
    # be keyframable: Blender animates integers and does not animate strings.
    # -1 means "the TCP marker", which is where every rig starts and how it
    # behaved before the tip could be moved at all.
    bpy.types.Object.kinema_ik_tip = IntProperty(
        name="IK Target Bone",
        description=(
            "Which joint bone the solver aims at, by index. -1 aims at the "
            "tool centre point. Keyframable, so a shot can hand the goal from "
            "one bone to another"
        ),
        default=-1,
        min=-1,
        update=_on_ik_tip_changed,
    )
    # UI state, so it belongs to the rig rather than the scene: two rigs in one
    # file each remember their own highlighted row.
    bpy.types.Object.kinema_active_bone_index = IntProperty(
        name="Active Bone",
        description="Row highlighted in the Bones list",
        default=0,
        min=0,
    )
    # Saved with the .blend on purpose: a file closed in this state opens in it,
    # and a tick that silently lied about where the meshes are would be worse
    # than one that persists.
    bpy.types.Object.kinema_meshes_at_file_origin = BoolProperty(
        name="Meshes at File Origin",
        description=(
            "Move every link mesh to the coordinates its own file uses, so it "
            "can be edited and exported as a drop-in replacement. The rig's "
            "kinematics still work, but the meshes no longer follow the bones "
            "correctly -- turn this off before posing"
        ),
        default=False,
        update=_on_meshes_at_file_origin,
    )
    # Which of the found configurations is showing. An index rather than the
    # values themselves, so the arrows can cycle and the list can highlight.
    bpy.types.Object.kinema_active_solution = IntProperty(
        name="Solution",
        description="Which of the configurations found is currently applied",
        default=0,
        min=0,
    )
    bpy.types.Object.kinema_elbow_strength = FloatProperty(
        name="Elbow Strength",
        description=(
            "How firmly the swivel holds the elbow at its angle, against the "
            "tool's own weight of 50. On an arm with a spherical shoulder and "
            "wrist the elbow's goal is exactly reachable, so this barely "
            "matters: the arm7 fixture keeps its tool within 0.0003 mm from 0.5 "
            "to 20. On an arm whose links are offset the goal is only close -- "
            "watch the solve readout for tool error if you raise it"
        ),
        default=2.0,
        min=0.0,
        soft_max=10.0,
        update=_on_elbow_strength_changed,
    )
    # On the pose bone rather than the rig, so it sits beside the joint's own
    # slider and keys per joint: a rail can be handed back to IK mid-shot.
    bpy.types.PoseBone.kinema_ik_hold = BoolProperty(
        name="Hold",
        description=(
            "Keep this joint where you put it while IK solves the rest -- a rail, "
            "gantry axis or turntable you position by hand. Drag the joint's own "
            "bone and the arm follows with the tool held"
        ),
        default=False,
        update=_on_ik_hold_changed,
    )
    bpy.types.Object.kinema_tcp_parent = StringProperty(
        name="Parent Bone",
        description="Joint bone the tool centre point rides",
        default="",
    )
    # Expressed in the flange's *link* frame, so the numbers match a URDF
    # <origin rpy="..."> for the same tool. Zero is the flange itself, which is
    # not usually where the importer leaves the marker -- it uses the deepest
    # link, past any fixed joints, and seeds that distance here.
    bpy.types.Object.kinema_tcp_offset = FloatVectorProperty(
        name="Tool Offset",
        description="Tool position relative to the flange link frame",
        size=3,
        default=(0.0, 0.0, 0.0),
        subtype="TRANSLATION",
        unit="LENGTH",
    )
    bpy.types.Object.kinema_tcp_rpy = FloatVectorProperty(
        name="Tool Rotation",
        description=(
            "Tool orientation relative to the flange link frame, as roll, pitch "
            "and yaw about fixed X, Y and Z -- the convention URDF uses"
        ),
        size=3,
        default=(0.0, 0.0, 0.0),
        subtype="EULER",
        unit="ROTATION",
    )


def unregister_props() -> None:
    del bpy.types.Scene.kinema
    del bpy.types.Object.kinema_ik_enabled
    del bpy.types.Object.kinema_solver_mode
    del bpy.types.Object.kinema_ik_tip
    del bpy.types.Object.kinema_active_bone_index
    del bpy.types.Object.kinema_meshes_at_file_origin
    del bpy.types.Object.kinema_active_solution
    del bpy.types.Object.kinema_elbow_strength
    del bpy.types.PoseBone.kinema_ik_hold
    del bpy.types.Object.kinema_tcp_parent
    del bpy.types.Object.kinema_tcp_offset
    del bpy.types.Object.kinema_tcp_rpy
