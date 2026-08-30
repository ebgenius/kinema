"""Kinema's 3D viewport sidebar (N-panel).

Layout intent: an animator should see controls, not machinery. The everyday
case -- pick a robot, pose the TCP, tweak a joint -- is always visible, and
diagnostics stay collapsed.

The joint sliders drive the pose bones' real transform channels rather than
proxy properties, so they are keyframable exactly like any other bone channel:
click the dot beside a slider and you get a keyframe on the actual rig.
"""

from __future__ import annotations

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Panel, PropertyGroup, UIList

from .. import runtime
from ..ops.import_robot import SETTING_NAMES, import_settings
from ..rig import builder

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
        column.operator("kinema.import_catalog", text="Import from Catalog…",
                        icon="OUTLINER_OB_ARMATURE")
        column.operator("kinema.import_urdf", text="Import URDF File…", icon="FILE_FOLDER")

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

        column = layout.column(align=True)
        for pose_bone in joints:
            bone = pose_bone.bone
            is_revolute = bone.get(builder.PROP_JOINT_TYPE, "revolute") != "prismatic"
            row = column.row(align=True)
            # index=1 is the Y channel: the one aligned to the joint axis.
            if is_revolute:
                row.prop(pose_bone, "rotation_euler", index=1, text=pose_bone.name)
            else:
                row.prop(pose_bone, "location", index=1, text=pose_bone.name)

            limited = builder.PROP_LOWER in bone
            icon = "CON_ROTLIMIT" if limited else "BLANK1"
            sub = row.row(align=True)
            sub.enabled = limited
            sub.label(text="", icon=icon)


def _joint_indices(rig: bpy.types.Object) -> dict[str, int]:
    """Bone name -> its index into ``builder.joint_bones``.

    That index, not the name, is what ``kinema_ik_tip`` stores: an integer is
    keyframable and a bone reference is not.
    """
    return {bone.name: index for index, bone in enumerate(builder.joint_bones(rig))}


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
        bone = item.bone
        is_joint = builder.PROP_JOINT_NAME in bone
        attachment = builder.bone_attachment(rig, item.name)

        split = layout.split(factor=0.42, align=True)

        left = split.row(align=True)
        target = left.row(align=True)
        target.enabled = is_joint
        if is_joint:
            joint_index = _joint_indices(rig).get(item.name, -1)
            selected = joint_index == getattr(rig, "kinema_ik_tip", -1)
            operator = target.operator(
                "kinema.set_ik_tip",
                text="",
                icon="RADIOBUT_ON" if selected else "RADIOBUT_OFF",
                emboss=False,
            )
            operator.index = joint_index
        else:
            # The TCP marker and Root can carry attachments but cannot be
            # solved to, so their radio is a spacer rather than a dead button.
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
            body.prop(attachment, "rotation_euler", text="Rotation")
            body.prop(attachment, "scale")
            body.operator(
                "kinema.reset_attachment_offset", text="Reset", icon="LOOP_BACK"
            ).bone = pose_bone.name


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

        if not tcp_name or tcp_name not in rig.pose.bones:
            layout.label(text="No TCP on this rig", icon="INFO")
            layout.operator("kinema.set_tcp", text="Create TCP from Active Bone",
                            icon="EMPTY_ARROWS")
            return

        tcp = rig.pose.bones[tcp_name]
        world = rig.matrix_world @ tcp.matrix

        box = layout.box()
        column = box.column(align=True)
        column.label(text=f"Link: {rig.get('kinema_tcp_link', '—')}", icon="EMPTY_AXIS")
        location = world.translation
        column.label(text=f"X {location.x:+.4f}   Y {location.y:+.4f}   Z {location.z:+.4f}")

        layout.operator("kinema.set_tcp", text="Move TCP to Active Bone", icon="EMPTY_ARROWS")


class KINEMA_PT_ik(KinemaPanelBase, Panel):
    bl_idname = "KINEMA_PT_ik"
    bl_parent_id = "KINEMA_PT_main"
    bl_label = "Inverse Kinematics"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return active_rig(context) is not None

    def draw(self, context: bpy.types.Context) -> None:
        from .. import handlers

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
        # list. This row exists so the channel is reachable for keyframing:
        # the decorator dot beside it is how a shot hands the goal from the
        # wrist to the elbow part-way through.
        from ..solver import manager

        tip = layout.column(align=True)
        tip.use_property_split = True
        tip.use_property_decorate = True
        tip.prop(rig, "kinema_ik_tip", text="Target Bone")
        tip.label(text=f"Solving to '{manager.tip_bone(rig)}'", icon="BONE_DATA")

        row = layout.row(align=True)
        row.operator("kinema.snap_ik", text="Snap to Tool", icon="SNAP_ON")
        row.operator("kinema.remove_ik", text="", icon="X")

        layout.operator("kinema.bake_ik", text="Bake to Keyframes", icon="RENDER_ANIMATION")

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


def manager_state(rig):
    from ..solver import manager

    return manager._cache.get(rig.name)


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
    KINEMA_PT_main,
    KINEMA_PT_joints,
    KINEMA_PT_bones,
    KINEMA_PT_tcp,
    KINEMA_PT_ik,
    KINEMA_PT_status,
)


def _on_ik_tip_changed(rig, context) -> None:
    """Forget the cached goal so the next update re-solves the new chain."""
    from .. import handlers

    handlers.reset(rig.name)


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


def unregister_props() -> None:
    del bpy.types.Scene.kinema
    del bpy.types.Object.kinema_ik_enabled
    del bpy.types.Object.kinema_solver_mode
    del bpy.types.Object.kinema_ik_tip
    del bpy.types.Object.kinema_active_bone_index
