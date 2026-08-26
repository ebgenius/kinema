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
from bpy.props import EnumProperty, PointerProperty, StringProperty
from bpy.types import Panel, PropertyGroup

from .. import runtime
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
    """Scene-level state. Per-rig state lives on the armature object itself."""

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


classes = (KinemaSceneProps, KINEMA_PT_main, KINEMA_PT_joints, KINEMA_PT_tcp, KINEMA_PT_status)


def register_props() -> None:
    bpy.types.Scene.kinema = PointerProperty(type=KinemaSceneProps)


def unregister_props() -> None:
    del bpy.types.Scene.kinema
