"""Kinema's 3D viewport sidebar (N-panel).

Layout intent: an animator should see controls, not machinery. The panel is
split so that the everyday case -- pick a robot, pose the TCP, tweak joints --
is always visible, while diagnostics stay collapsed.
"""

from __future__ import annotations

import bpy
from bpy.props import PointerProperty, StringProperty
from bpy.types import Panel, PropertyGroup

from .. import runtime

CATEGORY = "Kinema"


class KinemaSceneProps(PropertyGroup):
    """Scene-level state. Per-rig state lives on the armature object itself."""

    last_import: StringProperty(
        name="Last Import",
        description="Most recently imported robot description",
        default="",
    )


class KinemaPanelBase:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = CATEGORY


class KINEMA_PT_main(KinemaPanelBase, Panel):
    bl_idname = "KINEMA_PT_main"
    bl_label = "Kinema"

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        layout.label(text="Import a robot to begin", icon="ARMATURE_DATA")
        # Import operators land here in M3/M4.
        row = layout.row()
        row.enabled = False
        row.operator("kinema.check_dependencies", text="Import Robot…", icon="IMPORT")


class KINEMA_PT_status(KinemaPanelBase, Panel):
    bl_idname = "KINEMA_PT_status"
    bl_parent_id = "KINEMA_PT_main"
    bl_label = "Solver Status"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        column = layout.column(align=True)

        if runtime.solver_available():
            column.label(text="PyRoki ready", icon="CHECKMARK")
        elif runtime.solver_error():
            column.label(text="PyRoki unavailable", icon="ERROR")
            # The full traceback would overflow the panel; the preferences
            # dependency table names the specific missing module.
            column.label(text=runtime.solver_error()[:48], icon="BLANK1")
            column.label(text="Using NumPy fallback", icon="BLANK1")
        else:
            column.label(text="Solver loading…", icon="SORTTIME")
            column.label(text="Using NumPy fallback", icon="BLANK1")

        layout.operator("kinema.check_dependencies", text="Re-check", icon="FILE_REFRESH")


classes = (KinemaSceneProps, KINEMA_PT_main, KINEMA_PT_status)


def register_props() -> None:
    bpy.types.Scene.kinema = PointerProperty(type=KinemaSceneProps)


def unregister_props() -> None:
    del bpy.types.Scene.kinema
