"""Add-on preferences: dependency diagnostics and solver defaults."""

from __future__ import annotations

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    IntProperty,
    StringProperty,
)
from bpy.types import AddonPreferences, PropertyGroup, UIList

from . import runtime


class KinemaSearchPath(PropertyGroup):
    """One directory to look for ROS packages in."""

    path: StringProperty(
        name="Path",
        description="A directory holding ROS packages",
        subtype="DIR_PATH",
        default="",
    )


class KINEMA_UL_search_paths(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_prop, index):
        row = layout.row(align=True)
        if _is_unusable(item.path):
            # Skipped at import time rather than scanned. Saying so here is the
            # difference between "Kinema ignores this" and "Kinema is broken":
            # a whole drive cannot be indexed, and silently doing nothing would
            # look like the setting had no effect.
            row.label(text="", icon="ERROR")
            row.prop(item, "path", text="")
            row.label(text="a whole drive cannot be searched")
        else:
            row.prop(item, "path", text="", emboss=False, icon="FILE_FOLDER")


def _is_unusable(raw: str) -> bool:
    """True for a path that would be skipped: a filesystem root."""
    from pathlib import Path

    text = (raw or "").strip()
    if not text:
        return False
    try:
        resolved = Path(bpy.path.abspath(text)).resolve()
    except (OSError, ValueError):
        return False
    return bool(resolved.anchor) and resolved == Path(resolved.anchor)


class KINEMA_OT_add_search_path(bpy.types.Operator):
    bl_idname = "kinema.add_search_path"
    bl_label = "Add Package Search Path"
    bl_description = "Add a directory to search for ROS packages"
    bl_options = {"INTERNAL"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        preferences = get_prefs(context)
        preferences.search_paths.add()
        preferences.active_search_path = len(preferences.search_paths) - 1
        return {"FINISHED"}


class KINEMA_OT_remove_search_path(bpy.types.Operator):
    bl_idname = "kinema.remove_search_path"
    bl_label = "Remove Package Search Path"
    bl_description = "Remove the selected search path"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        preferences = get_prefs(context)
        return bool(preferences and preferences.search_paths)

    def execute(self, context: bpy.types.Context) -> set[str]:
        preferences = get_prefs(context)
        index = preferences.active_search_path
        if 0 <= index < len(preferences.search_paths):
            preferences.search_paths.remove(index)
            preferences.active_search_path = max(0, index - 1)
        return {"FINISHED"}


def package_search_paths() -> list[str]:
    """Extra directories to find ROS packages in, from the preferences.

    A robot's own checkout is found automatically; this is for the case that
    checkout cannot cover -- a cell whose macros live in one repository and
    whose robots live in others, where ``$(find shared_macros)`` has to reach
    across.

    Returns an empty list rather than raising when preferences are not readable,
    which happens while the add-on is being registered or unregistered.
    """
    preferences = get_prefs()
    if preferences is None:
        return []
    return [
        bpy.path.abspath(entry.path)
        for entry in preferences.search_paths
        if entry.path.strip()
    ]


class KinemaPreferences(AddonPreferences):
    # For an extension this must be the add-on's module name, which Blender
    # gives as "bl_ext.<repo>.kinema" -- __package__ resolves to exactly that.
    bl_idname = __package__

    solver_backend: EnumProperty(
        name="Default Solver",
        description="Which IK backend new robot rigs start with",
        items=[
            ("PYROKI", "PyRoki", "Singularity- and limit-aware solver (recommended)"),
            ("NUMPY", "NumPy", "Lightweight damped least squares; no JAX required"),
            ("OFF", "Off", "No IK; the rig behaves as a plain FK armature"),
        ],
        default="PYROKI",
    )
    solve_timeout_ms: IntProperty(
        name="Solve Budget (ms)",
        description=(
            "Skip a live IK update if the previous solve took longer than this, to "
            "keep the viewport responsive on very high-DoF rigs"
        ),
        default=33, min=4, max=1000,
    )
    debug_logging: BoolProperty(
        name="Debug Logging",
        description="Print solver diagnostics to the system console",
        default=False,
    )
    search_paths: CollectionProperty(type=KinemaSearchPath)
    active_search_path: IntProperty(default=0)

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout

        column = layout.column(align=True)
        column.prop(self, "solver_backend")
        column.prop(self, "solve_timeout_ms")
        column.prop(self, "debug_logging")

        layout.separator()
        header = layout.row()
        header.label(text="Dependencies", icon="PACKAGE")
        header.operator("kinema.check_dependencies", text="Re-check", icon="FILE_REFRESH")

        box = layout.box()
        report = runtime.dependency_report()
        for name, status, detail in report:
            row = box.row()
            row.label(text=name, icon="CHECKMARK" if status == "ok" else "ERROR")
            sub = row.row()
            sub.alignment = "RIGHT"
            sub.label(text=detail)

        if any(status != "ok" for _, status, _ in report):
            warn = layout.box()
            warn.label(text="Some dependencies are unavailable.", icon="INFO")
            warn.label(text="Kinema falls back to the NumPy solver; IK quality is reduced.")

        layout.separator()
        layout.label(text="Package Search Paths", icon="FILE_FOLDER")
        row = layout.row()
        row.template_list(
            "KINEMA_UL_search_paths", "", self, "search_paths",
            self, "active_search_path", rows=3,
        )
        column = row.column(align=True)
        column.operator("kinema.add_search_path", text="", icon="ADD")
        column.operator("kinema.remove_search_path", text="", icon="REMOVE")

        note = layout.column(align=True)
        note.label(
            text="Searched in addition to the repository holding the file.",
            icon="INFO",
        )
        note.label(
            text="For a cell whose macros live in a different repository from its robots.",
            icon="BLANK1",
        )


class KINEMA_OT_check_dependencies(bpy.types.Operator):
    bl_idname = "kinema.check_dependencies"
    bl_label = "Check Kinema Dependencies"
    bl_description = "Re-import the solver stack and refresh the status list"
    bl_options = {"REGISTER"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        runtime.unload_solver_stack()
        stack = runtime.load_solver_stack(debug=get_prefs(context).debug_logging)
        if stack:
            self.report({"INFO"}, "Kinema: solver stack loaded")
        else:
            self.report({"WARNING"}, f"Kinema: solver unavailable ({runtime.solver_error()})")
        return {"FINISHED"}


def get_prefs(context: bpy.types.Context | None = None) -> KinemaPreferences:
    """Fetch this add-on's preferences, or None if it is being unregistered."""
    context = context or bpy.context
    addon = context.preferences.addons.get(__package__)
    return addon.preferences if addon else None


# KinemaSearchPath first: KinemaPreferences declares a CollectionProperty of it,
# and Blender resolves the type at registration.
classes = (
    KinemaSearchPath,
    KINEMA_UL_search_paths,
    KINEMA_OT_add_search_path,
    KINEMA_OT_remove_search_path,
    KinemaPreferences,
    KINEMA_OT_check_dependencies,
)
