"""Add-on preferences: dependency diagnostics and solver defaults."""

from __future__ import annotations

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty
from bpy.types import AddonPreferences

from . import runtime


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


classes = (KinemaPreferences, KINEMA_OT_check_dependencies)
