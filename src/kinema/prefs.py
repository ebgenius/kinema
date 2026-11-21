"""Add-on preferences: solver defaults, dependency diagnostics and the Graph Editor set-up."""

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
from .ui import quick_bar


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
    job_as_strips: BoolProperty(
        name="Job as NLA Strips",
        description=(
            "Lay a generated job out on the robot's NLA, one strip per motion, so "
            "motions can be retimed and reused there. Off, the job's keys stay in "
            "the robot's active action"
        ),
        default=True,
    )
    keep_compiled: BoolProperty(
        name="Keep Compiled Solvers",
        description=(
            "Save each compiled solver in Kinema's user folder, so a later session "
            "loads it in seconds instead of compiling it again. Takes effect the "
            "next time Blender starts"
        ),
        default=True,
    )
    search_paths: CollectionProperty(type=KinemaSearchPath)
    active_search_path: IntProperty(default=0)

    # Graph Editor, in Joints (FK): see ops/graph_editor.py, whose DEFAULTS
    # repeat these defaults for when there are no preferences to read.
    graph_open_in: EnumProperty(
        name="Open In",
        description="Where Graph Editor opens when the window has none",
        items=[
            (
                "VIEWPORT", "Below the 3D Viewport",
                "Split the bottom third off the 3D viewport as the Graph Editor. The "
                "Timeline stays",
            ),
            ("WINDOW", "New Window", "Open the Graph Editor in a window of its own"),
        ],
        default="VIEWPORT",
    )
    graph_joints_only: BoolProperty(
        name="Joints Only",
        description=(
            "Select the robot's joint bones and none of its other bones, with Only "
            "Show Selected on, so the Graph Editor lists the joint curves"
        ),
        default=True,
    )
    graph_normalize: BoolProperty(
        name="Normalize",
        description=(
            "Scale each curve to -1..1, so a rail's metres and a wrist's radians "
            "share one scale"
        ),
        default=True,
    )
    graph_sliders: BoolProperty(
        name="Sliders",
        description="Show a slider beside each channel, for its value on the current frame",
        default=True,
    )
    graph_playback_sidebar: BoolProperty(
        name="Sidebar Follows Playback",
        description=(
            "Redraw the sidebar on every frame of playback, so the joint sliders and "
            "Motion Limits follow the animation. Blender leaves this off: Timeline, "
            "Playback, Play In, Properties and Sidebars"
        ),
        default=True,
    )

    #: The quick bar's buttons, in order, as keys of ui/quick_bar.BUTTONS joined
    #: by commas. Edited with the operators below, not typed.
    quick_buttons: StringProperty(
        name="Quick Bar",
        description="The buttons above the Kinema tab's panels, in order",
        default=",".join(quick_bar.DEFAULT),
        options={"HIDDEN"},
    )

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout

        column = layout.column(align=True)
        column.prop(self, "solver_backend")
        column.prop(self, "solve_timeout_ms")
        column.prop(self, "debug_logging")
        column.prop(self, "job_as_strips")
        row = column.row(align=True)
        row.prop(self, "keep_compiled")
        entries, size = runtime.compile_cache_usage(runtime.compile_cache_folder())
        row.operator(
            "kinema.clear_compile_cache",
            text=f"Clear ({entries} kept, {size / 1024**2:.0f} MB)",
            icon="TRASH",
        )

        layout.separator()
        layout.separator()
        self._draw_quick_bar(layout)

        layout.separator()
        header = layout.row()
        header.label(text="Graph Editor", icon="GRAPH")
        header.operator("kinema.reset_graph_preferences", text="Reset", icon="LOOP_BACK")
        column = layout.box().column()
        for name in GRAPH_SETTINGS:
            column.prop(self, name)

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


    def _draw_quick_bar(self, layout) -> None:
        """The quick bar's buttons, to put in order, take out and add to."""
        header = layout.row()
        header.label(text="Quick Bar", icon="TOOL_SETTINGS")
        header.operator("kinema.quick_button_reset", text="Reset", icon="LOOP_BACK")
        box = layout.box()
        keys = quick_button_keys(self)
        if not keys:
            box.label(text="No buttons: the quick bar shows only the robot's name.")
        for index, key in enumerate(keys):
            button = quick_bar.BUTTONS[key]
            row = box.row(align=True)
            row.label(text=button.label, icon=button.icon)
            if index == 0 and len(keys) % 2:
                row.label(text="Spans the bar")
            up = row.row(align=True)
            up.enabled = index > 0
            move = up.operator("kinema.quick_button_move", text="", icon="TRIA_UP")
            move.index, move.step = index, -1
            down = row.row(align=True)
            down.enabled = index < len(keys) - 1
            move = down.operator("kinema.quick_button_move", text="", icon="TRIA_DOWN")
            move.index, move.step = index, 1
            row.operator("kinema.quick_button_remove", text="", icon="X").index = index
        missing = [key for key in quick_bar.BUTTONS if key not in keys]
        if missing:
            box.operator_menu_enum(
                "kinema.quick_button_add", "button", text="Add Button", icon="ADD"
            )


def quick_button_keys(preferences=None) -> list[str]:
    """The quick bar's buttons, in order: those the preferences name that Kinema
    knows, or the defaults where there are no preferences to read."""
    preferences = preferences if preferences is not None else get_prefs()
    if preferences is None:
        return list(quick_bar.DEFAULT)
    keys = [key for key in preferences.quick_buttons.split(",") if key in quick_bar.BUTTONS]
    return list(dict.fromkeys(keys))


def _set_quick_buttons(context, keys) -> None:
    get_prefs(context).quick_buttons = ",".join(keys)
    context.preferences.is_dirty = True


class KINEMA_OT_quick_button_move(bpy.types.Operator):
    bl_idname = "kinema.quick_button_move"
    bl_label = "Move Quick Bar Button"
    bl_description = "Move this button up or down the quick bar"
    bl_options = {"INTERNAL"}

    index: IntProperty(options={"SKIP_SAVE"})
    step: IntProperty(default=1, options={"SKIP_SAVE"})

    def execute(self, context: bpy.types.Context) -> set[str]:
        keys = quick_button_keys(get_prefs(context))
        to = self.index + self.step
        if not (0 <= self.index < len(keys) and 0 <= to < len(keys)):
            return {"CANCELLED"}
        keys[self.index], keys[to] = keys[to], keys[self.index]
        _set_quick_buttons(context, keys)
        return {"FINISHED"}


class KINEMA_OT_quick_button_remove(bpy.types.Operator):
    bl_idname = "kinema.quick_button_remove"
    bl_label = "Remove Quick Bar Button"
    bl_description = "Take this button off the quick bar. Its panel keeps it"
    bl_options = {"INTERNAL"}

    index: IntProperty(options={"SKIP_SAVE"})

    def execute(self, context: bpy.types.Context) -> set[str]:
        keys = quick_button_keys(get_prefs(context))
        if not 0 <= self.index < len(keys):
            return {"CANCELLED"}
        del keys[self.index]
        _set_quick_buttons(context, keys)
        return {"FINISHED"}


#: The Add Button menu's items, kept referenced: Blender reads a dynamic enum's
#: strings after its callback has returned.
_MISSING_ITEMS: list = []


def _missing_buttons(_self, context) -> list:
    """The buttons not on the quick bar yet, for the Add Button menu."""
    keys = quick_button_keys(get_prefs(context))
    _MISSING_ITEMS[:] = [
        (key, button.label, button.description, button.icon, index)
        for index, (key, button) in enumerate(quick_bar.BUTTONS.items())
        if key not in keys
    ]
    return _MISSING_ITEMS


class KINEMA_OT_quick_button_add(bpy.types.Operator):
    bl_idname = "kinema.quick_button_add"
    bl_label = "Add Quick Bar Button"
    bl_description = "Put a button at the end of the quick bar"
    bl_options = {"INTERNAL"}

    button: EnumProperty(name="Button", items=_missing_buttons)

    def execute(self, context: bpy.types.Context) -> set[str]:
        keys = quick_button_keys(get_prefs(context))
        if self.button in keys or self.button not in quick_bar.BUTTONS:
            return {"CANCELLED"}
        _set_quick_buttons(context, [*keys, self.button])
        return {"FINISHED"}


class KINEMA_OT_quick_button_reset(bpy.types.Operator):
    bl_idname = "kinema.quick_button_reset"
    bl_label = "Reset Quick Bar"
    bl_description = "Put the quick bar's buttons back to Kinema's defaults"
    bl_options = {"INTERNAL"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        _set_quick_buttons(context, quick_bar.DEFAULT)
        return {"FINISHED"}


#: The Graph Editor preferences, in the order they are drawn, and the ones
#: Reset puts back.
GRAPH_SETTINGS = (
    "graph_open_in",
    "graph_joints_only",
    "graph_normalize",
    "graph_sliders",
    "graph_playback_sidebar",
)


class KINEMA_OT_reset_graph_preferences(bpy.types.Operator):
    bl_idname = "kinema.reset_graph_preferences"
    bl_label = "Reset Graph Editor Preferences"
    bl_description = "Put the Graph Editor preferences back to Kinema's defaults"
    bl_options = {"INTERNAL"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        preferences = get_prefs(context)
        if preferences is None:
            return {"CANCELLED"}
        for name in GRAPH_SETTINGS:
            preferences.property_unset(name)
        return {"FINISHED"}


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


class KINEMA_OT_clear_compile_cache(bpy.types.Operator):
    bl_idname = "kinema.clear_compile_cache"
    bl_label = "Clear Compiled Solvers"
    bl_description = (
        "Delete the compiled solvers kept on disk. Each is compiled again the next "
        "time it is needed"
    )
    bl_options = {"INTERNAL"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        # The folder whether or not the cache is ticked on: files kept from
        # before are still worth clearing. Not created just to be emptied.
        removed, left = runtime.clear_compile_cache(runtime.compile_cache_folder())
        message = f"Kinema: removed {removed} compiled solvers"
        if left:
            # Open in another Blender, or in JAX here: gone next time.
            self.report({"WARNING"}, f"{message}; {left} files in use were left")
        else:
            self.report({"INFO"}, message)
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
    KINEMA_OT_clear_compile_cache,
    KINEMA_OT_reset_graph_preferences,
    KINEMA_OT_quick_button_move,
    KINEMA_OT_quick_button_remove,
    KINEMA_OT_quick_button_add,
    KINEMA_OT_quick_button_reset,
)
