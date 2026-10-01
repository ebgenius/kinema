"""One click to a robot's joint curves: a Graph Editor set up for them (#53).

The Graph Editor is where a joint's motion gets shaped: eased, retimed,
smoothed. Getting to the right view takes a string of settings someone new to
it won't know:
- the editor itself;
- only this robot's joints;
- the curves normalised, so a rail's metres and a wrist's radians share one
  scale;
- sliders beside the channels;
- the sidebar redrawn during playback.
**Graph Editor**, in Joints (FK), does all of it.

**Where.** A Graph Editor already in the window is reused. Otherwise the
bottom third of the 3D viewport is split off as one, and the Timeline stays.
Or, as the preferences say, the 3D view is copied into a window of its own.

**Joints only.** *Only Show Selected* lists the curves of every selected
object, so the rig is made the one selected, and the active one. Its joint
bones are selected, and its other bones deselected. That's the Graph Editor's
own filter, so selecting another bone shows its curves as it always would. The
rig's live-IK switch is a channel of the object's own and stays listed. It
shows where a job hands the joints to IK.

**Undo.** Graph Editor and Kinema View change the selection, so each has an
undo step: Ctrl+Z straight after puts the selection back. Without the step,
that Ctrl+Z would take the edit before it too. The layout and the editor's
settings stay as they are, since Blender never undoes those. Blender Defaults
changes only those, so it has no undo step: one would hold nothing.

Every setting comes from the add-on preferences. The Graph Editor's sidebar
has a Kinema tab that puts them back, or returns the editor to Blender's own
defaults.
"""

from __future__ import annotations

from types import SimpleNamespace

import bpy
from bpy.types import Operator, Panel

from ..rig import builder
from ..ui.panel import active_rig

#: How much of the 3D viewport's height is split off for the Graph Editor.
SHARE = 1.0 / 3.0

#: What each preference is set to until someone changes it. prefs.py declares
#: the properties with these defaults, and a missing preferences object (an
#: add-on being registered) falls back to them.
DEFAULTS = SimpleNamespace(
    graph_open_in="VIEWPORT",
    graph_joints_only=True,
    graph_normalize=True,
    graph_sliders=True,
    graph_playback_sidebar=True,
)


def settings(context=None):
    """The preferences' Graph Editor settings, or the defaults if there are none."""
    from ..prefs import get_prefs

    preferences = get_prefs(context)
    if preferences is None:
        return DEFAULTS
    return SimpleNamespace(**{
        name: getattr(preferences, name, value) for name, value in vars(DEFAULTS).items()
    })


def select_joint_bones(rig) -> int:
    """Select the rig's joint bones, and none of its other bones. Returns how many."""
    joints = {pose_bone.name for pose_bone in builder.joint_bones(rig)}
    for pose_bone in rig.pose.bones:
        pose_bone.select = pose_bone.name in joints
    return len(joints)


def show_only_joints(context, rig) -> int:
    """Leave the rig the one object selected, and active, with only its joints selected.

    Another object in a mode of its own, edit or paint say, keeps its selection,
    and stays active if it is, so that leaving the mode works as before.
    Returns how many joints there are.
    """
    for obj in list(context.view_layer.objects.selected):
        if obj != rig and obj.mode == "OBJECT":
            obj.select_set(False)
    rig.select_set(True)
    active = context.view_layer.objects.active
    if active is None or active.mode == "OBJECT":
        context.view_layer.objects.active = rig
    return select_joint_bones(rig)


def apply_kinema_view(space, screen, chosen) -> None:
    """Set a Graph Editor, and the screen it's in, as ``chosen`` says."""
    space.use_normalization = chosen.graph_normalize
    space.show_sliders = chosen.graph_sliders
    if chosen.graph_joints_only:
        space.dopesheet.show_only_selected = True
    screen.use_play_properties_editors = chosen.graph_playback_sidebar


def apply_blender_defaults(space, screen) -> None:
    """Put back what Blender's startup file gives a new Graph Editor and screen."""
    space.use_normalization = False
    space.show_sliders = False
    space.dopesheet.show_only_selected = True
    screen.use_play_properties_editors = False


def _graph_editor(window):
    return next((area for area in window.screen.areas if area.ui_type == "FCURVES"), None)


def _split_viewport(context, window):
    """The bottom third of the 3D viewport, split off as the Graph Editor. None if refused.

    A split rather than the Timeline grown. Growing an area means moving its
    edge, and Blender allows that only with the pointer on the edge itself
    (``area_move``'s poll wants no active region), never on the button just
    clicked. Splitting has no such rule, and the Timeline keeps its playback
    controls.
    """
    view = context.area if context.area is not None and context.area.type == "VIEW_3D" else None
    if view is None:
        views = [area for area in window.screen.areas if area.type == "VIEW_3D"]
        if not views:
            return None
        view = max(views, key=lambda area: area.width * area.height)
    before = set(window.screen.areas)
    with context.temp_override(window=window, area=view):
        if not bpy.ops.screen.area_split.poll():
            return None
        # Horizontal at a factor under a half: the new area is the lower part.
        bpy.ops.screen.area_split(direction="HORIZONTAL", factor=SHARE)
    area = next((area for area in window.screen.areas if area not in before), None)
    if area is not None:
        area.ui_type = "FCURVES"
    return area


def _new_window(context):
    """The area the button was pressed in, copied into a window of its own."""
    before = set(context.window_manager.windows)
    source = context.area or max(context.window.screen.areas, key=lambda a: a.width * a.height)
    with context.temp_override(window=context.window, area=source):
        if not bpy.ops.screen.area_dupli.poll():
            return None, None
        bpy.ops.screen.area_dupli("INVOKE_DEFAULT")
    window = next((w for w in context.window_manager.windows if w not in before), None)
    if window is None:
        return None, None
    area = window.screen.areas[0]
    area.ui_type = "FCURVES"
    return window, area


def _open_and_frame(window) -> None:
    """Expand the channels, so each joint's shows with its slider, and frame the curves."""
    area = _graph_editor(window)
    if area is None:
        return

    def expand():
        # One level a call: only what is listed when it runs opens, so the
        # object opens to its action, the action to the joints' groups, and
        # the groups to the curves and their sliders.
        for _ in range(3):
            if bpy.ops.anim.channels_expand.poll():
                bpy.ops.anim.channels_expand(all=True)

    def frame():
        # Refused with nothing keyed to frame, which is no error here.
        if bpy.ops.graph.view_all.poll():
            bpy.ops.graph.view_all()

    for region_type, operator in (("CHANNELS", expand), ("WINDOW", frame)):
        region = next((r for r in area.regions if r.type == region_type), None)
        if region is not None:
            with bpy.context.temp_override(window=window, area=area, region=region):
                operator()


def _when_drawn(window) -> None:
    """:func:`_open_and_frame` once Blender has drawn the editor at its new size.

    A split-off area, or a new window, takes its size at the next redraw, and
    framing before then fits the curves to the old one.
    """
    def later():
        if window in bpy.context.window_manager.windows[:]:
            _open_and_frame(window)
        return None

    bpy.app.timers.register(later, first_interval=0.05)


def set_up(context, rig, window, area, chosen, pressed_in) -> int | None:
    """The Kinema view in ``area``, in ``window``, for ``rig``. Returns how many joints it shows.

    ``pressed_in`` is the window the button was pressed in, which is ``window``
    unless the editor opened in one of its own.
    """
    shown = show_only_joints(context, rig) if chosen.graph_joints_only else None
    apply_kinema_view(area.spaces.active, window.screen, chosen)
    # During playback Blender redraws every window as the screen it was started
    # in says, and that may be either.
    pressed_in.screen.use_play_properties_editors = chosen.graph_playback_sidebar
    _when_drawn(window)
    return shown


class KINEMA_OT_graph_editor(Operator):
    bl_idname = "kinema.graph_editor"
    bl_label = "Graph Editor"
    bl_description = (
        "Show this robot's joint curves in a Graph Editor, set up as Kinema's "
        "preferences say: normalised, with sliders, the sidebar following playback"
    )
    # UNDO for the selection it changes. See "Undo" above.
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return context.window is not None and active_rig(context) is not None

    def execute(self, context: bpy.types.Context) -> set[str]:
        rig = active_rig(context)
        chosen = settings(context)
        # Kept, as opening a window makes it the context's.
        pressed_in = window = context.window
        area = _graph_editor(window)
        where = "the Graph Editor"
        if area is None and chosen.graph_open_in == "VIEWPORT":
            area = _split_viewport(context, window)
            where = "the Graph Editor below the 3D Viewport"
        # Also when the split is refused.
        if area is None:
            window, area = _new_window(context)
            where = "a new window"
            if area is None:
                self.report({"ERROR"}, "Could not open a window for the Graph Editor")
                return {"CANCELLED"}

        shown = set_up(context, rig, window, area, chosen, pressed_in)
        what = f"{shown} joint curves of '{rig.name}'" if shown is not None else f"'{rig.name}'"
        self.report({"INFO"}, f"{what} in {where}")
        return {"FINISHED"}


def _in_graph_editor(context) -> bool:
    return context.area is not None and context.area.ui_type == "FCURVES"


class KINEMA_OT_graph_view(Operator):
    bl_idname = "kinema.graph_view"
    bl_label = "Kinema View"
    bl_description = (
        "Set this Graph Editor up for the robot's joints again, as Kinema's preferences say"
    )
    # UNDO for the selection it changes. See "Undo" above.
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return _in_graph_editor(context)

    def execute(self, context: bpy.types.Context) -> set[str]:
        rig = active_rig(context)
        if rig is None:
            self.report({"ERROR"}, "Select a Kinema robot first")
            return {"CANCELLED"}
        window = context.window
        shown = set_up(context, rig, window, context.area, settings(context), window)
        what = f"{shown} joint curves of '{rig.name}'" if shown is not None else f"'{rig.name}'"
        self.report({"INFO"}, f"Graph Editor showing {what}")
        return {"FINISHED"}


class KINEMA_OT_graph_defaults(Operator):
    bl_idname = "kinema.graph_defaults"
    bl_label = "Blender Defaults"
    bl_description = "Put this Graph Editor back the way Blender's startup file makes a new one"
    # No UNDO: it changes only what Blender never undoes. See "Undo" above.
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return _in_graph_editor(context)

    def execute(self, context: bpy.types.Context) -> set[str]:
        apply_blender_defaults(context.space_data, context.screen)
        self.report({"INFO"}, "Graph Editor back to Blender's defaults")
        return {"FINISHED"}


class KINEMA_PT_graph_editor(Panel):
    """In the Graph Editor's own sidebar, for when the view has been changed too far."""

    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_category = "Kinema"
    bl_label = "Kinema"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return _in_graph_editor(context)

    def draw(self, context: bpy.types.Context) -> None:
        column = self.layout.column(align=True)
        column.operator("kinema.graph_view", icon="GRAPH")
        column.operator("kinema.graph_defaults", icon="LOOP_BACK")
        self.layout.label(text="Set in Kinema's preferences", icon="PREFERENCES")


classes = (
    KINEMA_OT_graph_editor,
    KINEMA_OT_graph_view,
    KINEMA_OT_graph_defaults,
    KINEMA_PT_graph_editor,
)
