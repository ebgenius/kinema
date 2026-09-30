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
Timeline at the bottom becomes one, grown to a third of the window, or the 3D
view is copied into a window of its own, as the preferences say.

**Joints only.** The joint bones are selected, and the rig's other bones
deselected, with *Only Show Selected* on. That's the Graph Editor's own filter,
so selecting another bone shows its curves as it always would. The rig's
live-IK switch is a channel of the object's own and stays listed. It shows
where a job hands the joints to IK.

Every setting comes from the add-on preferences. The Graph Editor's sidebar
has a Kinema tab that puts them back, or returns the editor to Blender's own
defaults.
"""

from __future__ import annotations

from types import SimpleNamespace

import bpy
from bpy.props import EnumProperty
from bpy.types import Operator, Panel

from ..rig import builder
from ..ui.panel import active_rig

#: How much of the window's height the Timeline grows to when it becomes the
#: Graph Editor: at its own 68 pixels, a curve is a line.
GROW_TO = 1.0 / 3.0

#: What each preference is set to until someone changes it. prefs.py declares
#: the properties with these defaults, and a missing preferences object (an
#: add-on being registered) falls back to them.
DEFAULTS = SimpleNamespace(
    graph_open_in="TIMELINE",
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


def _timeline(window):
    return next((area for area in window.screen.areas if area.ui_type == "TIMELINE"), None)


def _grow(context, window, area) -> None:
    """Move the area's top edge up until it's a third of the window's height.

    The new height shows once Blender redraws the screen, not straight away.
    """
    target = int(window.height * GROW_TO)
    edge = area.y + area.height + 1
    if area.height >= target or edge >= window.height:
        return
    with context.temp_override(window=window, area=area):
        bpy.ops.screen.area_move(
            x=area.x + area.width // 2, y=edge, delta=target - area.height
        )


def _new_window(context):
    """The area the button was pressed in, copied into a window of its own."""
    before = set(context.window_manager.windows)
    source = context.area or max(context.window.screen.areas, key=lambda a: a.width * a.height)
    with context.temp_override(window=context.window, area=source):
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
            bpy.ops.anim.channels_expand(all=True)

    for region_type, operator in (("CHANNELS", expand), ("WINDOW", bpy.ops.graph.view_all)):
        region = next((r for r in area.regions if r.type == region_type), None)
        if region is not None:
            with bpy.context.temp_override(window=window, area=area, region=region):
                operator()


def _when_drawn(window) -> None:
    """:func:`_open_and_frame` once Blender has drawn the editor at its new size.

    A grown Timeline, or a new window, takes its size at the next redraw, and
    framing before then fits the curves to the old one.
    """
    def later():
        if window in bpy.context.window_manager.windows[:]:
            _open_and_frame(window)
        return None

    bpy.app.timers.register(later, first_interval=0.05)


def set_up(rig, window, area, chosen) -> int | None:
    """The Kinema view in ``area``, for ``rig``. Returns how many joints it shows."""
    shown = None
    if chosen.graph_joints_only:
        # Only Show Selected lists the channels of selected objects only.
        rig.select_set(True)
        shown = select_joint_bones(rig)
    apply_kinema_view(area.spaces.active, window.screen, chosen)
    _when_drawn(window)
    return shown


class KINEMA_OT_graph_editor(Operator):
    bl_idname = "kinema.graph_editor"
    bl_label = "Graph Editor"
    bl_description = (
        "Show this robot's joint curves in a Graph Editor, set up as Kinema's "
        "preferences say: normalised, with sliders, the sidebar following playback"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return context.window is not None and active_rig(context) is not None

    def execute(self, context: bpy.types.Context) -> set[str]:
        rig = active_rig(context)
        chosen = settings(context)
        window = context.window
        area = _graph_editor(window)
        where = "the Graph Editor"
        if area is None and chosen.graph_open_in == "TIMELINE":
            area = _timeline(window)
            if area is not None:
                area.ui_type = "FCURVES"
                _grow(context, window, area)
                where = "the Timeline, now the Graph Editor"
        if area is None:
            window, area = _new_window(context)
            where = "a new window"
            if area is None:
                self.report({"ERROR"}, "Could not open a window for the Graph Editor")
                return {"CANCELLED"}

        shown = set_up(rig, window, area, chosen)
        what = f"{shown} joint curves of '{rig.name}'" if shown is not None else f"'{rig.name}'"
        self.report({"INFO"}, f"{what} in {where}")
        return {"FINISHED"}


class KINEMA_OT_graph_view(Operator):
    bl_idname = "kinema.graph_view"
    bl_label = "Graph Editor View"
    bl_description = (
        "Set this Graph Editor up the way Kinema's preferences say, or as Blender "
        "makes a new one"
    )
    bl_options = {"REGISTER", "UNDO"}

    to: EnumProperty(
        name="To",
        items=[
            ("KINEMA", "Kinema View", "The joint curves, set up as Kinema's preferences say"),
            ("BLENDER", "Blender Defaults", "As Blender's startup file makes a new Graph Editor"),
        ],
        default="KINEMA",
        options={"SKIP_SAVE"},
    )

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return context.area is not None and context.area.ui_type == "FCURVES"

    def execute(self, context: bpy.types.Context) -> set[str]:
        if self.to == "BLENDER":
            apply_blender_defaults(context.space_data, context.screen)
            self.report({"INFO"}, "Graph Editor back to Blender's defaults")
            return {"FINISHED"}
        rig = active_rig(context)
        if rig is None:
            self.report({"ERROR"}, "Select a Kinema robot first")
            return {"CANCELLED"}
        shown = set_up(rig, context.window, context.area, settings(context))
        what = f"{shown} joint curves of '{rig.name}'" if shown is not None else f"'{rig.name}'"
        self.report({"INFO"}, f"Graph Editor showing {what}")
        return {"FINISHED"}


class KINEMA_PT_graph_editor(Panel):
    """In the Graph Editor's own sidebar, for when the view has been changed too far."""

    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_category = "Kinema"
    bl_label = "Kinema"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return context.area is not None and context.area.ui_type == "FCURVES"

    def draw(self, context: bpy.types.Context) -> None:
        column = self.layout.column(align=True)
        for to, text, icon in (
            ("KINEMA", "Kinema View", "GRAPH"),
            ("BLENDER", "Blender Defaults", "LOOP_BACK"),
        ):
            column.operator("kinema.graph_view", text=text, icon=icon).to = to
        self.layout.label(text="Set in Kinema's preferences", icon="PREFERENCES")


classes = (KINEMA_OT_graph_editor, KINEMA_OT_graph_view, KINEMA_PT_graph_editor)
