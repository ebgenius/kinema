"""Graph Editor, in Joints (FK): the parts that need no window. Needs a real ``bpy``.

Opening the editor needs a window, which a background Blender has none of. The
claims checked here are the rest:
- the joint bones are what gets selected, whatever kind of joint, and nothing
  else of the rig;
- the rig is left the one object selected, and active, unless another object is
  in a mode of its own;
- the Kinema view sets what the preferences say, in the window pressed in as
  well as the editor's, and Blender Defaults puts back what a new Graph Editor
  has;
- the editor set up is the one framed, when a window holds two;
- only what changes the selection has an undo step;
- the preferences' defaults are the ones used when there are no preferences,
  and Reset puts them back.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from ..conftest import requires_bpy

pytestmark = requires_bpy


@pytest.fixture
def graph(addon):
    return importlib.import_module(f"{addon.__name__}.ops.graph_editor")


@pytest.fixture
def builder(addon):
    return importlib.import_module(f"{addon.__name__}.rig.builder")


@pytest.fixture
def preferences(addon, graph):
    prefs = importlib.import_module(f"{addon.__name__}.prefs")
    found = prefs.get_prefs()
    before = {name: getattr(found, name) for name in prefs.GRAPH_SETTINGS}
    yield found
    for name, value in before.items():
        setattr(found, name, value)


@pytest.fixture
def rail6(addon, builder, fixture_dir, clean_scene):
    """rail6: a prismatic rail under six revolute joints, with an IK target."""
    import bpy

    assert "FINISHED" in bpy.ops.kinema.build_robot(filepath=str(fixture_dir / "rail6.urdf"))
    rig = next(o for o in bpy.data.objects if builder.is_kinema_rig(o))
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    rig.kinema_solver_mode = "NUMPY"
    assert "FINISHED" in bpy.ops.kinema.add_ik()
    return rig


def _editor():
    """Enough of a Graph Editor and its screen to set up, with Blender's defaults."""
    space = SimpleNamespace(
        use_normalization=False,
        show_sliders=False,
        dopesheet=SimpleNamespace(show_only_selected=False),
    )
    return space, SimpleNamespace(use_play_properties_editors=False)


def _keyed_cube():
    """An object with curves of its own, which Only Show Selected lists when it's selected."""
    import bpy

    cube = bpy.data.objects.new("Cube", bpy.data.meshes.new("Cube"))
    bpy.context.scene.collection.objects.link(cube)
    cube.keyframe_insert("location", frame=1)
    return cube


def _selected():
    import bpy

    return {obj.name for obj in bpy.context.view_layer.objects.selected}


class TestJointBones:
    def test_the_joints_are_selected_and_nothing_else(self, rail6, builder, graph):
        """The rail is prismatic and still a joint; the IK target, TCP and Root are not."""
        ik_name = rail6.get(builder.PROP_IK_BONE)
        for pose_bone in rail6.pose.bones:
            pose_bone.select = pose_bone.name == ik_name

        assert graph.select_joint_bones(rail6) == 7
        selected = {pb.name for pb in rail6.pose.bones if pb.select}
        assert selected == {pb.name for pb in builder.joint_bones(rail6)}
        assert "rail" in selected
        assert ik_name not in selected

    def test_the_robot_is_left_the_one_object_selected(self, rail6, builder, graph):
        """Another selected object's curves would be listed beside the joints.

        Measured in a window: a keyed cube selected, with a link mesh active, put
        the cube's three curves in the list beside the seven joints'.
        """
        import bpy

        cube = _keyed_cube()
        link = next(obj for obj in rail6.children if obj.type == "MESH")
        for obj in bpy.context.view_layer.objects:
            obj.select_set(obj in (cube, link))
        bpy.context.view_layer.objects.active = link
        assert _selected() == {cube.name, link.name}, "precondition: the robot isn't selected"

        assert graph.show_only_joints(bpy.context, rail6) == 7
        assert _selected() == {rail6.name}
        assert bpy.context.view_layer.objects.active == rail6
        assert {pb.name for pb in rail6.pose.bones if pb.select} == {
            pb.name for pb in builder.joint_bones(rail6)
        }

    def test_an_object_in_a_mode_of_its_own_is_left_as_it_is(self, rail6, graph):
        """A link mesh being edited stays selected and active, so Tab still leaves edit mode."""
        import bpy

        link = next(obj for obj in rail6.children if obj.type == "MESH")
        for obj in bpy.context.view_layer.objects:
            obj.select_set(obj == link)
        bpy.context.view_layer.objects.active = link
        assert "FINISHED" in bpy.ops.object.mode_set(mode="EDIT")
        try:
            # Selected after, or it would be in edit mode with the link.
            cube = _keyed_cube()
            cube.select_set(True)
            assert (link.mode, cube.mode) == ("EDIT", "OBJECT"), "precondition"
            graph.show_only_joints(bpy.context, rail6)
            assert _selected() == {link.name, rail6.name}, "the cube deselected, the link not"
            assert bpy.context.view_layer.objects.active == link
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")


class TestSettings:
    def test_the_kinema_view_sets_what_the_preferences_say(self, graph):
        space, screen = _editor()
        graph.apply_kinema_view(space, screen, graph.DEFAULTS)
        assert (space.use_normalization, space.show_sliders) == (True, True)
        assert space.dopesheet.show_only_selected
        assert screen.use_play_properties_editors

    def test_each_setting_follows_its_preference(self, graph):
        chosen = SimpleNamespace(**vars(graph.DEFAULTS))
        chosen.graph_normalize = False
        chosen.graph_playback_sidebar = False
        chosen.graph_joints_only = False
        space, screen = _editor()
        graph.apply_kinema_view(space, screen, chosen)
        assert not space.use_normalization
        assert space.show_sliders
        assert not screen.use_play_properties_editors
        assert not space.dopesheet.show_only_selected, "left as it was"

    def test_blender_defaults_are_a_new_graph_editors(self, graph):
        """As Blender 5.2's startup file makes one, measured in a window."""
        space, screen = _editor()
        graph.apply_kinema_view(space, screen, graph.DEFAULTS)
        graph.apply_blender_defaults(space, screen)
        assert (space.use_normalization, space.show_sliders) == (False, False)
        assert space.dopesheet.show_only_selected
        assert not screen.use_play_properties_editors

    @pytest.mark.parametrize("follows", [True, False])
    def test_a_new_window_and_the_one_pressed_in_both_follow_playback(
        self, graph, monkeypatch, follows
    ):
        """With the editor in a window of its own, playback may be started in either window.

        Blender redraws every window during playback as the screen it was started in
        says. Measured in a window, with live IK off and only the new window's screen
        set: played from the main window, the sidebar there drew 0 times in 47 frames.
        And the window Blender opens becomes the context's, as faked here: measured,
        ``context.window`` was the new one by the time the editor was set up.
        """
        space, opened = _editor()
        pressed = SimpleNamespace(screen=SimpleNamespace(use_play_properties_editors=not follows))
        context = SimpleNamespace(window=pressed)

        def new_window(context):
            context.window = SimpleNamespace(screen=opened)
            return context.window, SimpleNamespace(spaces=SimpleNamespace(active=space))

        chosen = SimpleNamespace(**vars(graph.DEFAULTS))
        chosen.graph_open_in = "WINDOW"
        chosen.graph_joints_only = False
        chosen.graph_playback_sidebar = follows
        monkeypatch.setattr(graph, "active_rig", lambda context: SimpleNamespace(name="robot"))
        monkeypatch.setattr(graph, "settings", lambda context: chosen)
        monkeypatch.setattr(graph, "_graph_editor", lambda window: None)
        monkeypatch.setattr(graph, "_new_window", new_window)
        monkeypatch.setattr(graph, "_when_drawn", lambda window, area: None)
        reports = []
        operator = SimpleNamespace(report=lambda kind, text: reports.append(text))

        assert graph.KINEMA_OT_graph_editor.execute(operator, context) == {"FINISHED"}
        assert reports == ["'robot' in a new window"], "precondition: a window was opened"
        assert opened.use_play_properties_editors is follows
        assert pressed.screen.use_play_properties_editors is follows


class TestFraming:
    def test_the_editor_set_up_is_the_one_framed(self, graph, monkeypatch):
        """With two Graph Editors in a window, the one set up is framed, not the first.

        Measured in a window: Kinema View pressed in the second editor framed the
        first, and left the second's view as it was. Framing waits for a redraw,
        which a background Blender never does, so the area it's handed is checked.
        """
        scheduled = []
        monkeypatch.setattr(graph, "_when_drawn", lambda window, area: scheduled.append(area))
        first, second = (
            SimpleNamespace(ui_type="FCURVES", spaces=SimpleNamespace(active=_editor()[0]))
            for _ in range(2)
        )
        window = SimpleNamespace(
            screen=SimpleNamespace(areas=[first, second], use_play_properties_editors=False)
        )
        chosen = SimpleNamespace(**vars(graph.DEFAULTS))
        chosen.graph_joints_only = False
        assert graph._graph_editor(window) is first, "precondition: a lookup finds the first"

        graph.set_up(SimpleNamespace(), None, window, second, chosen, window)
        # By identity: the two fakes compare equal by value.
        assert len(scheduled) == 1 and scheduled[0] is second


class TestUndo:
    def test_only_what_changes_the_selection_has_an_undo_step(self, graph):
        """Measured in a window, with an edit made just before.

        - Graph Editor with its undo step: one Ctrl+Z put the selection back and
          kept the edit. Without the step, it took the edit too.
        - Blender Defaults with an undo step: one Ctrl+Z changed nothing, as the
          editor and screen it changes aren't undone.
        """
        assert "UNDO" in graph.KINEMA_OT_graph_editor.bl_options
        assert "UNDO" in graph.KINEMA_OT_graph_view.bl_options
        assert "UNDO" not in graph.KINEMA_OT_graph_defaults.bl_options


class TestPreferences:
    def test_the_defaults_are_the_fallback(self, addon, graph):
        """What the preferences declare is what's used when there are none."""
        prefs = importlib.import_module(f"{addon.__name__}.prefs")
        declared = prefs.KinemaPreferences.bl_rna.properties
        for name, value in vars(graph.DEFAULTS).items():
            assert declared[name].default == value, name

    def test_reset_puts_them_back(self, preferences, graph):
        import bpy

        preferences.graph_open_in = "WINDOW"
        preferences.graph_normalize = False
        preferences.graph_playback_sidebar = False
        assert graph.settings().graph_open_in == "WINDOW", "precondition: read live"

        assert "FINISHED" in bpy.ops.kinema.reset_graph_preferences()
        assert vars(graph.settings()) == vars(graph.DEFAULTS)
