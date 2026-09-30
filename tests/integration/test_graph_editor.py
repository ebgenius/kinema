"""Graph Editor, in Joints (FK): the parts that need no window. Needs a real ``bpy``.

Opening the editor needs a window, which a background Blender has none of. The
claims checked here are the rest:
- the joint bones are what gets selected, whatever kind of joint, and nothing
  else of the rig;
- the Kinema view sets what the preferences say, and Blender Defaults puts back
  what a new Graph Editor has;
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
