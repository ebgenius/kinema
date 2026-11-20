"""The Kinema tab's layout (#103). Needs a real ``bpy``.

The everyday panels stand at the top level, where Blender lets them be dragged,
collapsed and pinned like its own sidebar's; what is set once sits under Robot
Settings, as its subpanels.
"""

from __future__ import annotations

import importlib

import pytest

from ..conftest import requires_bpy

pytestmark = requires_bpy


@pytest.fixture
def panel(addon):
    return importlib.import_module(f"{addon.__name__}.ui.panel")


def _kinema_panels(panel):
    return [
        cls for cls in panel.classes
        if isinstance(cls, type) and getattr(cls, "bl_category", None) == panel.CATEGORY
    ]


def _in_order(classes):
    return [cls.bl_idname for cls in sorted(classes, key=lambda cls: cls.bl_order)]


class TestLayout:
    def test_the_everyday_panels_are_top_level_in_order(self, panel):
        top = [cls for cls in _kinema_panels(panel) if not getattr(cls, "bl_parent_id", "")]
        assert _in_order(top) == [
            "KINEMA_PT_quick_bar",
            "KINEMA_PT_joints",
            "KINEMA_PT_ik",
            "KINEMA_PT_waypoints",
            "KINEMA_PT_motions",
            "KINEMA_PT_motion_check",
            "KINEMA_PT_main",
        ]

    def test_robot_settings_holds_what_is_set_once(self, panel):
        assert panel.KINEMA_PT_main.bl_label == "Robot Settings"
        under = [
            cls for cls in _kinema_panels(panel)
            if getattr(cls, "bl_parent_id", "") == "KINEMA_PT_main"
        ]
        assert _in_order(under) == [
            "KINEMA_PT_import",
            "KINEMA_PT_bones",
            "KINEMA_PT_tcp",
            "KINEMA_PT_external_axes",
            "KINEMA_PT_motion_limits",
            "KINEMA_PT_collision",
            "KINEMA_PT_status",
        ]

    def test_no_panel_is_labelled_kinema(self, panel):
        """The tab says Kinema; the panels say what they are."""
        assert all(cls.bl_label != "Kinema" for cls in _kinema_panels(panel))

    def test_every_panel_is_registered_after_its_parent(self, panel):
        import bpy

        seen = set()
        for cls in _kinema_panels(panel):
            parent = getattr(cls, "bl_parent_id", "")
            assert not parent or parent in seen, cls.bl_idname
            seen.add(cls.bl_idname)
            assert hasattr(bpy.types, cls.bl_idname)


class _Layout:
    """Anything a panel asks of a layout, recorded; whatever it returns, another one."""

    def __init__(self, calls):
        self.__dict__["calls"] = calls

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if name == "panel":
                return _Layout(self.calls), None
            return _Layout(self.calls)

        return call

    def __setattr__(self, name, value):
        pass


def _drawn(panel_class, context) -> list:
    """What ``panel_class`` draws: ("operator", idname) and ("prop", name), in order."""
    calls = []
    panel_class.draw(__import__("types").SimpleNamespace(layout=_Layout(calls)), context)
    return [
        (name, args[0] if name == "operator" else args[1])
        for name, args, _ in calls
        if name in ("operator", "prop")
    ]


@pytest.fixture
def arm6(addon, fixture_dir, clean_scene):
    import bpy

    assert "FINISHED" in bpy.ops.kinema.build_robot(filepath=str(fixture_dir / "arm6.urdf"))
    rig = bpy.context.view_layer.objects.active
    rig.kinema_solver_mode = "NUMPY"
    return rig


class TestQuickBar:
    def test_the_default_buttons_in_order(self, panel, arm6):
        import bpy

        assert "FINISHED" in bpy.ops.kinema.add_ik()
        assert _drawn(panel.KINEMA_PT_quick_bar, bpy.context) == [
            ("prop", "kinema_ik_enabled"),
            ("operator", "kinema.reset_pose"),
            ("operator", "kinema.graph_editor"),
            ("operator", "kinema.find_solutions"),
            ("operator", "kinema.check_motion"),
        ]

    def test_two_of_a_row_get_half_the_bar_each(self, panel, arm6):
        """Split down the middle, each in its own column: an aligned row shared its
        width by content, and Configurations' arrows took more than their half."""
        import types

        import bpy

        calls = []
        panel.KINEMA_PT_quick_bar.draw(types.SimpleNamespace(layout=_Layout(calls)), bpy.context)
        names = [name for name, _, _ in calls]
        splits = [kwargs for name, _, kwargs in calls if name == "split"]
        assert splits == [{"factor": 0.5}, {"factor": 0.5}], "two pairs, after Live IK alone"
        first_split = names.index("split")
        assert "operator" not in names[first_split:names.index("column", first_split)]
        assert names.count("column") >= 5, "the bar's, and one for each half"

    def test_live_ik_offers_an_ik_target_where_there_is_none(self, panel, arm6):
        import bpy

        assert _drawn(panel.KINEMA_PT_quick_bar, bpy.context)[0] == ("operator", "kinema.add_ik")

    def test_with_no_robot_it_offers_an_import(self, panel, clean_scene):
        import bpy

        assert _drawn(panel.KINEMA_PT_quick_bar, bpy.context) == [
            ("operator", "kinema.import_urdf")
        ]

    def test_every_button_draws(self, panel, arm6, addon):
        """Each one the preferences can list, with and without an IK target."""
        import bpy

        quick_bar = importlib.import_module(f"{addon.__name__}.ui.quick_bar")
        for _ in range(2):
            for key, button in quick_bar.BUTTONS.items():
                calls = []
                button.draw(_Layout(calls), bpy.context, arm6)
                assert calls, key
            assert "FINISHED" in bpy.ops.kinema.add_ik()


@pytest.fixture
def prefs(addon):
    """The add-on's preferences module, with the quick bar put back after."""
    import bpy

    module = importlib.import_module(f"{addon.__name__}.prefs")
    yield module
    assert "FINISHED" in bpy.ops.kinema.quick_button_reset()


class TestQuickBarPreferences:
    def test_the_defaults(self, prefs, addon):
        quick_bar = importlib.import_module(f"{addon.__name__}.ui.quick_bar")
        assert prefs.quick_button_keys() == list(quick_bar.DEFAULT)

    def test_buttons_moved_removed_added_and_reset(self, prefs, panel, arm6, addon):
        import bpy

        quick_bar = importlib.import_module(f"{addon.__name__}.ui.quick_bar")
        assert "FINISHED" in bpy.ops.kinema.quick_button_move(index=0, step=1)
        assert prefs.quick_button_keys()[:2] == ["REST_POSE", "LIVE_IK"]
        assert "CANCELLED" in bpy.ops.kinema.quick_button_move(index=0, step=-1)
        assert "FINISHED" in bpy.ops.kinema.quick_button_remove(index=4)
        assert "FINISHED" in bpy.ops.kinema.quick_button_add(button="KEY_JOINTS")
        assert prefs.quick_button_keys() == [
            "REST_POSE", "LIVE_IK", "GRAPH_EDITOR", "CONFIGURATIONS", "KEY_JOINTS"
        ]
        assert _drawn(panel.KINEMA_PT_quick_bar, bpy.context)[0] == (
            "operator", "kinema.reset_pose"
        ), "the panel draws them"
        assert "FINISHED" in bpy.ops.kinema.quick_button_reset()
        assert prefs.quick_button_keys() == list(quick_bar.DEFAULT)

    def test_unknown_and_repeated_keys_are_left_out(self, prefs):
        import types

        stored = types.SimpleNamespace(quick_buttons="LIVE_IK,FROM_THE_FUTURE,,LIVE_IK,REST_POSE")
        assert prefs.quick_button_keys(stored) == ["LIVE_IK", "REST_POSE"]

    def test_the_section_lists_each_button_with_its_controls(self, prefs):
        import types

        calls = []
        stored = types.SimpleNamespace(quick_buttons="LIVE_IK,REST_POSE,GRAPH_EDITOR")
        prefs.KinemaPreferences._draw_quick_bar(stored, _Layout(calls))
        labels = [kwargs.get("text") for name, _, kwargs in calls if name == "label"]
        assert labels[1:5] == ["Live IK", "Spans the bar", "Rest Pose", "Graph Editor"]
        operators = [args[0] for name, args, _ in calls if name == "operator"]
        assert operators.count("kinema.quick_button_move") == 6
        assert operators.count("kinema.quick_button_remove") == 3
        menus = [args[0] for name, args, _ in calls if name == "operator_menu_enum"]
        assert menus == ["kinema.quick_button_add"]


class TestJointsHeader:
    def test_rest_pose_graph_editor_and_key_all_sit_in_its_header(self, panel, arm6):
        """Here whether the quick bar has them or not; the panel's body has the joints."""
        import types

        import bpy

        calls = []
        panel.KINEMA_PT_joints.draw_header_preset(
            types.SimpleNamespace(layout=_Layout(calls)), bpy.context
        )
        operators = [args[0] for name, args, _ in calls if name == "operator"]
        assert operators == ["kinema.reset_pose", "kinema.graph_editor", "kinema.key_joints"]
        body = _drawn(panel.KINEMA_PT_joints, bpy.context)
        assert ("operator", "kinema.reset_pose") not in body


class TestRobotNamed:
    def test_the_quick_bar_names_the_robot_and_says_when_it_isnt_selected(self, panel, arm6):
        import bpy

        def labels():
            calls = []
            panel.KINEMA_PT_quick_bar.draw(
                __import__("types").SimpleNamespace(layout=_Layout(calls)), bpy.context
            )
            return [kwargs.get("text") for name, _, kwargs in calls if name == "label"]

        assert labels()[:2] == ["arm6", "6 DoF"]
        assert "Not selected: the robot last shown" not in labels()
        arm6.select_set(False)
        bpy.context.view_layer.objects.active = None
        assert "Not selected: the robot last shown" in labels()

    def test_robot_settings_names_it_in_its_header(self, panel, arm6):
        import types

        import bpy

        calls = []
        panel.KINEMA_PT_main.draw_header(types.SimpleNamespace(layout=_Layout(calls)), bpy.context)
        assert [kwargs.get("text") for name, _, kwargs in calls if name == "label"] == ["arm6"]
