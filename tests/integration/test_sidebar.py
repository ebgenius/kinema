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
        cls
        for cls in panel.classes
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
            cls
            for cls in _kinema_panels(panel)
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
            ("operator", "kinema.find_solutions"),
            ("prop", "kinema_auto_configuration"),
            ("operator", "kinema.reset_pose"),
            ("operator", "kinema.snap_ik"),
            ("operator", "kinema.check_motion"),
            ("operator", "kinema.graph_editor"),
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
        assert splits == [{"factor": 0.5}] * 3, "three pairs, after Live IK alone"
        first_split = names.index("split")
        assert "operator" not in names[first_split : names.index("column", first_split)]
        assert names.count("column") >= 5, "the bar's, and one for each half"

    def test_configurations_say_the_one_picked_once_the_target_moved(self, panel, arm6):
        """n/m while they are for this pose; n alone once it moved, and nothing red."""
        import types

        import bpy
        from mathutils import Vector

        def texts():
            calls = []
            panel.KINEMA_PT_quick_bar.draw(
                types.SimpleNamespace(layout=_Layout(calls)), bpy.context
            )
            return [
                kwargs.get("text")
                for name, args, kwargs in calls
                if name == "operator" and args[0] == "kinema.find_solutions"
            ]

        assert "FINISHED" in bpy.ops.kinema.add_ik()
        arm6.kinema_ik_enabled = False
        assert texts() == ["Configurations"]
        assert "FINISHED" in bpy.ops.kinema.find_solutions(seeds=20)
        (fresh,) = texts()
        assert fresh.startswith("Config ") and "/" in fresh
        goal = arm6.pose.bones[arm6["kinema_ik_bone"]]
        goal.matrix.translation = goal.matrix.translation + Vector((0.0, 0.05, 0.0))
        bpy.context.view_layer.update()
        assert texts() == [fresh.split("/")[0]]

    def test_the_switches_play_while_on_and_pause_while_off(self, panel, arm6, addon):
        """Live IK and Auto Config, as the Inverse Kinematics panel's Live IK shows."""
        import bpy

        quick_bar = importlib.import_module(f"{addon.__name__}.ui.quick_bar")
        assert "FINISHED" in bpy.ops.kinema.add_ik()

        def icon(key):
            calls = []
            quick_bar.BUTTONS[key].draw(_Layout(calls), bpy.context, arm6)
            (prop,) = [kwargs for name, _, kwargs in calls if name == "prop"]
            return prop["icon"]

        for key, name in (
            ("LIVE_IK", "kinema_ik_enabled"),
            ("AUTO_CONFIG", "kinema_auto_configuration"),
        ):
            setattr(arm6, name, False)
            assert icon(key) == "PAUSE", key
            setattr(arm6, name, True)
            assert icon(key) == "PLAY", key
        arm6.kinema_ik_enabled = False

    def test_live_ik_offers_an_ik_target_where_there_is_none(self, panel, arm6):
        import bpy

        drawn = _drawn(panel.KINEMA_PT_quick_bar, bpy.context)
        assert ("operator", "kinema.add_ik") in drawn
        assert ("prop", "kinema_ik_enabled") not in drawn

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
        assert prefs.quick_button_keys()[:2] == ["CONFIGURATIONS", "LIVE_IK"]
        assert "CANCELLED" in bpy.ops.kinema.quick_button_move(index=0, step=-1)
        assert "FINISHED" in bpy.ops.kinema.quick_button_remove(index=6)
        assert "FINISHED" in bpy.ops.kinema.quick_button_add(button="KEY_JOINTS")
        assert prefs.quick_button_keys() == [
            "CONFIGURATIONS",
            "LIVE_IK",
            "AUTO_CONFIG",
            "REST_POSE",
            "SNAP_IK",
            "MOTION_CHECK",
            "KEY_JOINTS",
        ]
        assert _drawn(panel.KINEMA_PT_quick_bar, bpy.context)[0] == (
            "operator",
            "kinema.find_solutions",
        ), "the panel draws them: Configurations first"
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
    def test_the_quick_bar_has_a_header_to_drag_collapse_and_pin_it_by(self, panel):
        """Header-less, Blender outlined it as a panel all the same."""
        assert "HIDE_HEADER" not in getattr(panel.KINEMA_PT_quick_bar, "bl_options", set())

    def test_the_quick_bar_names_the_robot_and_says_when_it_isnt_selected(self, panel, arm6):
        import bpy

        def labels():
            calls = []
            panel.KINEMA_PT_quick_bar.draw(
                __import__("types").SimpleNamespace(layout=_Layout(calls)), bpy.context
            )
            return [kwargs.get("text") for name, _, kwargs in calls if name == "label"]

        header, preset = [], []
        panel.KINEMA_PT_quick_bar.draw_header(
            __import__("types").SimpleNamespace(layout=_Layout(header)), bpy.context
        )
        panel.KINEMA_PT_quick_bar.draw_header_preset(
            __import__("types").SimpleNamespace(layout=_Layout(preset)), bpy.context
        )
        assert [kwargs.get("text") for name, _, kwargs in header if name == "label"] == ["arm6"]
        assert [kwargs.get("text") for name, _, kwargs in preset if name == "label"] == ["6 DoF"]
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


class _Settings(_Layout):
    """A layout that records what is set on it, ``enabled`` among it."""

    def __setattr__(self, name, value):
        self.calls.append(("set", (name, value), {}))

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return _Settings(self.calls)

        return call


class TestSnapToTool:
    def test_pressable_only_while_the_target_is_off_the_tool(self, panel, arm6, addon):
        """Dragged out of reach, live IK tracks it but the tool can't follow."""
        import bpy
        from mathutils import Vector

        quick_bar = importlib.import_module(f"{addon.__name__}.ui.quick_bar")
        ik_ops = importlib.import_module(f"{addon.__name__}.ops.ik")

        def enabled():
            calls = []
            quick_bar.BUTTONS["SNAP_IK"].draw(_Settings(calls), bpy.context, arm6)
            assert ("operator", ("kinema.snap_ik",)) in [(n, a[:1]) for n, a, _ in calls]
            return [
                value
                for name, (attr, value), _ in (c for c in calls if c[0] == "set")
                if attr == "enabled"
            ] == [True]

        assert not ik_ops.goal_off_tool(arm6), "no IK target: nothing to snap"
        assert "FINISHED" in bpy.ops.kinema.add_ik()
        assert not ik_ops.goal_off_tool(arm6) and not enabled()
        goal = arm6.pose.bones[arm6["kinema_ik_bone"]]
        goal.location += Vector((0.0, 0.0, 5.0))
        bpy.context.view_layer.update()
        assert ik_ops.goal_off_tool(arm6) and enabled(), "5 m out of reach"
        assert "FINISHED" in bpy.ops.kinema.snap_ik()
        assert not ik_ops.goal_off_tool(arm6) and not enabled()

    def test_a_turned_target_on_a_scaled_robot_is_off_the_tool(self, panel, arm6, addon):
        """Root scaled by 2 scales the matrices compared: their rotation is compared
        without it, or a 0.1 rad turn read as none."""
        import bpy
        from mathutils import Euler, Vector

        ik_ops = importlib.import_module(f"{addon.__name__}.ops.ik")
        assert "FINISHED" in bpy.ops.kinema.add_ik()
        arm6.kinema_ik_enabled = False
        arm6.pose.bones["Root"].scale = Vector((2.0, 2.0, 2.0))
        bpy.context.view_layer.update()
        assert "FINISHED" in bpy.ops.kinema.snap_ik()
        assert not ik_ops.goal_off_tool(arm6), "precondition: on the tool"
        goal = arm6.pose.bones[arm6["kinema_ik_bone"]]
        goal.rotation_mode = "XYZ"
        goal.rotation_euler = Euler((0.1, 0.0, 0.0))
        bpy.context.view_layer.update()
        assert ik_ops.goal_off_tool(arm6)


class TestQuickBarBox:
    """The buttons in a box, and under them what most needs saying."""

    def _said(self, panel):
        import types

        import bpy

        calls = []
        panel.KINEMA_PT_quick_bar.draw(types.SimpleNamespace(layout=_Layout(calls)), bpy.context)
        return calls, [kwargs.get("text") for name, _, kwargs in calls if name == "label"]

    def test_the_buttons_sit_in_a_box(self, panel, arm6):
        calls, _ = self._said(panel)
        names = [name for name, _, _ in calls]
        assert names.index("box") < names.index("operator")

    def test_what_it_says_as_the_robot_goes(self, panel, arm6, addon):
        import bpy
        from mathutils import Vector

        handlers = importlib.import_module(f"{addon.__name__}.handlers")

        assert self._said(panel)[1] == ["Add an IK target to pose the tool by it"]
        # Bent, so a short move of the target is in reach: at rest arm6 is straight out.
        builder = importlib.import_module(f"{addon.__name__}.rig.builder")
        for bone, value in zip(
            builder.joint_bones(arm6), [0.4, -0.7, 1.1, 0.2, 0.5, 0.0], strict=True
        ):
            bone.rotation_euler[1] = value
        bpy.context.view_layer.update()
        assert "FINISHED" in bpy.ops.kinema.add_ik()
        arm6.kinema_ik_enabled = False
        assert self._said(panel)[1] == ["Live IK paused: pose the joints, or turn it on"]
        arm6.kinema_ik_enabled = True
        bpy.context.view_layer.update()
        assert self._said(panel)[1] == ["Following the target"]

        ik_ops = importlib.import_module(f"{addon.__name__}.ops.ik")
        assert "FINISHED" in bpy.ops.kinema.find_solutions(seeds=12)
        goal = arm6.pose.bones[arm6["kinema_ik_bone"]]
        goal.location += Vector((0.0, 0.0, 0.01))
        bpy.context.view_layer.update()
        handlers.solve_rig(arm6, force=True, live=True)  # as the handler would
        bpy.context.view_layer.update()
        assert self._said(panel)[1] == ["Following the target"], "a moved target isn't news"

        goal.location += Vector((0.0, 0.0, 5.0))
        bpy.context.view_layer.update()
        assert self._said(panel)[1] == [
            "The tool can't reach the target",
            "Try Config or Auto Config, or Snap to Tool",
        ]
        arm6.kinema_auto_configuration = True
        assert self._said(panel)[1] == ["The tool can't reach the target"], "not searched yet"
        handlers.solve_rig(arm6, force=True, live=True)  # searches, and finds nothing
        bpy.context.view_layer.update()
        assert self._said(panel)[1] == ["The target is out of reach"]

        assert "FINISHED" in bpy.ops.kinema.snap_ik()
        ik_ops.forget_auto_search(arm6.name)
        ik_ops._auto_switched.add(arm6.name)  # as a switch records it
        said = [
            "The configuration changed",
            "A linear move from the last waypoint can't cross it",
        ]
        assert self._said(panel)[1] == said
        assert self._said(panel)[1] == said, "kept, not for a while"
        api = importlib.import_module(f"{addon.__name__}.api")
        api.add_waypoint(arm6, "after the switch")
        assert self._said(panel)[1] == ["Following the target"], "until one is recorded"
        arm6.kinema_ik_enabled = False
