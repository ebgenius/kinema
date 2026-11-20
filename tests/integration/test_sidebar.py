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
