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
