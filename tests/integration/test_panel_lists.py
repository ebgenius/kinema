"""The sidebar's lists show every row, so the mouse wheel over one scrolls the sidebar.

A list holding more rows than it shows scrolls itself under the wheel, and the
sidebar stops dead wherever the pointer crosses it. Drawn whole, a list has
nothing of its own to scroll.

Panels are drawn into a stand-in layout that records what was asked of it:
draw handlers run headless, but a real layout needs a window.
"""

from __future__ import annotations

import importlib
import types

import numpy as np
import pytest

from ..conftest import requires_bpy

pytestmark = requires_bpy

HOME = np.array([0.0, -0.6, 1.0, 0.0, 0.6, 0.0])


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


def _lists(panel_class, context) -> list[dict]:
    calls = []
    panel_class.draw(types.SimpleNamespace(layout=_Layout(calls)), context)
    return [kwargs for name, _, kwargs in calls if name == "template_list"]


@pytest.fixture
def modules(addon):
    def module(name):
        return importlib.import_module(f"{addon.__name__}.{name}")

    return module


@pytest.fixture
def arm6(addon, fixture_dir, clean_scene, modules):
    import bpy

    builder = modules("rig.builder")
    assert "FINISHED" in bpy.ops.kinema.build_robot(filepath=str(fixture_dir / "arm6.urdf"))
    rig = next(o for o in bpy.data.objects if builder.is_kinema_rig(o))
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    return rig


def test_the_bones_list_shows_every_bone_it_lists(arm6, modules):
    """The IK target included in the rig, left out of the list: rows for the rest."""
    import bpy

    builder, panel = modules("rig.builder"), modules("ui.panel")
    arm6.kinema_solver_mode = "NUMPY"
    assert "FINISHED" in bpy.ops.kinema.add_ik()
    ik_name = arm6[builder.PROP_IK_BONE]
    shown = len([bone for bone in arm6.pose.bones if bone.name != ik_name])
    assert shown > 6, "precondition: more bones than the list used to show"

    (bones,) = _lists(panel.KINEMA_PT_bones, bpy.context)
    assert bones["rows"] == bones["maxrows"] == shown


def test_the_waypoints_list_shows_every_waypoint(arm6, modules):
    import bpy

    builder, manager, panel = modules("rig.builder"), modules("solver.manager"), modules("ui.panel")
    arm6.kinema_solver_mode = "NUMPY"
    assert "FINISHED" in bpy.ops.kinema.add_ik()
    ik_name = arm6[builder.PROP_IK_BONE]
    for index in range(7):
        bpy.context.scene.frame_set(1 + 10 * index)
        for pose_bone, value in zip(builder.joint_bones(arm6), HOME + 0.1 * index, strict=True):
            pose_bone.rotation_euler[1] = float(value)
        bpy.context.view_layer.update()
        arm6.pose.bones[ik_name].matrix = arm6.pose.bones[manager.tip_bone(arm6)].matrix.copy()
        bpy.context.view_layer.update()
        assert "FINISHED" in bpy.ops.kinema.add_waypoint(name=f"P{index}")
    assert len(arm6.kinema_waypoints) > 4, "precondition: more waypoints than it used to show"

    (waypoints,) = _lists(panel.KINEMA_PT_waypoints, bpy.context)
    assert waypoints["rows"] == waypoints["maxrows"] == len(arm6.kinema_waypoints)
