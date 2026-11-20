"""kinema.api, the public module (#102): rigs, joints, tools. Needs a real ``bpy``.

The claims:

* another extension finds the module as the README says, by the enabled
  add-on whose name ends in ``kinema``;
* rigs, joints and their limits, and the tool, read as the rig holds them;
* joint values read and set, the tool following in world space, and live IK
  leaving them where they were set;
* what isn't a rig, or the wrong number of values, is refused with why.
"""

from __future__ import annotations

import importlib
import math

import numpy as np
import pytest

from ..conftest import requires_bpy

pytestmark = requires_bpy


def _find_kinema():
    """The README's way: what another extension does."""
    import bpy

    for name in bpy.context.preferences.addons.keys():
        if name.rpartition(".")[2] == "kinema":
            return importlib.import_module(name + ".api")
    return None


@pytest.fixture
def api(addon):
    return importlib.import_module(f"{addon.__name__}.api")


def _build(source):
    import bpy

    assert "FINISHED" in bpy.ops.kinema.build_robot(filepath=str(source))
    return bpy.context.view_layer.objects.active


@pytest.fixture
def arm6(api, fixture_dir, clean_scene):
    rig = _build(fixture_dir / "arm6.urdf")
    assert api.is_rig(rig), "precondition: the import leaves its rig active"
    return rig


class TestFinding:
    def test_another_extension_finds_it_by_the_add_ons_name(self, api):
        found = _find_kinema()
        assert found is api
        assert found.VERSION == (1, 0)

    def test_the_readme_example_runs(self, api, arm6):
        """As written, on arm6: the joints it sets are where the rig then stands."""
        from pathlib import Path

        readme = (Path(__file__).parents[2] / "README.md").read_text(encoding="utf-8")
        section = readme.split("## Scripting Kinema", 1)[1]
        example = section.split("```python\n", 1)[1].split("```", 1)[0]
        exec(compile(example, "README.md", "exec"), {"__name__": "readme"})
        np.testing.assert_allclose(api.joint_values(arm6), [0, -0.5, 1, 0, 0.5, 0], atol=1e-6)

    def test_the_rigs_in_the_scene(self, api, arm6):
        import bpy

        assert api.rigs() == [arm6]
        assert api.rigs(bpy.context.scene) == [arm6]
        assert not api.is_rig(next(o for o in arm6.children if o.type == "MESH"))


class TestJoints:
    def test_joints_and_their_limits_as_the_rig_holds_them(self, api, arm6):
        joints = api.joints(arm6)
        assert [j.name for j in joints] == [f"joint{i}" for i in range(1, 7)]
        second = joints[1]
        assert (second.type, second.lower, second.upper) == ("revolute", -2.5, 2.5)
        assert (second.velocity, second.effort) == (3.15, 150.0)
        assert (second.acceleration, second.jerk, second.external) == (None, None, None)
        assert second.bone in arm6.pose.bones

    def test_a_limit_changed_on_the_rig_is_the_one_given(self, api, arm6, addon):
        """As Load Joint Limits writes it: the rig's, not the description's."""
        builder = importlib.import_module(f"{addon.__name__}.rig.builder")
        bone = arm6.pose.bones[api.joints(arm6)[1].bone].bone
        bone[builder.PROP_UPPER] = 1.25
        bone[builder.PROP_ACCELERATION] = 8.0
        second = api.joints(arm6)[1]
        assert (second.upper, second.acceleration) == (1.25, 8.0)

    def test_continuous_and_prismatic_joints(self, api, fixture_dir, clean_scene):
        rig = _build(fixture_dir / "continuous_and_mimic.urdf")
        by_name = {j.name: j for j in api.joints(rig)}
        assert (by_name["spin"].type, by_name["spin"].lower, by_name["spin"].upper) == (
            "continuous", None, None,
        )
        assert by_name["grip"].type == "prismatic"
        assert "grip_mirror" not in by_name, "a mimic joint follows: it isn't set"

    def test_an_external_axis_says_how_it_is_mounted(self, api, arm6):
        import bpy

        assert "FINISHED" in bpy.ops.kinema.add_external_axis(
            axis_name="spindle", mount="AFTER", kind="ROTARY", direction="Z", continuous=True,
        )
        joints = api.joints(arm6)
        assert [j.external for j in joints].count("AFTER") == 1
        assert sum(j.external is None for j in joints) == 6


class TestJointValues:
    def test_set_and_read_back(self, api, arm6):
        values = np.array([0.3, -0.4, 0.9, 0.2, 0.5, -0.1])
        api.set_joint_values(arm6, values)
        np.testing.assert_allclose(api.joint_values(arm6), values, atol=1e-6)

    def test_a_prismatic_joint_is_set_in_metres(self, api, fixture_dir, clean_scene):
        rig = _build(fixture_dir / "continuous_and_mimic.urdf")
        order = [j.name for j in api.joints(rig)]
        values = np.zeros(len(order))
        values[order.index("grip")] = 0.03
        api.set_joint_values(rig, values)
        np.testing.assert_allclose(api.joint_values(rig), values, atol=1e-6)

    def test_the_tool_follows_in_world_space(self, api, arm6):
        import bpy
        from mathutils import Vector

        arm6.location = Vector((1.0, 2.0, 0.0))
        bpy.context.view_layer.update()
        start = api.tool_frame(arm6)
        api.set_joint_values(arm6, [math.pi / 2, 0, 0, 0, 0, 0])
        turned = api.tool_frame(arm6)
        base = np.array([1.0, 2.0, 0.0])
        # A quarter turn of the base joint, about world Z through the base.
        x, y, z = start[:3, 3] - base
        np.testing.assert_allclose(turned[:3, 3] - base, [-y, x, z], atol=1e-5)
        assert abs(x) > 0.5, "precondition: the tool reaches out from the base"

    def test_live_ik_leaves_them_where_they_were_set(self, api, arm6, addon):
        """The goal goes with the tool: a solve, as a frame change forces, keeps
        the joints rather than dragging the arm back to the goal it had."""
        import bpy

        manager = importlib.import_module(f"{addon.__name__}.solver.manager")
        arm6.kinema_solver_mode = manager.MODE_NUMPY  # The same rule; no compile.
        assert "FINISHED" in bpy.ops.kinema.add_ik()
        arm6.kinema_ik_enabled = True
        values = np.array([0.6, -0.3, 0.8, 0.0, 0.4, 0.0])
        api.set_joint_values(arm6, values)
        scene = bpy.context.scene
        scene.frame_set(scene.frame_current + 1)
        scene.frame_set(scene.frame_current - 1)
        np.testing.assert_allclose(api.joint_values(arm6), values, atol=1e-3)

    def test_live_ik_on_a_rig_it_hasnt_solved_doesnt_undo_them(self, api, arm6, addon):
        """As on a file just opened: live IK has no goal on record for the rig, so
        the update that writes the joints would solve for the goal they left."""
        import bpy

        manager = importlib.import_module(f"{addon.__name__}.solver.manager")
        handlers = importlib.import_module(f"{addon.__name__}.handlers")
        arm6.kinema_solver_mode = manager.MODE_NUMPY
        assert "FINISHED" in bpy.ops.kinema.add_ik()
        arm6.kinema_ik_enabled = True
        bpy.context.view_layer.update()
        handlers.reset(arm6.name)
        values = np.array([0.6, -0.3, 0.8, 0.0, 0.4, 0.0])
        api.set_joint_values(arm6, values)
        np.testing.assert_allclose(api.joint_values(arm6), values, atol=1e-3)

    def test_the_wrong_number_of_values_is_refused(self, api, arm6):
        with pytest.raises(api.Error, match="has 6 joints; 5 values were given"):
            api.set_joint_values(arm6, [0.0] * 5)
        with pytest.raises(api.Error, match="finite"):
            api.set_joint_values(arm6, [0.0] * 5 + [float("nan")])
        with pytest.raises(api.Error, match="must be numbers"):
            api.set_joint_values(arm6, ["a"] * 6)

    def test_what_isnt_a_rig_is_refused(self, api, arm6):
        mesh = next(o for o in arm6.children if o.type == "MESH")
        for call in (api.joints, api.joint_values, api.tools, api.tool_frame):
            with pytest.raises(api.Error, match="isn't a Kinema rig"):
                call(mesh)


class TestTools:
    def test_the_tool_and_where_it_is_measured_from(self, api, arm6):
        import bpy

        (tool,) = api.tools(arm6)
        assert tool.name in arm6.pose.bones
        assert tool.link == "tool"
        assert tool.parent == api.joints(arm6)[-1].bone
        arm6.kinema_tcp_offset = (0.01, 0.02, 0.03)
        bpy.context.view_layer.update()
        np.testing.assert_allclose(api.tools(arm6)[0].offset, (0.01, 0.02, 0.03), atol=1e-6)
        np.testing.assert_allclose(
            api.tool_frame(arm6, api.tools(arm6)[0]), api.tool_frame(arm6), atol=1e-9
        )

    def test_the_tool_frame_is_the_flange_turned_by_the_offset(self, api, arm6, addon):
        """At rest: the frame a tool offset is measured from, moved and turned by it,
        so its Z is out of the tool -- not the marker bone's own axes."""
        import bpy
        from mathutils import Euler, Matrix, Vector

        builder = importlib.import_module(f"{addon.__name__}.rig.builder")
        assert "FINISHED" in bpy.ops.kinema.edit_tcp(
            parent=api.joints(arm6)[-1].bone, source="OFFSET",
            offset=(0.01, 0.02, 0.15), rotation=(0.3, -0.2, 0.5),
        )
        bpy.context.view_layer.update()
        (tool,) = api.tools(arm6)
        assert tool.rotation == pytest.approx((0.3, -0.2, 0.5)), "precondition: turned"
        flange = builder.tool_zero_frame(arm6.data.bones[tool.parent])
        expected = (
            arm6.matrix_world @ flange @ Matrix.Translation(Vector(tool.offset))
            @ Euler(tool.rotation, "XYZ").to_matrix().to_4x4()
        )
        np.testing.assert_allclose(api.tool_frame(arm6), np.array(expected), atol=1e-5)
