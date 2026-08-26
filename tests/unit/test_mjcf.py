"""MJCF reading. Pure Python -- runs without Blender."""

from __future__ import annotations

import numpy as np
import pytest

from ..conftest import load_addon_module

mjcf = load_addon_module("io.mjcf")


@pytest.fixture(scope="module")
def arm(fixture_dir):
    return mjcf.model_from_mjcf(fixture_dir / "mjcf_arm.xml")


class TestStructure:
    def test_reads_the_model_name(self, arm):
        assert arm.name == "mjcf_arm"

    def test_root_is_the_world(self, arm):
        assert arm.root_link == "world"

    def test_every_hinge_becomes_an_actuated_joint(self, arm):
        assert [j.name for j in arm.actuated_joints] == ["j1", "j2", "j3", "j4"]

    def test_a_body_without_joints_is_welded(self, arm):
        """'tool' has no <joint>, so it must arrive as a fixed attachment."""
        tool = next(j for j in arm.joints if j.child_link == "tool")
        assert tool.joint_type == "fixed"

    def test_multi_joint_body_expands_into_a_chain(self, arm):
        """URDF allows one joint per link, so a 2-DoF wrist needs an
        intermediate link between its two hinges."""
        assert "wrist__j3" in arm.links
        assert "wrist" in arm.links
        j4 = next(j for j in arm.joints if j.name == "j4")
        assert j4.parent_link == "wrist__j3"


class TestAngles:
    def test_degree_limits_are_converted_to_radians(self, arm):
        """MuJoCo defaults to degrees and URDF to radians -- getting this wrong
        scales every joint limit by 57."""
        j1 = next(j for j in arm.joints if j.name == "j1")
        assert j1.lower == pytest.approx(-np.pi / 2)
        assert j1.upper == pytest.approx(np.pi / 2)

    def test_euler_attribute_is_read_in_degrees(self, arm):
        """The wrist body is rotated 30 degrees about Z."""
        rotation = arm.link_frames()["wrist"][:3, :3]
        assert rotation[0, 0] == pytest.approx(np.cos(np.radians(30)), abs=1e-9)


class TestDefaults:
    def test_class_defaults_override_the_root_default(self, arm):
        """Root default range is +-90; the 'wrist' class narrows it to +-45."""
        j3 = next(j for j in arm.joints if j.name == "j3")
        assert j3.upper == pytest.approx(np.radians(45))

    def test_root_default_applies_where_no_class_is_named(self, arm):
        j1 = next(j for j in arm.joints if j.name == "j1")
        assert j1.upper == pytest.approx(np.radians(90))

    def test_axis_on_the_element_beats_the_default(self, arm):
        j2 = next(j for j in arm.joints if j.name == "j2")
        assert np.allclose(j2.axis, [0, 1, 0])


class TestJointPivotOffset:
    def test_pivot_offset_still_places_the_body_frame_correctly(self, arm):
        """MuJoCo pivots about the joint's own pos and leaves the body frame
        alone; a URDF joint's child frame *is* the pivot. 'upper' sits 0.3 along
        x from 'shoulder' with its pivot pulled back to the shoulder, so the
        body frame must still land at x = 0.3."""
        frames = arm.link_frames()
        assert np.allclose(
            frames["upper"][:3, 3], frames["shoulder"][:3, 3] + [0.3, 0, 0]
        )

    def test_the_pivot_itself_sits_at_the_parent(self, arm):
        """j2's pos of -0.3 puts its rotation axis back at the shoulder."""
        pivot = arm.joint_frames()["j2"]
        shoulder = arm.link_frames()["shoulder"]
        assert np.allclose(pivot[:3, 3], shoulder[:3, 3], atol=1e-9)


class TestGeometry:
    def test_box_half_extents_become_full_extents(self, arm):
        box = arm.links["base"].visuals[0]
        assert box.primitive[0] == "box"
        assert box.primitive[1]["size"] == pytest.approx([0.1, 0.1, 0.1])

    def test_capsules_become_cylinders(self, arm):
        """Blender has no capsule primitive, and the caps are cosmetic here."""
        visual = arm.links["shoulder"].visuals[0]
        assert visual.primitive[0] == "cylinder"
        assert visual.primitive[1]["length"] == pytest.approx(0.2)

    def test_class_geometry_defaults_apply(self, arm):
        visual = arm.links["wrist"].visuals[0]
        assert visual.primitive[0] == "sphere"
        assert visual.primitive[1]["radius"] == pytest.approx(0.03)

    def test_rgba_becomes_a_material_colour(self, arm):
        visual = arm.links["shoulder"].visuals[0]
        assert visual.material_color == pytest.approx((0.5, 0.5, 0.9, 1.0))


class TestErrors:
    def test_non_mjcf_file_is_rejected(self, fixture_dir):
        with pytest.raises(mjcf.MjcfError, match="not an MJCF"):
            mjcf.model_from_mjcf(fixture_dir / "arm3.urdf")

    def test_missing_file_is_rejected(self, fixture_dir):
        with pytest.raises(mjcf.MjcfError, match="no such file"):
            mjcf.model_from_mjcf(fixture_dir / "nope.xml")

    def test_ball_joint_is_rejected_clearly(self, tmp_path):
        """A 3-DoF spherical joint has no honest single-axis bone equivalent."""
        path = tmp_path / "ball.xml"
        path.write_text(
            '<mujoco model="b"><worldbody><body name="a">'
            '<joint name="j" type="ball"/></body></worldbody></mujoco>',
            encoding="utf-8",
        )
        with pytest.raises(mjcf.MjcfError, match="ball"):
            mjcf.model_from_mjcf(path)


class TestFloatingBase:
    def test_freejoint_becomes_a_fixed_attachment(self, tmp_path):
        """Every legged MJCF model has a floating base. Blender's Root bone is
        already a free control, so expanding it into six IK joints would hand
        an animator a control they would only ever move by hand."""
        path = tmp_path / "floating.xml"
        path.write_text(
            '<mujoco model="f"><worldbody>'
            '<body name="base" pos="0 0 0.5"><freejoint/>'
            '<geom type="box" size="0.1 0.1 0.1"/>'
            '<body name="leg" pos="0.1 0 0"><joint name="hip" axis="0 1 0"/>'
            '<geom type="capsule" size="0.02 0.1"/></body>'
            "</body></worldbody></mujoco>",
            encoding="utf-8",
        )
        model = mjcf.model_from_mjcf(path)
        base = next(j for j in model.joints if j.child_link == "base")
        assert base.joint_type == "fixed"
        assert [j.name for j in model.actuated_joints] == ["hip"]
        assert np.allclose(model.link_frames()["base"][:3, 3], [0, 0, 0.5])
