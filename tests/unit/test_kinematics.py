"""Kinematics IR tests. Pure NumPy -- these run without Blender."""

from __future__ import annotations

import numpy as np
import pytest

from ..conftest import load_addon_module

kin = load_addon_module("rig.kinematics")


@pytest.fixture(scope="module")
def arm3(fixture_dir):
    yourdfpy = pytest.importorskip("yourdfpy")
    urdf = yourdfpy.URDF.load(str(fixture_dir / "arm3.urdf"))
    return kin.model_from_urdf(urdf)


@pytest.fixture(scope="module")
def mimicbot(fixture_dir):
    yourdfpy = pytest.importorskip("yourdfpy")
    urdf = yourdfpy.URDF.load(str(fixture_dir / "continuous_and_mimic.urdf"))
    return kin.model_from_urdf(urdf)


class TestRpyToMatrix:
    def test_identity(self):
        assert np.allclose(kin.rpy_to_matrix(0, 0, 0), np.eye(3))

    def test_yaw_90_rotates_x_onto_y(self):
        rotated = kin.rpy_to_matrix(0, 0, np.pi / 2) @ np.array([1.0, 0, 0])
        assert np.allclose(rotated, [0, 1, 0], atol=1e-12)

    def test_roll_90_rotates_y_onto_z(self):
        rotated = kin.rpy_to_matrix(np.pi / 2, 0, 0) @ np.array([0, 1.0, 0])
        assert np.allclose(rotated, [0, 0, 1], atol=1e-12)

    def test_is_orthonormal(self):
        matrix = kin.rpy_to_matrix(0.3, -0.7, 1.1)
        assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(matrix), 1.0)


class TestNormalize:
    def test_scales_to_unit_length(self):
        assert np.allclose(kin.normalize([0, 0, 5]), [0, 0, 1])

    def test_zero_axis_falls_back(self):
        """Malformed URDF: a zero axis must not produce NaNs."""
        assert np.allclose(kin.normalize([0, 0, 0]), [0, 0, 1])


class TestModelStructure:
    def test_parses_links_and_joints(self, arm3):
        assert arm3.name == "arm3"
        assert set(arm3.links) == {"base", "mount", "l1", "l2", "l3", "tool"}
        assert len(arm3.joints) == 5

    def test_root_is_the_link_with_no_parent(self, arm3):
        assert arm3.root_link == "base"

    def test_only_movable_joints_are_actuated(self, arm3):
        assert [j.name for j in arm3.actuated_joints] == ["joint1", "joint2", "joint3"]

    def test_axes_are_read_and_normalised(self, arm3):
        axes = {j.name: j.axis for j in arm3.actuated_joints}
        assert np.allclose(axes["joint1"], [0, 0, 1])
        assert np.allclose(axes["joint2"], [0, 1, 0])
        assert np.allclose(axes["joint3"], [1, 0, 0])

    def test_limits_are_read(self, arm3):
        joint1 = next(j for j in arm3.joints if j.name == "joint1")
        assert joint1.lower == pytest.approx(-1.5)
        assert joint1.upper == pytest.approx(1.5)
        assert joint1.has_limits


class TestForwardKinematics:
    def test_zero_pose_frames_are_hand_checkable(self, arm3):
        """Offsets chain: 0.1z fixed, 0.2z, 0.3x, 0.4x, 0.05x fixed."""
        frames = arm3.link_frames()
        assert np.allclose(frames["base"], np.eye(4))
        assert np.allclose(frames["mount"][:3, 3], [0, 0, 0.1])
        assert np.allclose(frames["l1"][:3, 3], [0, 0, 0.3])
        assert np.allclose(frames["l2"][:3, 3], [0.3, 0, 0.3])
        assert np.allclose(frames["l3"][:3, 3], [0.7, 0, 0.3])
        assert np.allclose(frames["tool"][:3, 3], [0.75, 0, 0.3])

    def test_joint_frame_equals_child_link_frame(self, arm3):
        joint_frames = arm3.joint_frames()
        link_frames = arm3.link_frames()
        assert np.allclose(joint_frames["joint2"], link_frames["l2"])


class TestFixedJointFolding:
    def test_fixed_joints_get_no_bone_but_links_keep_an_owner(self, arm3):
        """A fixed joint is inert; its child must ride the last real joint."""
        owner = arm3.nearest_actuated_ancestor()
        assert owner["base"] is None
        assert owner["mount"] is None, "fixed joint before any DoF stays on root"
        assert owner["l1"] == "joint1"
        assert owner["l2"] == "joint2"
        assert owner["l3"] == "joint3"
        assert owner["tool"] == "joint3", "fixed tool must follow the last joint"


class TestContinuousAndMimic:
    def test_continuous_joint_is_never_limited(self, mimicbot):
        """Clamping a continuous joint would break multi-turn rotation."""
        spin = next(j for j in mimicbot.joints if j.name == "spin")
        assert spin.joint_type == "continuous"
        assert spin.is_actuated
        assert not spin.has_limits

    def test_mimic_joint_is_not_a_control(self, mimicbot):
        mirror = next(j for j in mimicbot.joints if j.name == "grip_mirror")
        assert mirror.mimic_joint == "grip"
        assert mirror.mimic_multiplier == pytest.approx(-1.0)
        assert not mirror.is_actuated, "a mimic joint must not become a control"
        assert [j.name for j in mimicbot.actuated_joints] == ["spin", "grip"]


class TestVisuals:
    def test_primitives_are_captured_with_origin(self, arm3):
        visuals = arm3.links["l2"].visuals
        assert len(visuals) == 1
        assert visuals[0].primitive[0] == "cylinder"
        assert visuals[0].primitive[1]["radius"] == pytest.approx(0.03)
        assert np.allclose(visuals[0].origin[:3, 3], [0.05, 0, 0])

    def test_named_material_colour_is_resolved(self, arm3):
        visual = arm3.links["l1"].visuals[0]
        assert visual.material_color is not None
        assert visual.material_color[:3] == pytest.approx((0.5, 0.5, 0.5))


class TestUnsupportedJoints:
    def test_floating_joint_is_rejected_loudly(self, tmp_path):
        """Better an explicit error than a silently mis-rigged robot."""
        yourdfpy = pytest.importorskip("yourdfpy")
        path = tmp_path / "floating.urdf"
        path.write_text(
            '<robot name="f"><link name="a"/><link name="b"/>'
            '<joint name="j" type="floating">'
            '<parent link="a"/><child link="b"/></joint></robot>',
            encoding="utf-8",
        )
        urdf = yourdfpy.URDF.load(str(path))
        with pytest.raises(kin.UnsupportedJointError, match="floating"):
            kin.model_from_urdf(urdf)
