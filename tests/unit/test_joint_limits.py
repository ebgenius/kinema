"""Reading velocity limits from MoveIt and ros2_control joint_limits.yaml."""

from __future__ import annotations

import pytest

from ..conftest import load_addon_module

pytest.importorskip("yaml")
joint_limits = load_addon_module("io.joint_limits")

MOVEIT = """
# From a MoveIt Setup Assistant config.
default_velocity_scaling_factor: 0.1
default_acceleration_scaling_factor: 0.1
joint_limits:
  panda_joint1:
    has_velocity_limits: true
    max_velocity: 2.1750
    has_acceleration_limits: false
    max_acceleration: 0
  panda_joint2:
    has_velocity_limits: true
    max_velocity: 2
  panda_finger_joint1:
    has_velocity_limits: false
    max_velocity: 0
  panda_joint3:
    has_acceleration_limits: true
    max_acceleration: 3.75
"""

ROS2_CONTROL = """
/**:
  ros__parameters:
    joint_limits:
      joint_1:
        has_position_limits: true
        min_position: -1.57
        max_position: 1.57
        has_velocity_limits: true
        max_velocity: 1.5
"""


class TestMoveIt:
    def test_limits_that_are_on_are_read(self):
        limits = joint_limits.velocity_limits(MOVEIT)
        assert limits["panda_joint1"] == pytest.approx(2.175)

    def test_an_integer_limit_is_a_float(self):
        limits = joint_limits.velocity_limits(MOVEIT)
        assert limits["panda_joint2"] == pytest.approx(2.0)

    def test_a_limit_turned_off_says_so(self):
        """MoveIt: 'Joint limits can be turned off with has_velocity_limits'."""
        assert joint_limits.velocity_limits(MOVEIT)["panda_finger_joint1"] is None

    def test_a_joint_without_the_flag_keeps_the_description_value(self):
        """Absent, not None: the file says nothing about this joint's velocity."""
        assert "panda_joint3" not in joint_limits.velocity_limits(MOVEIT)


class TestRos2Control:
    def test_a_nested_section_is_found(self):
        assert joint_limits.velocity_limits(ROS2_CONTROL) == {"joint_1": 1.5}


class TestUnusable:
    def test_no_section_is_an_error(self):
        with pytest.raises(joint_limits.JointLimitsError, match="No 'joint_limits'"):
            joint_limits.velocity_limits("controller_manager:\n  update_rate: 100\n")

    def test_broken_yaml_is_an_error(self):
        with pytest.raises(joint_limits.JointLimitsError, match="YAML"):
            joint_limits.velocity_limits("joint_limits: [unclosed\n")

    @pytest.mark.parametrize("value", ["0", "-1.0", ".inf", "fast"])
    def test_a_limit_that_means_nothing_is_skipped(self, value):
        text = f"joint_limits:\n  j:\n    has_velocity_limits: true\n    max_velocity: {value}\n"
        assert joint_limits.velocity_limits(text) == {}

    def test_a_quoted_false_is_false(self):
        text = 'joint_limits:\n  j:\n    has_velocity_limits: "false"\n'
        assert joint_limits.velocity_limits(text) == {"j": None}

    def test_a_missing_file_is_an_error(self, tmp_path):
        with pytest.raises(joint_limits.JointLimitsError, match="Could not open"):
            joint_limits.read_velocity_limits(tmp_path / "joint_limits.yaml")

    def test_a_file_on_disk(self, tmp_path):
        path = tmp_path / "joint_limits.yaml"
        path.write_text(MOVEIT, encoding="utf-8")
        assert joint_limits.read_velocity_limits(path)["panda_joint1"] == pytest.approx(
            2.175
        )
