"""Reading joint limits from MoveIt and ros2_control joint_limits.yaml."""

from __future__ import annotations

import sys

import pytest

from ..conftest import load_addon_module

# No importorskip("yaml"): PyYAML is a shipped dependency, so its absence is a
# failure to see, not a reason to pass quietly.
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
  panda_joint4:
    has_velocity_limits: true
    max_velocity: 2.175
    has_acceleration_limits: true
    max_acceleration: 3.25
    has_jerk_limits: true
    max_jerk: 6500
    has_effort_limits: true
    max_effort: 87
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
        limits = joint_limits.joint_limits(MOVEIT)
        assert limits["panda_joint1"]["velocity"] == pytest.approx(2.175)

    def test_an_integer_limit_is_a_float(self):
        limits = joint_limits.joint_limits(MOVEIT)
        assert limits["panda_joint2"]["velocity"] == pytest.approx(2.0)

    def test_a_limit_turned_off_says_so(self):
        """MoveIt: 'Joint limits can be turned off with has_velocity_limits'."""
        limits = joint_limits.joint_limits(MOVEIT)
        assert limits["panda_finger_joint1"] == {"velocity": None}
        assert limits["panda_joint1"]["acceleration"] is None

    def test_a_kind_without_its_flag_keeps_the_description_value(self):
        """Absent, not None: the file says nothing about panda_joint3's speed."""
        assert joint_limits.joint_limits(MOVEIT)["panda_joint3"] == {"acceleration": 3.75}

    def test_every_kind_is_read(self):
        """Velocity, acceleration and jerk bound the motion; effort the torque."""
        assert joint_limits.joint_limits(MOVEIT)["panda_joint4"] == pytest.approx(
            {"velocity": 2.175, "acceleration": 3.25, "jerk": 6500.0, "effort": 87.0}
        )


class TestRos2Control:
    def test_a_nested_section_is_found(self):
        assert joint_limits.joint_limits(ROS2_CONTROL) == {"joint_1": {"velocity": 1.5}}

    def test_sections_are_merged_kind_by_kind(self):
        """A joint in two sections keeps what each says, not only the last one's."""
        text = ROS2_CONTROL + (
            "joint_limits:\n"
            "  joint_1:\n"
            "    has_acceleration_limits: true\n"
            "    max_acceleration: 4.0\n"
        )
        assert joint_limits.joint_limits(text) == {
            "joint_1": {"velocity": 1.5, "acceleration": 4.0}
        }


class TestUnusable:
    def test_no_section_is_an_error(self):
        with pytest.raises(joint_limits.JointLimitsError, match="No 'joint_limits'"):
            joint_limits.joint_limits("controller_manager:\n  update_rate: 100\n")

    def test_broken_yaml_is_an_error(self):
        with pytest.raises(joint_limits.JointLimitsError, match="YAML"):
            joint_limits.joint_limits("joint_limits: [unclosed\n")

    @pytest.mark.parametrize("kind", joint_limits.KINDS)
    @pytest.mark.parametrize("value", ["0", "-1.0", ".inf", "fast"])
    def test_a_limit_that_means_nothing_is_skipped(self, value, kind):
        text = f"joint_limits:\n  j:\n    has_{kind}_limits: true\n    max_{kind}: {value}\n"
        assert joint_limits.joint_limits(text) == {}

    def test_a_quoted_false_is_false(self):
        text = 'joint_limits:\n  j:\n    has_velocity_limits: "false"\n'
        assert joint_limits.joint_limits(text) == {"j": {"velocity": None}}

    def test_missing_pyyaml_is_an_error_not_a_traceback(self, monkeypatch):
        """A broken install should say what is wrong, from the operator's report."""
        monkeypatch.setitem(sys.modules, "yaml", None)
        with pytest.raises(joint_limits.JointLimitsError, match="PyYAML"):
            joint_limits.joint_limits(MOVEIT)

    def test_a_missing_file_is_an_error(self, tmp_path):
        with pytest.raises(joint_limits.JointLimitsError, match="Could not open"):
            joint_limits.read_joint_limits(tmp_path / "joint_limits.yaml")

    def test_a_file_on_disk(self, tmp_path):
        path = tmp_path / "joint_limits.yaml"
        path.write_text(MOVEIT, encoding="utf-8")
        limits = joint_limits.read_joint_limits(path)
        assert limits["panda_joint1"]["velocity"] == pytest.approx(2.175)
