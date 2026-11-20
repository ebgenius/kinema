"""Telling a rig's robot from another description's: rig/same_robot.py (#92)."""

from __future__ import annotations

import pytest

from ..conftest import load_addon_module

kin = load_addon_module("rig.kinematics")
same = load_addon_module("rig.same_robot")


def _model(text: str, tmp_path, name="robot.urdf"):
    yourdfpy = pytest.importorskip("yourdfpy")
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return kin.model_from_urdf(yourdfpy.URDF.load(str(path), load_meshes=False))


@pytest.fixture
def arm6_text(fixture_dir) -> str:
    return (fixture_dir / "arm6.urdf").read_text(encoding="utf-8")


def _edited(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, old
    return text.replace(old, new)


def test_the_same_description_is_the_same_robot(arm6_text, tmp_path):
    result = same.compare(_model(arm6_text, tmp_path), _model(arm6_text, tmp_path, "b.urdf"))
    assert result.same_robot and result.limits == []


@pytest.mark.parametrize(
    "old, new, says",
    [
        # Another joint's name: one missing, one more.
        ('<joint name="joint6" type="revolute">', '<joint name="wrist" type="revolute">',
         "no joint 'joint6'"),
        ('<joint name="joint6" type="revolute">', '<joint name="joint6" type="prismatic">',
         "'joint6' is prismatic"),
        ('<origin xyz="0.4 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/>',
         '<origin xyz="0.4 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>',
         "'joint3' turns about an axis 90.0°"),
        # A longer upper arm moves every joint after it.
        ('<origin xyz="0.4 0 0" rpy="0 0 0"/>', '<origin xyz="0.45 0 0" rpy="0 0 0"/>',
         "'joint3' sits 50.0 mm"),
    ],
    ids=["renamed", "retyped", "axis", "moved"],
)
def test_another_robot_is_told_apart(arm6_text, tmp_path, old, new, says):
    result = same.compare(
        _model(arm6_text, tmp_path), _model(_edited(arm6_text, old, new), tmp_path, "b.urdf")
    )
    assert not result.same_robot
    assert any(says in line for line in result.differences), result.differences


def test_a_joint_moved_elsewhere_in_the_chain_is_told_apart(arm6_text, tmp_path):
    """joint6 hung from joint4's link instead of joint5's, its origin moved to keep
    it where it was at rest: the chain is compared on its own."""
    edited = _edited(
        arm6_text,
        '<parent link="wrist_2"/><child link="wrist_3"/>\n    <origin xyz="0.08 0 0"',
        '<parent link="wrist_1"/><child link="wrist_3"/>\n    <origin xyz="0.18 0 0"',
    )
    result = same.compare(_model(arm6_text, tmp_path), _model(edited, tmp_path, "b.urdf"))
    says = "'joint6' follows 'joint4' there, 'joint5' on the rig"
    assert any(says in line for line in result.differences), result.differences


def test_limits_that_differ_are_listed_not_refused(arm6_text, tmp_path):
    edited = _edited(
        arm6_text,
        '<limit lower="-2.5" upper="2.5" effort="150" velocity="3.15"/>',
        '<limit lower="-2.4" upper="2.5" effort="150" velocity="2.0"/>',
    )
    result = same.compare(_model(arm6_text, tmp_path), _model(edited, tmp_path, "b.urdf"))
    assert result.same_robot, result.differences
    assert result.limits == [
        "'joint2' lower limit -2.4, the rig's -2.5",
        "'joint2' speed limit 2, the rig's 3.15",
    ]
