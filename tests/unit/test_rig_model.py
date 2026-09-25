"""The model PyRoki gets for a rig with external axes, written from its bones.

The claim is that the model's forward kinematics *is* the rig's, at any joint
values. Checked here against the rig's own composition rule -- a pose bone is its
parent's pose, times the rest offset between them, times its own motion about +Y --
on random trees with random corrections, so it holds for any rig the builder makes,
not only for the ones in the fixtures.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ..conftest import load_addon_module

rig_model = load_addon_module("solver.rig_model")
kinematics = load_addon_module("rig.kinematics")


def _rotation(rng) -> np.ndarray:
    matrix = kinematics.make_transform((0.0, 0.0, 0.0), rng.uniform(-math.pi, math.pi, 3))
    return matrix


def _pose(rng) -> np.ndarray:
    return kinematics.make_transform(rng.uniform(-1, 1, 3), rng.uniform(-math.pi, math.pi, 3))


def _motion(joint_type: str, value: float) -> np.ndarray:
    """A bone's own motion: about or along its +Y."""
    matrix = np.eye(4)
    if joint_type == "prismatic":
        matrix[1, 3] = value
    else:
        c, s = math.cos(value), math.sin(value)
        matrix[0, 0], matrix[0, 2], matrix[2, 0], matrix[2, 2] = c, s, -s, c
    return matrix


def _urdf_motion(axis, joint_type: str, value: float) -> np.ndarray:
    """URDF's motion: about or along ``axis`` in the joint frame (Rodrigues)."""
    matrix = np.eye(4)
    axis = np.asarray(axis, dtype=float)
    if joint_type == "prismatic":
        matrix[:3, 3] = axis * value
        return matrix
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    matrix[:3, :3] = np.eye(3) + math.sin(value) * k + (1 - math.cos(value)) * (k @ k)
    return matrix


def _random_rig(rng, count: int = 7):
    """A random tree of joint bones, parents always earlier in the list."""
    records = []
    for index in range(count):
        parent = None if index == 0 or rng.random() < 0.2 else records[rng.integers(index)].name
        records.append(
            rig_model.BoneRecord(
                name=f"j{index}",
                parent=parent,
                rest=_pose(rng),
                correction=_rotation(rng),
                joint_type=["revolute", "prismatic", "continuous"][index % 3],
                child_link=f"link{index}",
                lower=-1.0 if index % 3 != 2 else None,
                upper=1.0 if index % 3 != 2 else None,
                velocity=0.5 * (index + 1),
            )
        )
    return records


def _rig_link_frames(records, q) -> dict[str, np.ndarray]:
    """The rig's link frames at ``q``, composed the way Blender poses bones."""
    rest = {r.name: r.rest for r in records}
    pose: dict[str, np.ndarray] = {}
    for record, value in zip(records, q, strict=True):
        if record.parent is None:
            parent_pose, parent_rest = np.eye(4), np.eye(4)
        else:
            parent_pose, parent_rest = pose[record.parent], rest[record.parent]
        pose[record.name] = (
            parent_pose @ np.linalg.inv(parent_rest) @ record.rest
            @ _motion(record.joint_type, value)
        )
    return {r.child_link: pose[r.name] @ r.correction for r in records}


def _model_link_frames(model, q) -> dict[str, np.ndarray]:
    values = dict(zip([j.name for j in model.joints], q, strict=True))
    frames = {model.root_link: np.eye(4)}
    pending = list(model.joints)
    while pending:
        for joint in list(pending):
            if joint.parent_link in frames:
                frames[joint.child_link] = (
                    frames[joint.parent_link] @ joint.origin
                    @ _urdf_motion(joint.axis, joint.joint_type, values[joint.name])
                )
                pending.remove(joint)
    return frames


@pytest.mark.parametrize("seed", range(5))
def test_the_model_moves_exactly_like_the_rig(seed):
    rng = np.random.default_rng(seed)
    records = _random_rig(rng)
    model = rig_model.model_from_records("robot", records)
    for _ in range(5):
        q = rng.uniform(-2.0, 2.0, len(records))
        expected = _rig_link_frames(records, q)
        actual = _model_link_frames(model, q)
        for link, frame in expected.items():
            np.testing.assert_allclose(actual[link], frame, atol=1e-10, err_msg=link)


def test_at_rest_the_links_are_where_the_bones_say():
    records = _random_rig(np.random.default_rng(9))
    frames = rig_model.model_from_records("robot", records).link_frames()
    for record in records:
        np.testing.assert_allclose(frames[record.child_link], record.link_frame, atol=1e-10)


def test_joints_carry_the_bones_names_types_and_limits():
    records = _random_rig(np.random.default_rng(4))
    model = rig_model.model_from_records("robot", records)
    assert model.root_link == rig_model.WORLD_LINK
    for record, joint in zip(records, model.joints, strict=True):
        assert (joint.name, joint.joint_type, joint.child_link) == (
            record.name, record.joint_type, record.child_link
        )
        assert (joint.lower, joint.upper, joint.velocity) == (
            record.lower, record.upper, record.velocity
        )
        parent = next((r for r in records if r.name == record.parent), None)
        assert joint.parent_link == (parent.child_link if parent else rig_model.WORLD_LINK)
        assert np.linalg.norm(joint.axis) == pytest.approx(1.0)


def test_it_renders_as_urdf():
    urdf_bridge = load_addon_module("solver.urdf_bridge")
    xml = urdf_bridge.urdf_xml(
        rig_model.model_from_records("robot", _random_rig(np.random.default_rng(5)))
    )
    assert f'<link name="{rig_model.WORLD_LINK}"' in xml
    assert xml.count("<joint ") == 7
