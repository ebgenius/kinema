"""Finding the mounting flange a TCP offset is measured from. Pure NumPy.

The claims: ``tool0`` wins over ``flange`` whatever the file order; a ``flange``
alone is turned Z out of the face; a name counts bare or behind the description's
own prefix, not behind any other; only links fixed to the joint's own link are
searched; a joint with neither gets nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from ..conftest import load_addon_module

kin = load_addon_module("rig.kinematics")

HALF = np.pi / 2.0
#: The KR10 R1100-2's own placements, from link_6.
FLANGE = ((0.0, 0.0, -0.0285), (0.0, HALF, 0.0))
TOOL0 = ((0.0, 0.0, -0.0285), (np.pi, 0.0, np.pi))


def _joint(name, parent, child, xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0), kind="revolute"):
    return kin.JointSpec(
        name=name, joint_type=kind, parent_link=parent, child_link=child,
        origin=kin.make_transform(xyz, rpy), axis=np.array([0.0, 0.0, 1.0]),
        lower=-1.0, upper=1.0,
    )


def _model(*extra, wrist_rpy=(0.3, -0.2, 0.5), prefix="") -> kin.RobotModel:
    """base -> shoulder -> wrist (turned, so a transform that ignores it shows),
    plus ``extra`` joints hanging off the wrist. ``prefix`` goes on every link, as a
    xacro macro's does; ``extra`` is expected to carry it already."""
    joints = [
        _joint("j1", prefix + "base", prefix + "shoulder", xyz=(0.0, 0.0, 0.4)),
        _joint("j2", prefix + "shoulder", prefix + "wrist", xyz=(0.5, 0.0, 0.0), rpy=wrist_rpy),
        *extra,
    ]
    links = {prefix + "base": kin.LinkSpec(prefix + "base")}
    for joint in joints:
        links[joint.child_link] = kin.LinkSpec(joint.child_link)
    return kin.RobotModel(name="probe", links=links, joints=joints, root_link=prefix + "base")


def _fixed(name, parent, child, placement):
    return _joint(name, parent, child, *placement, kind="fixed")


class TestWhichFrame:
    @pytest.mark.parametrize("flange_first", [True, False])
    def test_tool0_wins_whatever_the_order(self, flange_first):
        pair = [_fixed("to_flange", "wrist", "flange", FLANGE),
                _fixed("to_tool0", "wrist", "tool0", TOOL0)]
        model = _model(*(pair if flange_first else pair[::-1]))
        mount = kin.mount_frames(model)["j2"]
        assert (mount.link, mount.kind) == ("tool0", "tool0")
        np.testing.assert_allclose(mount.transform, kin.make_transform(*TOOL0), atol=1e-12)

    def test_its_z_is_out_of_the_flange_face(self):
        """The face is 28.5 mm along the wrist link's -Z: out of it is -Z."""
        model = _model(_fixed("to_flange", "wrist", "flange", FLANGE),
                       _fixed("to_tool0", "wrist", "tool0", TOOL0))
        z = kin.mount_frames(model)["j2"].transform[:3, 2]
        np.testing.assert_allclose(z, [0.0, 0.0, -1.0], atol=1e-12)

    def test_a_flange_alone_is_turned_to_what_tool0_would_be(self):
        """On the KR10, flange turned Z out of the face is exactly tool0."""
        alone = kin.mount_frames(_model(_fixed("to_flange", "wrist", "flange", FLANGE)))["j2"]
        assert (alone.link, alone.kind) == ("flange", "flange")
        np.testing.assert_allclose(alone.transform, kin.make_transform(*TOOL0), atol=1e-12)

    def test_the_descriptions_own_prefix_counts(self):
        """A xacro macro prefixes every link: kr10_tool0 on a robot that is all kr10_."""
        model = _model(_fixed("to_tool0", "kr10_wrist", "kr10_tool0", TOOL0), prefix="kr10_")
        assert kin.mount_frames(model)["j2"].link == "kr10_tool0"

    @pytest.mark.parametrize(
        "name", ["wrist_flange", "base_tool0", "tool0_holder", "mytool0x", "flanged", "tool"]
    )
    def test_other_names_do_not(self, name):
        """A structural wrist_flange on an unprefixed robot is a bracket, not the mount."""
        assert kin.mount_frames(_model(_fixed("to_it", "wrist", name, TOOL0))) == {}

    def test_another_prefix_does_not_either(self):
        model = _model(_fixed("to_it", "kr10_wrist", "kr10_arm_flange", TOOL0), prefix="kr10_")
        assert kin.mount_frames(model) == {}

    def test_it_is_found_through_a_chain_of_fixed_joints(self):
        model = _model(
            _fixed("to_adapter", "wrist", "adapter", ((0.0, 0.0, -0.01), (0.0, 0.0, 0.0))),
            _fixed("to_tool0", "adapter", "tool0", ((0.0, 0.0, -0.0185), TOOL0[1])),
        )
        mount = kin.mount_frames(model)["j2"]
        np.testing.assert_allclose(mount.transform[:3, 3], [0.0, 0.0, -0.0285], atol=1e-12)

    def test_not_past_another_joint(self):
        """A flange behind a gripper's finger joint is the finger's, not the wrist's."""
        model = _model(
            _joint("finger", "wrist", "finger_link", kind="prismatic"),
            _fixed("to_tool0", "finger_link", "tool0", TOOL0),
        )
        found = kin.mount_frames(model)
        assert "j2" not in found
        assert found["finger"].link == "tool0"

    def test_a_joint_with_no_flange_gets_nothing(self):
        assert kin.mount_frames(_model()) == {}
        assert "j1" not in kin.mount_frames(_model(_fixed("t", "wrist", "tool0", TOOL0)))

    def test_it_is_from_the_link_not_the_world(self):
        """Measured from the joint's link, so turning the wrist does not change it."""
        a = kin.mount_frames(_model(_fixed("t", "wrist", "tool0", TOOL0)))["j2"].transform
        b = kin.mount_frames(
            _model(_fixed("t", "wrist", "tool0", TOOL0), wrist_rpy=(-1.0, 0.7, 2.0))
        )["j2"].transform
        np.testing.assert_allclose(a, b, atol=1e-12)


class TestTheDescriptionsPrefix:
    @pytest.mark.parametrize(
        ("names", "prefix"),
        [
            (["kr10_base_link", "kr10_link_1", "kr10_tool0"], "kr10_"),
            (["ur/base", "ur/wrist_3", "ur/tool0"], "ur/"),
            (["base_link", "link_1", "tool0"], ""),
            (["link_1", "link_2", "link_3"], "link_"),
            (["robot", "rtool0"], ""),
        ],
    )
    def test_it_is_what_every_link_starts_with_up_to_a_separator(self, names, prefix):
        model = kin.RobotModel(
            name="probe", links={n: kin.LinkSpec(n) for n in names}, joints=[],
            root_link=names[0],
        )
        assert kin.description_prefix(model) == prefix
