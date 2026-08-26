"""Kinematic chain maths. Pure NumPy -- runs without Blender."""

from __future__ import annotations

import numpy as np
import pytest

from ..conftest import load_addon_module

chain_mod = load_addon_module("solver.chain")
numpy_backend = load_addon_module("solver.numpy_backend")


def make_chain(dof: int = 3, link: float = 0.3):
    """A planar arm: three revolute joints about Y, each `link` apart in X."""
    rest = np.zeros((dof, 4, 4))
    for index in range(dof):
        rest[index] = np.eye(4)
        if index > 0:
            rest[index][0, 3] = link
    tool = np.eye(4)
    tool[0, 3] = link
    return chain_mod.Chain(
        bone_names=[f"j{i}" for i in range(dof)],
        is_revolute=np.array([True] * dof),
        rest_relative=rest,
        tool_offset=tool,
        lower=np.full(dof, -2.0),
        upper=np.full(dof, 2.0),
        limited=np.array([True] * dof),
    )


class TestForwardKinematics:
    def test_zero_configuration_stacks_the_links(self):
        chain = make_chain()
        assert np.allclose(chain.forward(np.zeros(3))[:3, 3], [0.9, 0, 0])

    def test_first_joint_rotates_the_whole_arm(self):
        """Rotation is about local Y, so the arm swings in the XZ plane."""
        chain = make_chain()
        pose = chain.forward(np.array([np.pi / 2, 0, 0]))
        assert np.allclose(pose[:3, 3], [0, 0, -0.9], atol=1e-9)

    def test_folding_back_returns_toward_the_base(self):
        chain = make_chain()
        reach = chain.forward(np.array([0.0, np.pi, 0.0]))[:3, 3]
        assert reach[0] == pytest.approx(0.3 - 0.3 - 0.3, abs=1e-9)

    def test_frames_includes_every_joint_plus_the_tool(self):
        chain = make_chain()
        assert chain.frames(np.zeros(3)).shape == (4, 4, 4)


class TestJacobian:
    def test_matches_finite_differences(self):
        """The analytic Jacobian is the thing DLS relies on; check it numerically."""
        chain = make_chain()
        q = np.array([0.3, -0.5, 0.8])
        analytic = chain.jacobian(q)

        epsilon = 1e-6
        for index in range(chain.dof):
            step = np.zeros(chain.dof)
            step[index] = epsilon
            forward = chain.forward(q + step)
            backward = chain.forward(q - step)
            linear = (forward[:3, 3] - backward[:3, 3]) / (2 * epsilon)
            assert np.allclose(analytic[:3, index], linear, atol=1e-5), index

    def test_prismatic_joint_contributes_no_rotation(self):
        chain = make_chain()
        chain.is_revolute = np.array([True, False, True])
        jacobian = chain.jacobian(np.zeros(3))
        assert np.allclose(jacobian[3:, 1], 0.0)


class TestPoseError:
    def test_identical_poses_give_zero(self):
        pose = np.eye(4)
        pose[:3, 3] = [1, 2, 3]
        assert np.allclose(chain_mod.pose_error(pose, pose), 0.0)

    def test_pure_translation(self):
        target = np.eye(4)
        target[:3, 3] = [0.1, 0, 0]
        error = chain_mod.pose_error(np.eye(4), target)
        assert np.allclose(error[:3], [0.1, 0, 0])
        assert np.allclose(error[3:], 0.0)

    def test_rotation_magnitude_is_the_angle(self):
        angle = 0.7
        target = np.eye(4)
        target[:3, :3] = [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
        error = chain_mod.pose_error(np.eye(4), target)
        assert np.linalg.norm(error[3:]) == pytest.approx(angle, abs=1e-9)

    def test_near_180_degrees_does_not_blow_up(self):
        """The usual axis-angle formula divides by sin(angle); guard the pole."""
        target = np.eye(4)
        target[:3, :3] = np.diag([1.0, -1.0, -1.0])  # exactly pi about X
        error = chain_mod.pose_error(np.eye(4), target)
        assert np.all(np.isfinite(error))
        assert np.linalg.norm(error[3:]) == pytest.approx(np.pi, abs=1e-6)


class TestClamp:
    def test_limits_are_applied(self):
        chain = make_chain()
        assert np.allclose(chain.clamp(np.array([5.0, -5.0, 0.5])), [2.0, -2.0, 0.5])

    def test_unlimited_joints_pass_through(self):
        """Continuous joints must keep multi-turn values."""
        chain = make_chain()
        chain.limited = np.array([False, True, True])
        assert chain.clamp(np.array([9.0, 5.0, 0.0]))[0] == pytest.approx(9.0)


class TestNumpySolver:
    def test_reaches_a_reachable_target(self):
        chain = make_chain()
        goal_q = np.array([0.4, -0.6, 0.3])
        target = chain.forward(goal_q)
        result = numpy_backend.solve(chain, np.zeros(3), target)
        assert result.converged
        assert result.position_error < 1e-3
        assert np.allclose(chain.forward(result.q)[:3, 3], target[:3, 3], atol=1e-3)

    def test_respects_joint_limits(self):
        chain = make_chain()
        chain.lower = np.full(3, -0.2)
        chain.upper = np.full(3, 0.2)
        target = np.eye(4)
        target[:3, 3] = [0.2, 0.0, -0.8]  # far outside the limited workspace
        result = numpy_backend.solve(chain, np.zeros(3), target)
        assert np.all(result.q >= chain.lower - 1e-9)
        assert np.all(result.q <= chain.upper + 1e-9)

    def test_unreachable_target_reports_failure_rather_than_diverging(self):
        chain = make_chain()
        target = np.eye(4)
        target[:3, 3] = [50.0, 0, 0]
        result = numpy_backend.solve(chain, np.zeros(3), target)
        assert not result.converged
        assert np.all(np.isfinite(result.q))

    def test_seeding_keeps_the_solution_nearby(self):
        """Seeding from the current pose is what stops viewport IK flipping."""
        chain = make_chain()
        goal_q = np.array([0.5, -0.4, 0.2])
        target = chain.forward(goal_q)
        near = numpy_backend.solve(chain, goal_q + 0.01, target)
        assert np.abs(near.q - goal_q).max() < 0.2
