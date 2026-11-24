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

    def test_a_small_turn_off_a_float32_frame_is_measured(self):
        """A goal read off Blender's float32 matrices is orthonormal to ~1e-7, which
        can push the trace past 3: taken from its arccos, a turn of 4e-4 rad read 0,
        and the NumPy solver stopped there, reporting it had converged."""
        turn = 4e-4
        frame = np.eye(4)
        c, s = np.cos(0.7), np.sin(0.7)
        frame[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        noisy = frame.astype(np.float32).astype(float)
        noisy[:3, :3] *= 1.0 + 2e-7  # the most a float32 frame is off by, about
        twist = np.eye(4)
        c, s = np.cos(turn), np.sin(turn)
        twist[:3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]
        error = chain_mod.pose_error(frame @ twist, noisy)
        assert np.linalg.norm(error[3:]) == pytest.approx(turn, rel=0.01)

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


def _mimic(
    after, leader, *, revolute=True, multiplier=1.0, offset=0.0, x=0.3,
    lower=-np.inf, upper=np.inf, fixed=0.0,
):
    rest = np.eye(4)
    rest[0, 3] = x
    return chain_mod.MimicLink(
        after=after, rest_relative=rest, is_revolute=revolute, leader=leader,
        steps=(chain_mod.MimicStep(multiplier, offset, revolute, lower, upper),),
        fixed=fixed,
    )


class TestMimicInTheChain:
    """A mimic joint's bone between two joints (#104): it moves the joints below
    it by its leader's value, without being a degree of freedom of its own."""

    def _sliders(self):
        """A -> M -> B, all sliding along Y; M follows A one for one."""
        rest = np.stack([np.eye(4), np.eye(4)])
        return chain_mod.Chain(
            bone_names=["a", "b"],
            is_revolute=np.array([False, False]),
            rest_relative=rest,
            tool_offset=np.eye(4),
            lower=np.full(2, -1.0),
            upper=np.full(2, 1.0),
            limited=np.array([True, True]),
            mimics=(_mimic(0, 0, revolute=False, x=0.0),),
        )

    def test_a_collinear_mimic_doubles_its_leaders_slide(self):
        """The review's example: A = 0.02 with B = 0 moves B 0.04, not 0.02."""
        chain = self._sliders()
        assert chain.dof == 2
        tool = chain.forward(np.array([0.02, 0.0]))
        assert tool[1, 3] == pytest.approx(0.04)

    def test_its_column_adds_to_its_leaders(self):
        chain = self._sliders()
        jacobian = chain.jacobian(np.array([0.02, 0.0]))
        np.testing.assert_allclose(jacobian[:3, 0], [0.0, 2.0, 0.0])
        np.testing.assert_allclose(jacobian[:3, 1], [0.0, 1.0, 0.0])

    @pytest.mark.parametrize("after", [-1, 0, 1, 2])
    def test_the_jacobian_matches_finite_differences(self, after):
        """A turning mimic of j0, halved and offset, wherever it sits in the arm."""
        base = make_chain()
        chain = chain_mod.Chain(
            **{**base.__dict__, "mimics": (_mimic(after, 0, multiplier=0.5, offset=0.1),)}
        )
        q = np.array([0.3, -0.4, 0.5])
        analytic = chain.jacobian(q)
        step = 1e-6
        for index in range(3):
            dq = np.zeros(3)
            dq[index] = step
            ahead, behind = chain.forward(q + dq), chain.forward(q - dq)
            np.testing.assert_allclose(
                analytic[:3, index], (ahead[:3, 3] - behind[:3, 3]) / (2 * step), atol=1e-6
            )

    def test_a_mimic_off_the_chain_is_held_where_it_is(self):
        """Its leader isn't solved for: it stands at ``fixed`` whatever q is, bending
        the arm there, and adds nothing to any column."""
        base = make_chain()
        held = _mimic(1, -1, fixed=0.4, x=0.0)
        chain = chain_mod.Chain(**{**base.__dict__, "mimics": (held,)})
        q = np.array([0.3, -0.4, 0.5])
        bent = base.forward(np.array([0.3, -0.4 + 0.4, 0.5]))
        np.testing.assert_allclose(chain.forward(q), bent, atol=1e-12)
        step = 1e-6
        for index in range(3):
            dq = np.zeros(3)
            dq[index] = step
            numeric = (chain.forward(q + dq)[:3, 3] - chain.forward(q - dq)[:3, 3]) / (2 * step)
            np.testing.assert_allclose(chain.jacobian(q)[:3, index], numeric, atol=1e-6)

    def test_held_at_its_own_limit_it_stops_moving(self):
        mimic = _mimic(0, 0, multiplier=1.0, lower=-0.2, upper=0.2)
        assert mimic.value(np.array([0.5])) == pytest.approx(0.2)
        assert mimic.gain(np.array([0.5])) == 0.0
        assert mimic.gain(np.array([0.1])) == pytest.approx(1.0)

    def test_a_limit_partway_down_a_chain_of_mimics_holds(self):
        """M = A held to [0, 0.1], N = 2 * M: at A = 0.2, M stands at 0.1 and N at
        0.2 -- not 0.4, as N folded straight onto A would."""
        Step = chain_mod.MimicStep
        n = chain_mod.MimicLink(
            after=0, rest_relative=np.eye(4), is_revolute=False, leader=0,
            steps=(Step(1.0, 0.0, False, 0.0, 0.1), Step(2.0, 0.0, False, 0.0, 1.0)),
        )
        assert n.value(np.array([0.2])) == pytest.approx(0.2)
        assert n.gain(np.array([0.2])) == 0.0, "held at M's limit: it doesn't move"
        assert n.value(np.array([0.05])) == pytest.approx(0.1)
        assert n.gain(np.array([0.05])) == pytest.approx(2.0)


class TestClampAngle:
    """Blender's Limit Rotation holds a turn around the circle."""

    @pytest.mark.parametrize(
        "angle, lower, upper, held",
        [
            (0.5, -1.0, 1.0, 0.5),            # inside
            (2.5, -1.0, 1.0, 1.0),            # nearer the top end
            (-3.0, -1.0, 1.0, -1.0),          # nearer the bottom end, around the circle
            (2 * np.pi + 0.2, -1.0, 1.0, 0.2),  # past a whole turn: inside, read as a turn
            (0.0, 1.0, 2.0, 1.0),             # a range that leaves out zero
            (-2.5, 1.0, 2.0, 2.0),            # its nearer end is the far one along the line
            (5.0, -4.0, 4.0, 5.0),            # a whole turn or more: nothing held
        ],
    )
    def test_as_a_turn(self, angle, lower, upper, held):
        got = chain_mod.clamp_angle(angle, lower, upper)
        assert np.cos(got) == pytest.approx(np.cos(held))
        assert np.sin(got) == pytest.approx(np.sin(held))


class TestWrappedReading:
    def test_a_step_reading_a_limited_leaders_turn_reads_it_wrapped(self):
        """A continuous A at 2 pi + 0.2; M = A, limited to [-1, 1]; N = 0.5 * M.
        M's limit holds a turn around the circle, so M stands at 2 pi + 0.2 -- a
        turn of 0.2 -- and N's driver reads that turn off M's matrix: 0.1, not
        pi + 0.1."""
        Step = chain_mod.MimicStep
        n = chain_mod.MimicLink(
            after=0, rest_relative=np.eye(4), is_revolute=True, leader=0,
            steps=(Step(1.0, 0.0, True, -1.0, 1.0), Step(0.5, 0.0, True, wraps=True)),
        )
        assert n.value(np.array([2 * np.pi + 0.2])) == pytest.approx(0.1)
        assert n.gain(np.array([2 * np.pi + 0.2])) == pytest.approx(0.5)


class TestHeldTurnExpression:
    """The driver's cyclic hold: clamp_angle, written as Blender's simple
    expression, and short enough for its 256 characters."""

    @pytest.mark.parametrize(
        "lower, upper, multiplier, offset",
        [
            (0.0, 4.0, 0.5, 0.1),
            (-3.5, 0.0, 1.0, 0.0),
            (-4.0, 2.0, -1.0, 0.0),
            (1.0, 2.0, 0.3333333333333333, 0.016666666666666666),
            (-0.3333333333333333, 6.0, 2.0, -0.5),
            # The longest constants a description could hold, from the review.
            (-3.141593123456789, 3.141590987654321,
             -1.2345678901234567e-05, -1.2345678901234567e-05),
        ],
    )
    def test_it_is_clamp_angle_and_fits_a_driver(self, lower, upper, multiplier, offset):
        import math

        expression = chain_mod.mimic_expression(
            multiplier, offset, chain_mod.held_turn_expression(lower, upper)
        )
        assert len(expression) <= chain_mod.DRIVER_EXPRESSION_LIMIT, len(expression)
        names = {"floor": math.floor, "pi": math.pi, "min": min, "max": max}
        for angle in np.linspace(-9.0, 9.0, 181):
            got = eval(expression, {"__builtins__": {}}, {**names, "q": float(angle)})
            held = chain_mod.clamp_angle(angle, lower, upper)
            assert got == pytest.approx(multiplier * held + offset, abs=1e-9)

    def test_one_too_long_is_refused_rather_than_cut(self):
        with pytest.raises(ValueError, match="characters"):
            chain_mod.mimic_expression(1.0, 0.0, "q" + " + 0" * 100)


class TestWoundLeader:
    def test_a_small_hold_on_a_leader_wound_many_turns_counts(self):
        """q = 200 pi + 1.001 and M = -q held to [-1, 1]: held by 0.001 rad, which
        a relative tolerance would read as free."""
        Step = chain_mod.MimicStep
        m = chain_mod.MimicLink(
            after=0, rest_relative=np.eye(4), is_revolute=True, leader=0,
            steps=(Step(-1.0, 0.0, True, -1.0, 1.0),),
        )
        assert m.gain(np.array([200 * np.pi + 1.001])) == 0.0
        assert m.gain(np.array([200 * np.pi + 0.5])) == pytest.approx(-1.0)


class TestUnpinned:
    """#56: a joint a missed solve left at a limit is moved off it for a retry."""

    def test_a_pinned_joint_goes_halfway_to_the_middle_of_its_range(self):
        chain = make_chain()
        q = np.array([0.3, 2.0 - 0.001, -2.0 + 0.004])
        start = chain_mod.unpinned(chain, q)
        np.testing.assert_allclose(start, [0.3, (2.0 - 0.001) / 2.0, (-2.0 + 0.004) / 2.0])

    def test_none_when_no_joint_is_at_a_limit(self):
        assert chain_mod.unpinned(make_chain(), np.array([0.3, 1.9, -1.5])) is None

    def test_a_held_joint_stays_where_it_is(self):
        held = np.array([False, True, False])
        start = chain_mod.unpinned(make_chain(), np.array([0.3, 2.0, -2.0]), held)
        np.testing.assert_allclose(start, [0.3, 2.0, -1.0])

    @pytest.mark.parametrize("lower, upper, at", [(-2.0, np.inf, -2.0), (-np.inf, 2.0, 2.0)])
    def test_a_joint_limited_on_one_side_only_is_never_pinned(self, lower, upper, at):
        # A rig can carry a lower limit with no upper one, and still reads as limited;
        # the middle of its range is infinite, so moving toward it is no seed at all.
        chain = make_chain()
        chain.lower[0], chain.upper[0] = lower, upper
        assert chain.limited[0]
        assert chain_mod.unpinned(chain, np.array([at, 0.0, 0.0])) is None
