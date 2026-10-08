"""Optimize Motion's problem, outside Blender: padding, and the limits it keeps to.

The claims checked here:
- a job is padded to its bucket with the robot at rest either side, pinned,
  and nothing asked of the tool there;
- the speed and acceleration costs measure exactly what the Motion Check
  measures, so a job the solve keeps within its limits is one the check passes;
- a move's frames-needed estimate counts only its own frames.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ..conftest import load_addon_module

trajectory = load_addon_module("solver.trajectory")
base = load_addon_module("solver.base")
velocity = load_addon_module("rig.velocity")
check = load_addon_module("rig.motion_check")


def _problem(frames=10, dof=2):
    q = np.linspace(0.0, 1.0, frames)[:, None] * np.arange(1, dof + 1)
    return trajectory.Problem(
        q_init=q,
        targets=np.repeat(np.eye(4)[None], frames, axis=0),
        position_weight=np.ones(frames),
        orientation_weight=np.ones(frames),
        prior=q.copy(),
        prior_weight=np.full((frames, dof), trajectory.JOB_WEIGHT),
        speed_limit=np.full(dof, 2.0),
        acceleration_limit=np.full(dof, 5.0),
        dt=1.0 / 30.0,
        link_index=3,
    )


class TestPadding:
    def test_a_job_is_padded_to_the_next_bucket(self):
        assert trajectory.bucket(1) == trajectory.BUCKETS[0]
        assert trajectory.bucket(32) == 32
        assert trajectory.bucket(33) == 64
        assert trajectory.bucket(trajectory.BUCKETS[-1]) == trajectory.BUCKETS[-1]

    def test_a_job_too_long_for_every_bucket_is_refused(self):
        """In the job's own frames: the rest frames either side are the solver's."""
        rest = 2 * trajectory.REST_FRAMES
        most = trajectory.BUCKETS[-1] - rest
        expected = f"a job of {most + 1} frames .* the most is {most}"
        with pytest.raises(base.SolverError, match=expected):
            trajectory.bucket(most + 1 + rest)

    def test_the_robot_rests_either_side_of_the_job(self):
        """Held at its first and last values, pinned, and nothing asked of the tool."""
        problem = _problem(frames=10)
        padded, own = trajectory.at_rest(problem)
        rest = trajectory.REST_FRAMES

        assert padded.frames == trajectory.bucket(10 + 2 * rest) == 32
        assert (own.start, own.stop) == (rest, rest + 10)
        np.testing.assert_array_equal(padded.q_init[own], problem.q_init)
        after = padded.frames - own.stop
        first, last = problem.q_init[:1], problem.q_init[-1:]
        np.testing.assert_array_equal(padded.q_init[:rest], first.repeat(rest, 0))
        np.testing.assert_array_equal(padded.q_init[own.stop:], last.repeat(after, 0))
        np.testing.assert_array_equal(padded.prior[own.stop:], problem.prior[-1:].repeat(after, 0))
        assert (padded.prior_weight[:rest] == trajectory.PIN_WEIGHT).all()
        assert (padded.prior_weight[own.stop:] == trajectory.PIN_WEIGHT).all()
        np.testing.assert_array_equal(padded.prior_weight[own], problem.prior_weight)
        assert not padded.position_weight[:rest].any()
        assert not padded.orientation_weight[own.stop:].any()
        assert padded.position_weight[own].all()


class TestLimitCosts:
    """The residuals against the Motion Check's own measures, on the same values."""

    jnp = pytest.importorskip("jax.numpy")

    def test_speed_is_measured_as_the_check_measures_it(self):
        fps, limit = 30.0, np.array([2.0, 4.0])
        before, now = np.array([0.0, 1.0]), np.array([0.1, 0.9])
        vals = {"now": self.jnp.asarray(now), "before": self.jnp.asarray(before)}
        over = np.asarray(trajectory.speed_over(vals, "now", "before", 1.0 / fps, limit, 1.0))
        measured = np.array(
            [velocity.speed(n, b, 1, fps) for n, b in zip(now, before, strict=True)]
        )
        np.testing.assert_allclose(over, np.maximum(0.0, measured / limit - 1.0), rtol=1e-6)
        assert over[0] > 0.0 and over[1] == 0.0, "precondition: one over its limit, one within"

    def test_acceleration_is_measured_as_the_check_measures_it(self):
        fps, limit = 30.0, np.array([5.0, 50.0])
        q = np.array([[0.0, 0.0], [0.02, 0.01], [0.05, 0.02]])
        vals = {name: self.jnp.asarray(row) for name, row in zip("abc", q, strict=True)}
        over = np.asarray(
            trajectory.acceleration_over(vals, "a", "b", "c", 1.0 / fps, limit, 1.0)
        )
        measured = np.array(
            [velocity.acceleration([(f, q[f, j]) for f in range(3)], fps) for j in range(2)]
        )
        np.testing.assert_allclose(over, np.maximum(0.0, measured / limit - 1.0), rtol=1e-5)
        assert over[0] > 0.0 and over[1] == 0.0, "precondition: one over its limit, one within"

    def test_each_is_weighted(self):
        vals = {"now": self.jnp.asarray([3.0]), "before": self.jnp.asarray([0.0])}
        once = np.asarray(trajectory.speed_over(vals, "now", "before", 1.0, np.array([2.0]), 1.0))
        thrice = np.asarray(trajectory.speed_over(vals, "now", "before", 1.0, np.array([2.0]), 3.0))
        np.testing.assert_allclose(thrice, 3.0 * once)


FPS = 30.0
SPEEDY = check.Joint("j1", velocity=1.0, acceleration=10.0)


def _samples(values):
    tool = np.eye(4)
    return [
        check.Sample(frame=frame, q=np.array([value]), tool=tool)
        for frame, value in enumerate(values)
    ]


class TestFramesNeeded:
    def test_a_move_too_fast_needs_frames_in_proportion(self):
        """At 1.5 times its top speed, half as many frames again."""
        values = [0.0, 0.0] + [1.5 * k / FPS for k in range(21)] + [1.0, 1.0]
        samples = _samples(values)
        move = check.check_move("B", "JOINT", 2, 22, np.eye(4), np.eye(4), samples, [SPEEDY], FPS)
        assert move.speed_ratio == pytest.approx(1.5)
        # Between two moves: its sudden start and stop are its neighbours' too.
        assert check.frames_needed(move, samples, [SPEEDY], FPS, False, False) == 30

    def test_a_move_within_its_limits_needs_none(self):
        values = [0.5 * k / FPS for k in range(21)]
        samples = _samples(values)
        move = check.check_move("B", "JOINT", 0, 20, np.eye(4), np.eye(4), samples, [SPEEDY], FPS)
        assert check.frames_needed(move, samples, [SPEEDY], FPS, False, False) == 0

    def test_a_waypoint_shared_with_a_faster_move_asks_nothing_of_this_one(self):
        """The acceleration where the next move sets off is that move's to answer for.

        As on arm6, where a move with 60 frames to spare was asked for 75 for the
        acceleration at the waypoint the next, too short, move set off from.
        """
        # Still until frame 10, then off at 1.5 times the top speed at once.
        values = [0.0] * 11 + [1.5 * k / FPS for k in range(1, 11)]
        samples = _samples(values)
        joint = check.Joint("j1", velocity=1.0, acceleration=1.0)
        before = check.check_move("A", "JOINT", 0, 10, np.eye(4), np.eye(4), samples, [joint], FPS)
        assert before.accel_ratio > 1.0, "precondition: the check reports it in this move"
        assert check.frames_needed(before, samples, [joint], FPS, True, False) == 0
        after = check.check_move("B", "JOINT", 10, 20, np.eye(4), np.eye(4), samples, [joint], FPS)
        assert check.frames_needed(after, samples, [joint], FPS, False, False) == 15

    def test_the_check_says_so(self):
        move = check.MoveCheck("D", "JOINT", 0, 30, speed_ratio=1.6, frames_needed=48)
        assert "needs about 48 frames, has 30" in check.problems(move)
        move.frames_needed = 0
        assert not any("needs about" in found for found in check.problems(move))


def test_frames_needed_scales_with_the_square_root_of_acceleration():
    """Accelerations fall with the square of the time a move takes."""
    values = [0.5 * (k / FPS) ** 2 * 4.0 for k in range(31)]  # 4 rad/s², twice the limit
    samples = _samples(values)
    joint = check.Joint("j1", velocity=100.0, acceleration=2.0)
    move = check.check_move("B", "JOINT", 0, 30, np.eye(4), np.eye(4), samples, [joint], FPS)
    expected = math.ceil(30 * math.sqrt(2.0))
    assert check.frames_needed(move, samples, [joint], FPS, True, True) in (expected, expected + 1)
