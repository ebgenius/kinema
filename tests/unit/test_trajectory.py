"""Optimize Motion's problem, outside Blender: padding, and the limits it keeps to.

The claims checked here:
- a job is padded to its bucket with the robot at rest either side, pinned,
  and nothing asked of the tool there;
- the speed and acceleration costs measure exactly what the Motion Check
  measures, so a job the solve keeps within its limits is one the check passes;
- a move's frames-needed estimate counts only its own frames;
- the limit costs are made by the jaxls the problem is, and forgetting drops them;
- the collision cost measures what the Motion Check measures, box, sphere and
  floor, costs nothing where a capsule keeps far enough away, and has a
  gradient everywhere; capsules and obstacles are padded to a count, with the
  padding asking nothing, and the obstacles stand still with the robot either
  side of the job.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from ..conftest import load_addon_module

trajectory = load_addon_module("solver.trajectory")
base = load_addon_module("solver.base")
velocity = load_addon_module("rig.velocity")
check = load_addon_module("rig.motion_check")
geometry = load_addon_module("rig.collision")


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


def _rounded(over, band=None):
    """What the limit costs make of ``over``: a hinge with its corner rounded."""
    band = trajectory.LIMIT_BAND if band is None else band
    over = np.asarray(over, dtype=float)
    return np.where(over < band, np.clip(over, 0.0, band) ** 2 / (2.0 * band), over - band / 2.0)


class TestTheHinge:
    """Nothing within, the excess past, and no corner for a step to fall off."""

    jnp = pytest.importorskip("jax.numpy")

    def test_nothing_within_and_the_excess_past_the_band(self):
        over = np.array([-1.0, -1e-3, 0.0, 0.3, 0.5])
        got = np.asarray(trajectory.hinge(self.jnp, self.jnp.asarray(over), 0.1))
        np.testing.assert_allclose(got, [0.0, 0.0, 0.0, 0.25, 0.45], atol=1e-7)

    def test_its_slope_rises_without_a_jump(self):
        """A plain hinge's slope jumps from 0 to 1 at its corner; this one's doesn't."""
        jax = pytest.importorskip("jax")
        band = 0.1
        slope = jax.vmap(jax.grad(lambda x: trajectory.hinge(self.jnp, x, band)))
        over = self.jnp.linspace(-0.05, 0.2, 251)
        slopes = np.asarray(slope(over))
        assert slopes.min() == 0.0 and slopes.max() == pytest.approx(1.0)
        assert np.abs(np.diff(slopes)).max() < 0.02, "no jump anywhere"
        np.testing.assert_allclose(slopes[over <= 0.0], 0.0)
        np.testing.assert_allclose(slopes[over >= band + 1e-6], 1.0, rtol=1e-6)


class TestPasses:
    def test_each_pass_is_read_from_the_kernel_s_account(self):
        """Only the passes taken, each starting from the best cost before it."""
        why = (
            np.int32(0), np.array([0.5]),
            np.array([10.0]), np.array([3]),
            np.array([[2.0, 0.5, 0.5, np.nan, np.nan]]),
            np.array([[[0, 0, 1], [0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=bool),
            np.array([[40, 17, 8, 0, 0]]),
        )
        pending = trajectory.Pending(q=np.zeros((4, 2)), iterations=65, own=slice(1, 3), why=why)
        passes = pending.passes()
        assert [(p.before, p.after) for p in passes] == [(10.0, 2.0), (2.0, 0.5), (0.5, 0.5)]
        assert [p.iterations for p in passes] == [40, 17, 8]
        assert [(p.settled, p.stopped_on_step) for p in passes] == [
            (False, True), (False, False), (True, False),
        ]

    def test_each_start_s_passes_are_read_and_the_kept_one_is_the_default(self):
        """Two starts, the second with a pass at a larger saturation first: it is kept."""
        why = (
            np.int32(1), np.array([0.9, 0.4]),
            np.array([10.0, 10.0]), np.array([1, 2]),
            np.array([[0.9, np.nan], [3.0, 0.4]]),
            np.zeros((2, 2, 3), dtype=bool),
            np.array([[30, 0], [20, 15]]),
        )
        pending = trajectory.Pending(
            q=np.zeros((4, 2)), iterations=65, own=slice(1, 3), why=why,
            saturation=np.array([[0.02, 0.02], [0.2, 0.02]]),
        )
        assert pending.kept() == 1 and pending.ended() == [0.9, 0.4]
        kept = pending.passes()
        assert [(p.before, p.after, p.saturation) for p in kept] == [
            (10.0, 3.0, 0.2), (3.0, 0.4, 0.02),
        ]
        assert [(p.before, p.after) for p in pending.passes(start=0)] == [(10.0, 0.9)]

    def test_with_obstacles_there_are_two_starts_and_without_one(self):
        rows, most = trajectory.saturations(True)
        schedule, restarts = trajectory.SATURATION_SCHEDULE, trajectory.RESTARTS
        assert rows.shape == (2, len(schedule) + restarts + 1)
        np.testing.assert_allclose(rows[0], trajectory.SATURATION)
        np.testing.assert_allclose(rows[1, : len(schedule)], schedule)
        np.testing.assert_allclose(rows[1, len(schedule) :], trajectory.SATURATION)
        # Each as many restarts past its schedule as a solve had before.
        assert most.tolist() == [restarts + 1, len(schedule) + restarts + 1]
        rows, most = trajectory.saturations(False)
        assert rows.shape == (1, restarts + 1) and most.tolist() == [restarts + 1]

    def test_a_pending_solve_without_an_account_has_no_passes(self):
        assert trajectory.Pending(q=np.zeros((2, 1)), iterations=0, own=slice(0, 2)).passes() == []


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
        np.testing.assert_allclose(over, _rounded(measured / limit - 1.0), rtol=1e-6)
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
        np.testing.assert_allclose(over, _rounded(measured / limit - 1.0), rtol=1e-5)
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


class TestCostsBelongToTheirJaxls:
    """A cost made by one jaxls module is foreign to a problem built by another."""

    @staticmethod
    def _jaxls(tag):
        class Cost:
            @staticmethod
            def factory(function):
                return (tag, function)

        return type("Jaxls", (), {"Cost": Cost})()

    def test_another_jaxls_gets_costs_of_its_own(self, monkeypatch):
        monkeypatch.setattr(trajectory, "_costs", {})
        old, new = self._jaxls("old"), self._jaxls("new")
        assert trajectory._limit_costs(old)[0][0] == "old"
        speed, acceleration = trajectory._limit_costs(new)
        assert (speed[0], acceleration[0]) == ("new", "new")

    def test_the_same_jaxls_keeps_them(self, monkeypatch):
        monkeypatch.setattr(trajectory, "_costs", {})
        jaxls = self._jaxls("one")
        assert trajectory._limit_costs(jaxls)[0] is trajectory._limit_costs(jaxls)[0]

    def test_the_collision_cost_is_made_by_its_jaxls_too(self, monkeypatch):
        monkeypatch.setattr(trajectory, "_costs", {})
        old = self._jaxls("old")
        assert trajectory._collision_cost(old) is trajectory._collision_cost(old)
        assert trajectory._collision_cost(self._jaxls("new"))[0] == "new"

    def test_forgetting_everything_drops_them(self, monkeypatch):
        monkeypatch.setattr(trajectory, "_costs", {})
        trajectory._limit_costs(self._jaxls("one"))
        trajectory.forget()
        assert trajectory._costs == {}


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


def _obstacles():
    """A box, a sphere and a floor, each turned, round the origin."""
    turned = np.array([[0.0, -0.6, 0.8], [1.0, 0.0, 0.0], [0.0, 0.8, 0.6]])
    made = []
    for index, (kind, size) in enumerate(
        (
            (geometry.KIND_BOX, np.array([0.2, 0.1, 0.3])),
            (geometry.KIND_SPHERE, np.array([0.25])),
            (geometry.KIND_FLOOR, np.zeros(3)),
        )
    ):
        matrix = np.eye(4)
        matrix[:3, :3] = turned if index != 1 else np.eye(3)
        matrix[:3, 3] = (0.1 * index, -0.2, 0.05 * index - 0.4 * (index == 2))
        made.append(geometry.Obstacle(f"o{index}", kind, matrix, size))
    return made


def _as_arrays(jnp, obstacles):
    kind = jnp.asarray([trajectory.SHAPES.index(o.kind) for o in obstacles])
    rotation = jnp.asarray(np.array([o.matrix[:3, :3] for o in obstacles]), dtype=jnp.float32)
    centre = jnp.asarray(np.array([o.matrix[:3, 3] for o in obstacles]), dtype=jnp.float32)
    size = jnp.asarray(
        np.array([np.resize(np.asarray(o.size, dtype=float), 3) for o in obstacles]),
        dtype=jnp.float32,
    )
    return kind, rotation, centre, size


def _collision(frames=10, capsules=3, obstacles=2):
    frame = np.repeat(np.eye(4)[None, None], frames, axis=0).repeat(obstacles, axis=1)
    frame[:, :, 0, 3] = np.arange(frames)[:, None]
    return trajectory.Collision(
        link=np.arange(capsules),
        a=np.zeros((capsules, 3)),
        b=np.ones((capsules, 3)),
        radius=np.full(capsules, 0.05),
        placement=np.repeat(np.eye(4)[None], frames, axis=0),
        kind=np.zeros(obstacles, dtype=int),
        frame=frame,
        size=np.full((frames, obstacles, 3), 0.1),
        # Not zero at the job's ends, so the rest frames' zero is padding's own.
        weight=np.ones(frames),
        keep_clear=0.01,
    )


class TestKeepingClear:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    def test_each_shape_is_measured_as_the_check_measures_it(self):
        """Segments outside, touching and through each: the distance the check finds."""
        random = np.random.default_rng(3)
        a = random.uniform(-0.6, 0.6, (60, 3))
        b = a + random.uniform(-0.5, 0.5, (60, 3))
        obstacles = _obstacles()
        expected = geometry.clearances(a, b, np.zeros(len(a)), obstacles)
        got = trajectory.segment_clearance(
            self.jax, self.jnp, self.jnp.asarray(a, dtype=self.jnp.float32),
            self.jnp.asarray(b, dtype=self.jnp.float32), *_as_arrays(self.jnp, obstacles),
        )
        np.testing.assert_allclose(np.asarray(got), expected, atol=2e-5)
        assert (expected < 0.0).any() and (expected > 0.0).any(), "precondition: both"

    def test_it_has_a_gradient_everywhere(self):
        """At a sphere's centre and deep in a box: no square root of zero to make a NaN."""
        obstacles = _obstacles()
        points = self.jnp.asarray(
            [obstacles[0].matrix[:3, 3], obstacles[1].matrix[:3, 3]], dtype=self.jnp.float32
        )

        def total(a):
            return trajectory.segment_clearance(
                self.jax, self.jnp, a, a + 0.01, *_as_arrays(self.jnp, obstacles)
            ).sum()

        gradient = np.asarray(self.jax.grad(total)(points))
        assert np.isfinite(gradient).all()

    def test_its_gradient_is_the_distance_s_own(self):
        """The closest point found with the gradient stopped; the distance's own there."""
        from jax import config

        obstacles = _obstacles()
        a = np.array([[0.5, 0.3, 0.4], [-0.1, 0.6, 0.2], [0.3, -0.3, -0.5]])
        b = a + np.array([[-0.4, 0.1, 0.3], [0.5, -0.2, 0.1], [0.2, 0.3, 0.6]])
        config.update("jax_enable_x64", True)
        try:
            arrays = _as_arrays(self.jnp, obstacles)
            arrays = tuple(self.jnp.asarray(x, dtype=self.jnp.float64) if i else x
                           for i, x in enumerate(arrays))

            def total(flat):
                return trajectory.segment_clearance(
                    self.jax, self.jnp, flat.reshape(-1, 3), self.jnp.asarray(b), *arrays
                ).sum()

            flat = self.jnp.asarray(a.ravel())
            gradient = np.asarray(self.jax.grad(total)(flat))
            step = 1e-6
            numeric = np.array([
                (float(total(flat.at[i].add(step))) - float(total(flat.at[i].add(-step))))
                / (2 * step)
                for i in range(flat.size)
            ])
        finally:
            config.update("jax_enable_x64", False)
        np.testing.assert_allclose(gradient, numeric, atol=1e-4)

    def test_a_capsule_far_enough_away_costs_nothing(self):
        """Short of ``aim`` it costs about the shortfall, weighted; past it, nothing."""

        class Robot:
            @staticmethod
            def forward_kinematics(q):
                # Every link at the origin, unturned.
                return self.jnp.tile(self.jnp.asarray([1.0, 0, 0, 0, 0, 0, 0]), (2, 1))

        jnp = self.jnp
        sphere = geometry.Obstacle("ball", geometry.KIND_SPHERE, np.eye(4), np.array([0.1]))
        kind, rotation, centre, size = _as_arrays(jnp, [sphere])
        shared = {
            "link": jnp.asarray([0, 1, 0]),
            "a": jnp.asarray([[0.2, 0.0, 0.0], [0.135, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            "b": jnp.asarray([[0.2, 0.0, 1.0], [0.135, 0.0, 1.0], [0.0, 0.0, 0.0]]),
            "radius": jnp.asarray([0.05, 0.03, 0.05]),
            "capsule_mask": jnp.asarray([1.0, 1.0, 0.0]),
            "kind": kind,
            "obstacle_mask": jnp.asarray([1.0]),
            "aim": jnp.asarray(0.01),
            "saturation": jnp.asarray(trajectory.SATURATION),
        }
        own = {
            "placement": jnp.eye(4),
            "rotation": rotation,
            "centre": centre,
            "size": size,
            "weight": jnp.asarray(7.0),
        }
        short = np.asarray(trajectory.clearance_short({"q": jnp.zeros(1)}, Robot, "q", shared, own))
        # 50 mm clear of the ball, 5 mm clear, and a padded capsule inside it.
        # 5 mm clear is 5 short of the 10 asked: past the rounded corner.
        level = trajectory.SATURATION
        rounded = 0.005 - trajectory.CLEARANCE_BAND / 2.0
        np.testing.assert_allclose(
            short, [0.0, 7.0 * level * np.log1p(rounded / level), 0.0], atol=1e-6
        )


class TestPaddingTheCell:
    def test_counts_are_padded_to_the_next_bucket(self):
        assert trajectory.padded(1, trajectory.CAPSULE_BUCKETS) == 8
        assert trajectory.padded(11, trajectory.CAPSULE_BUCKETS) == 16
        assert trajectory.padded(0, trajectory.OBSTACLE_BUCKETS) == 4
        assert _collision(capsules=9, obstacles=5).shape == (16, 8)

    def test_too_many_are_refused(self):
        most = trajectory.OBSTACLE_BUCKETS[-1]
        with pytest.raises(base.SolverError, match=f"{most + 1} is too many"):
            trajectory.padded(most + 1, trajectory.OBSTACLE_BUCKETS)

    def test_the_cell_stands_still_either_side_and_asks_nothing(self):
        problem = _problem(frames=10)
        collision = _collision(frames=10)
        padded, own = trajectory.at_rest(replace(problem, collision=collision))
        rest, after = trajectory.REST_FRAMES, padded.frames - own.stop
        cell = padded.collision
        np.testing.assert_array_equal(cell.frame[own], collision.frame)
        np.testing.assert_array_equal(cell.frame[:rest], collision.frame[:1].repeat(rest, 0))
        np.testing.assert_array_equal(cell.frame[own.stop:], collision.frame[-1:].repeat(after, 0))
        np.testing.assert_array_equal(cell.placement[own], collision.placement)
        np.testing.assert_array_equal(cell.size[own], collision.size)
        assert len(cell.placement) == len(cell.size) == padded.frames
        np.testing.assert_array_equal(cell.weight[own], collision.weight)
        assert not cell.weight[:rest].any() and not cell.weight[own.stop:].any()

    def test_padding_asks_nothing(self):
        jnp = pytest.importorskip("jax.numpy")
        collision = _collision(frames=4, capsules=3, obstacles=2)
        arrays = trajectory._collision_arrays(jnp, collision)
        assert arrays["a"].shape == (8, 3) and arrays["rotation"].shape == (4, 4, 3, 3)
        np.testing.assert_array_equal(np.asarray(arrays["capsule_mask"]), [1] * 3 + [0] * 5)
        np.testing.assert_array_equal(np.asarray(arrays["obstacle_mask"]), [1, 1, 0, 0])
        assert float(arrays["aim"]) == pytest.approx(0.01 + trajectory.CLEARANCE_MARGIN)
        np.testing.assert_allclose(
            np.asarray(arrays["weight"]), collision.weight * trajectory.COLLISION_WEIGHT
        )

    def test_a_kernel_keeps_to_its_cell(self):
        optimizer = trajectory.Optimizer(32, (8, 4))
        with pytest.raises(base.SolverError, match="don't match this kernel"):
            optimizer.dispatch(None, _problem(frames=10))

