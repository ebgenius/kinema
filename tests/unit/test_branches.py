"""Finding distinct IK solutions: the arithmetic half, no Blender needed."""

from __future__ import annotations

import numpy as np
import pytest

from ..conftest import load_addon_module

branches = load_addon_module("solver.branches")
chain_mod = load_addon_module("solver.chain")


def _chain(dof: int = 3, lengths=None):
    """A planar arm of revolute joints, each offset along X from the last."""
    lengths = lengths or [0.4] * dof
    rest = np.stack([np.eye(4) for _ in range(dof)])
    for index in range(1, dof):
        rest[index][0, 3] = lengths[index - 1]
    tool = np.eye(4)
    tool[0, 3] = lengths[-1]
    return chain_mod.Chain(
        bone_names=[f"j{i}" for i in range(dof)],
        is_revolute=np.ones(dof, dtype=bool),
        rest_relative=rest,
        tool_offset=tool,
        lower=np.full(dof, -np.pi),
        upper=np.full(dof, np.pi),
        limited=np.ones(dof, dtype=bool),
    )


def _pose_error(chain, q, goal) -> float:
    """Position and orientation as one number, for the crude search below.

    Each scaled by its own tolerance, so a score under 1 means both halves are
    inside. Weighting them in raw units instead let position dominate: the walk
    settled on the right point facing a radian the wrong way and stopped, which
    is precisely the kind of non-solution ``collect`` now rejects.
    """
    position, orientation = branches.reach_error(chain, q, goal)
    return (
        position / branches.REACH_TOLERANCE
        + orientation / branches.ORIENTATION_TOLERANCE
    )


def _other_solution(chain, goal, avoid):
    """A second genuine solution for ``goal``, or None if there is none.

    Found rather than written down: these bones turn about their local Y with
    offsets along X, so the arm works in XZ and the elbow-up/elbow-down pair is
    not the sign flip it would be on a textbook planar 2R.

    A coarse grid first, then refinement. Coordinate descent alone settles into
    whichever basin it started in, and picking starts by hand is how a fixture
    ends up asserting something about the search rather than about the code.
    """
    axis = np.linspace(-np.pi, np.pi, 25)
    best, best_score = None, np.inf
    for q0 in axis:
        for q1 in axis:
            for q2 in axis:
                q = np.array([q0, q1, q2])
                if branches.joint_distance(q, avoid) <= 0.4:
                    continue
                score = _pose_error(chain, q, goal)
                if score < best_score:
                    best, best_score = q, score
    if best is None:
        return None

    refined = _walk_toward(chain, best, goal, target=0.05)
    position, orientation = branches.reach_error(chain, refined, goal)
    if position >= branches.REACH_TOLERANCE:
        return None
    if orientation >= branches.ORIENTATION_TOLERANCE:
        return None
    if branches.joint_distance(refined, avoid) <= branches.DISTINCT_TOLERANCE:
        return None
    return refined


def _walk_toward(chain, q, goal, target: float):
    """Nudge ``q`` until its tool pose is within ``target`` of ``goal``.

    Crude coordinate descent. Only here so a test can find a second genuine
    solution rather than assert one that was written down and might be wrong
    for this chain's axes.
    """
    q = np.array(q, dtype=float)
    step = 0.05
    for _ in range(4000):
        error = _pose_error(chain, q, goal)
        if error <= target:
            return q
        best, best_error = q, error
        for index in range(chain.dof):
            for sign in (1, -1):
                trial = q.copy()
                trial[index] += sign * step
                trial_error = _pose_error(chain, trial, goal)
                if trial_error < best_error:
                    best, best_error = trial, trial_error
        if best_error >= error:
            step *= 0.5
            if step < 1e-7:
                break
        q = best
    return q


class TestJointDistance:
    def test_identical_configurations_are_zero_apart(self):
        q = np.array([0.3, -1.2, 0.8])
        assert branches.joint_distance(q, q) == pytest.approx(0.0)

    def test_a_whole_turn_is_the_same_place(self):
        """A continuous joint reaches the same pose at any number of turns.

        Without wrapping, every one of those would be reported as a different
        configuration and the list would fill with the same arm.
        """
        q = np.array([0.3, -1.2, 0.8])
        assert branches.joint_distance(q, q + 2 * np.pi) == pytest.approx(0.0, abs=1e-9)

    def test_plus_and_minus_pi_are_the_same_place(self):
        assert branches.joint_distance(
            np.array([np.pi]), np.array([-np.pi])
        ) == pytest.approx(0.0, abs=1e-9)

    def test_it_reports_the_joint_that_differs_most(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([0.05, 0.9, 0.02])
        assert branches.joint_distance(a, b) == pytest.approx(0.9)

    def test_a_prismatic_joint_is_not_wrapped(self):
        """A rail at 0 m and the same rail at 6.28 m are not the same place.

        Wrapping a linear axis is nonsense that loses solutions quietly: the
        two compare equal modulo 2*pi, and one of two genuinely different
        alternatives would be folded away.
        """
        a = np.array([0.0, 0.0])
        b = np.array([0.0, 2 * np.pi])
        revolute = np.array([True, False])

        assert branches.joint_distance(a, b) == pytest.approx(0.0, abs=1e-9)
        assert branches.joint_distance(a, b, revolute) == pytest.approx(2 * np.pi)

    def test_revolute_joints_still_wrap_alongside_a_prismatic_one(self):
        a = np.array([0.0, 0.5])
        b = np.array([2 * np.pi, 0.5])
        revolute = np.array([True, False])
        assert branches.joint_distance(a, b, revolute) == pytest.approx(0.0, abs=1e-9)


class TestDistinct:
    def test_nothing_found_yet_is_always_distinct(self):
        assert branches.is_distinct(np.array([0.0, 0.0, 0.0]), [])

    def test_a_configuration_already_found_is_not(self):
        found = [np.array([0.3, -1.2, 0.8])]
        assert not branches.is_distinct(np.array([0.3, -1.2, 0.8]), found)

    def test_solver_noise_does_not_make_a_new_branch(self):
        """Two runs onto the same branch differ by far less than the tolerance."""
        found = [np.array([0.3, -1.2, 0.8])]
        nudged = np.array([0.3001, -1.2002, 0.7999])
        assert not branches.is_distinct(nudged, found)

    def test_a_genuinely_different_arm_is_distinct(self):
        found = [np.array([0.3, -1.2, 0.8])]
        assert branches.is_distinct(np.array([0.3, 1.2, -0.8]), found)


class TestSeeding:
    def test_seeds_stay_inside_the_joint_limits(self):
        chain = _chain()
        chain.lower[:] = [-0.5, -1.0, -2.0]
        chain.upper[:] = [0.5, 1.0, 2.0]
        seeds = branches.seed_configurations(chain, 200, np.random.default_rng(0))

        assert seeds.shape == (200, 3)
        assert np.all(seeds >= chain.lower)
        assert np.all(seeds <= chain.upper)

    def test_an_unbounded_joint_is_sampled_over_one_turn(self):
        """Sampling a continuous joint over all of R would put nearly every
        seed in the same place modulo a turn."""
        chain = _chain()
        chain.limited[:] = [True, False, True]
        seeds = branches.seed_configurations(chain, 200, np.random.default_rng(0))

        assert np.all(np.abs(seeds[:, 1]) <= branches.UNBOUNDED_RANGE)

    def test_degenerate_limits_do_not_raise(self):
        """A description with lower == upper, or the two the wrong way round."""
        chain = _chain()
        chain.lower[:] = [0.0, 1.0, 0.0]
        chain.upper[:] = [0.0, -1.0, 0.0]
        seeds = branches.seed_configurations(chain, 5, np.random.default_rng(0))
        assert np.all(np.isfinite(seeds))


class TestCollect:
    def test_a_configuration_that_missed_is_dropped(self):
        chain = _chain()
        goal = chain.forward(np.array([0.4, -0.7, 0.2]))
        missed = np.array([1.4, 0.9, -1.1])

        assert branches.collect(chain, goal, [missed]) == []

    def test_a_configuration_that_reached_is_kept(self):
        chain = _chain()
        q = np.array([0.4, -0.7, 0.2])
        goal = chain.forward(q)

        found = branches.collect(chain, goal, [q])
        assert len(found) == 1
        assert np.allclose(found[0].q, q)
        assert found[0].position_error == pytest.approx(0.0, abs=1e-9)

    def test_duplicates_are_folded(self):
        chain = _chain()
        q = np.array([0.4, -0.7, 0.2])
        goal = chain.forward(q)

        found = branches.collect(chain, goal, [q, q + 1e-5, q + 2 * np.pi])
        assert len(found) == 1, "the same arm was reported three times"

    def test_two_real_solutions_both_survive(self):
        """Three joints reaching a point in a plane is redundant, so the same
        point has genuinely different arms behind it. Both must be reported.

        The second one is found rather than written down: these bones turn about
        their local Y with offsets along X, so the arm works in XZ and the
        elbow-up/elbow-down pair is not the sign flip it would be on a textbook
        planar 2R.
        """
        chain = _chain(dof=3)
        first = np.array([0.0, 1.2, -1.2])
        goal = chain.forward(first)

        second = _other_solution(chain, goal, avoid=first)
        assert second is not None, (
            "the fixture found no second solution; the test proves nothing"
        )

        found = branches.collect(chain, goal, [second, first])
        assert len(found) == 2
        # Best-first, so the tie between two branches is broken by something
        # rather than by the order the seeds happened to come back in.
        assert found[0].position_error <= found[1].position_error

    def test_no_seed_reaching_the_goal_gives_nothing(self):
        chain = _chain()
        goal = chain.forward(np.array([0.4, -0.7, 0.2]))
        assert branches.collect(chain, goal, [np.array([2.0, 2.0, 2.0])]) == []

    def test_the_right_point_facing_the_wrong_way_is_not_a_solution(self):
        """The goal is a pose, not a point.

        Where the position is reachable and the orientation is not, a
        least-squares solve lands on the point pointing somewhere else. Keeping
        that and calling it a way to reach the goal is a lie the user would
        only catch by looking at the robot.
        """
        chain = _chain()
        q = np.array([0.4, -0.7, 0.2])
        goal = chain.forward(q).copy()
        # Same tool position, turned a quarter turn about the axis the joints
        # do not control -- so no configuration can achieve it.
        twist = np.eye(4)
        twist[:3, :3] = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
        )
        goal[:3, :3] = goal[:3, :3] @ twist[:3, :3]

        position, orientation = branches.reach_error(chain, q, goal)
        assert position == pytest.approx(0.0, abs=1e-9), "the point is still reached"
        assert orientation > branches.ORIENTATION_TOLERANCE

        assert branches.collect(chain, goal, [q]) == []

    def test_prismatic_alternatives_both_survive(self):
        """Two rail positions reaching one point must not fold into one.

        The regression: wrapping a prismatic axis made 0 m and 2*pi m compare
        equal, so one of them was dropped as a duplicate.
        """
        chain = _chain(dof=3)
        chain.is_revolute[:] = [True, True, False]
        near = np.array([0.0, 0.6, 0.0])
        goal = chain.forward(near)
        far = near.copy()
        far[2] = 2 * np.pi
        # Not the same arm, and not the same tool -- so `far` is dropped for
        # missing, which is the correct reason. What matters is that the
        # distance function no longer calls the two identical.
        assert branches.joint_distance(
            near, far, chain.is_revolute
        ) > branches.DISTINCT_TOLERANCE

        found = branches.collect(chain, goal, [near])
        assert len(found) == 1
