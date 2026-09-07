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


def _walk_toward(chain, q, goal, target: float):
    """Nudge ``q`` until its tool lands ``target`` metres from ``goal``.

    Crude gradient descent, only so a test can produce a configuration that
    *nearly* reaches -- which is what distinguishes ordering from luck.
    """
    q = np.array(q, dtype=float)
    step = 0.05
    for _ in range(2000):
        error = branches.reach_error(chain, q, goal)
        if error <= target:
            return q
        best, best_error = q, error
        for index in range(chain.dof):
            for sign in (1, -1):
                trial = q.copy()
                trial[index] += sign * step
                trial_error = branches.reach_error(chain, trial, goal)
                if trial_error < best_error:
                    best, best_error = trial, trial_error
        if best_error >= error:
            step *= 0.5
            if step < 1e-6:
                break
        q = best
    return q


class TestWrappedDistance:
    def test_identical_configurations_are_zero_apart(self):
        q = np.array([0.3, -1.2, 0.8])
        assert branches.wrapped_distance(q, q) == pytest.approx(0.0)

    def test_a_whole_turn_is_the_same_place(self):
        """A continuous joint reaches the same pose at any number of turns.

        Without wrapping, every one of those would be reported as a different
        configuration and the list would fill with the same arm.
        """
        q = np.array([0.3, -1.2, 0.8])
        assert branches.wrapped_distance(q, q + 2 * np.pi) == pytest.approx(0.0, abs=1e-9)

    def test_plus_and_minus_pi_are_the_same_place(self):
        assert branches.wrapped_distance(
            np.array([np.pi]), np.array([-np.pi])
        ) == pytest.approx(0.0, abs=1e-9)

    def test_it_reports_the_joint_that_differs_most(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([0.05, 0.9, 0.02])
        assert branches.wrapped_distance(a, b) == pytest.approx(0.9)


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

        second = _walk_toward(chain, np.array([0.0, -0.4, 1.6]), goal, target=1e-6)
        assert branches.reach_error(chain, second, goal) < branches.REACH_TOLERANCE, (
            "the fixture failed to find a second solution; the test proves nothing"
        )
        assert branches.wrapped_distance(first, second) > branches.DISTINCT_TOLERANCE, (
            "the walk landed back on the first solution"
        )

        found = branches.collect(chain, goal, [first, second])
        assert len(found) == 2

    def test_the_closest_comes_first(self):
        """Best-first, so the tie between two branches is broken by something."""
        chain = _chain()
        exact = np.array([0.4, -0.7, 0.2])
        goal = chain.forward(exact)
        # A second, genuinely different arm nudged until it lands just inside
        # the reach tolerance rather than on the goal.
        loose = np.array([0.4, 0.7, -0.2])
        loose = _walk_toward(chain, loose, goal, target=branches.REACH_TOLERANCE * 0.6)

        found = branches.collect(chain, goal, [loose, exact])

        assert len(found) == 2
        assert found[0].position_error < found[1].position_error
        assert np.allclose(found[0].q, exact)

    def test_no_seed_reaching_the_goal_gives_nothing(self):
        chain = _chain()
        goal = chain.forward(np.array([0.4, -0.7, 0.2]))
        assert branches.collect(chain, goal, [np.array([2.0, 2.0, 2.0])]) == []
