"""Held joints in the NumPy fallback: a rail the solver must not touch."""

from __future__ import annotations

import numpy as np

from ..conftest import load_addon_module

chain_mod = load_addon_module("solver.chain")
numpy_backend = load_addon_module("solver.numpy_backend")


def _rail_arm():
    """A rail, then a planar two-link arm.

    Every bone moves along or turns about a Y axis no joint tips, so the rail
    alone sets the tool's Y and the two revolute joints set X, Z and the pitch.
    That is what makes the rail's effect unmistakable: nothing else can move the
    tool along Y, so a solver that ignores the hold has to move the rail to
    reach a goal off to the side.
    """
    rest = np.stack([np.eye(4) for _ in range(3)])
    rest[1][0, 3] = 0.4
    rest[2][0, 3] = 0.4
    tool = np.eye(4)
    tool[0, 3] = 0.4
    return chain_mod.Chain(
        bone_names=["rail", "j1", "j2"],
        is_revolute=np.array([False, True, True]),
        rest_relative=rest,
        tool_offset=tool,
        lower=np.array([-1.5, -np.pi, -np.pi]),
        upper=np.array([1.5, np.pi, np.pi]),
        limited=np.ones(3, dtype=bool),
    )


GOAL_Q = np.array([0.2, 0.9, -1.1])
SEED = np.array([0.0, 0.5, -0.5])
HOLD_RAIL = np.array([True, False, False])


def test_unheld_the_solver_moves_the_rail_to_reach():
    """The control for the test below: the goal really does need the rail."""
    chain = _rail_arm()
    result = numpy_backend.solve(chain, SEED, chain.forward(GOAL_Q))
    assert abs(result.q[0] - GOAL_Q[0]) < 1e-3, "the fixture does not need the rail"


def test_a_held_rail_stays_exactly_where_it_was():
    chain = _rail_arm()
    result = numpy_backend.solve(chain, SEED, chain.forward(GOAL_Q), held=HOLD_RAIL)
    assert result.q[0] == SEED[0]


def test_the_other_joints_still_solve_around_a_held_rail():
    """Hold the rail where the goal needs it, and the arm reaches exactly."""
    chain = _rail_arm()
    seed = SEED.copy()
    seed[0] = GOAL_Q[0]
    result = numpy_backend.solve(chain, seed, chain.forward(GOAL_Q), held=HOLD_RAIL)

    assert result.q[0] == GOAL_Q[0]
    assert result.converged
    assert np.allclose(chain.forward(result.q), chain.forward(GOAL_Q), atol=1e-4)


def test_a_rail_held_outside_its_limits_is_not_clamped_back():
    """The backend holds exactly what it is given.

    Deciding what that is -- where the rig displays the joint, limit constraint
    applied -- is the caller's job; see manager.RigSolver.solve.
    """
    chain = _rail_arm()
    seed = SEED.copy()
    seed[0] = 2.0
    result = numpy_backend.solve(chain, seed, chain.forward(GOAL_Q), held=HOLD_RAIL)
    assert result.q[0] == 2.0
