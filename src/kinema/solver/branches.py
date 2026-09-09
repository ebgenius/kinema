"""Finding the other configurations that reach a pose.

A six-axis arm can put its tool in one place up to eight different ways: elbow
bent one way or the other, wrist flipped, base swung round behind. They are all
correct, and which one you get out of an IK solver is decided by nothing more
principled than where the arm happened to be when it solved.

PyRoki does not enumerate them. It is a nonlinear least-squares solver, so it
converges to whichever solution its seed is nearest -- which is exactly what
makes live IK feel stable, and exactly what hides the alternatives. So they are
*found* rather than derived: seed from configurations scattered across the joint
limits, solve each, keep the ones that actually reached the goal, and fold away
the duplicates.

That makes this a search, with a search's honesty problem. It reports what these
seeds reached, which is a **lower bound** -- more seeds find more. Nothing here
claims to have enumerated anything.

Deliberately free of ``bpy``, like ``solver/chain.py``: the seeding, the reach
test and the duplicate test are arithmetic over a :class:`~.chain.Chain`, so
they can be tested in a plain venv. The Blender half -- running the solver,
storing what came back -- lives in ``ops/ik.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chain import Chain

#: How far apart two configurations must be to count as different, on the joint
#: that differs most: radians for a revolute joint, metres for a prismatic one.
#: A tenth of either is far wider than solver noise between two runs onto the
#: same branch, and far narrower than the gap between genuinely different ones,
#: which is typically most of a half-turn or most of a rail.
DISTINCT_TOLERANCE = 0.1

#: How close the tool has to land for a solution to count as having reached the
#: goal: metres of position, radians of orientation. Generous next to the
#: sub-micron a converged PyRoki solve manages -- these are here to throw out
#: the seeds that did not converge at all, not to grade the ones that did.
REACH_TOLERANCE = 5e-4
ORIENTATION_TOLERANCE = 1e-2

#: Seeds to try. Enough that a 6R arm reliably turns up its distinct branches,
#: few enough that the search stays interactive: PyRoki solves warm in ~5-20 ms,
#: so this is well under a second once the kernel is compiled.
DEFAULT_SEEDS = 40

#: What a joint's sampling range falls back to when the description does not
#: bound it. Continuous joints are genuinely unbounded, and seeding one across
#: all of R would put almost every seed in the same place modulo a turn.
UNBOUNDED_RANGE = np.pi


@dataclass(frozen=True)
class Solution:
    """One configuration that reaches the goal."""

    q: np.ndarray
    #: How far the tool ended up from where it was asked to be, in metres.
    position_error: float


def seed_configurations(chain: Chain, count: int, rng) -> np.ndarray:
    """``count`` configurations scattered across the chain's joint limits.

    Uniform rather than clever. The point of a seed is only to land in a
    different basin of attraction from the last one, and a solver that is going
    to converge does so from anywhere in the basin -- so spending effort on
    where exactly to start buys nothing that more seeds would not.
    """
    lower = np.where(chain.limited, chain.lower, -UNBOUNDED_RANGE)
    upper = np.where(chain.limited, chain.upper, UNBOUNDED_RANGE)
    # A joint whose limits are inverted or degenerate would make uniform() raise
    # or return a constant; neither is worth failing a search over.
    span = np.maximum(upper - lower, 0.0)
    return lower + rng.random((count, chain.dof)) * span


def joint_distance(a: np.ndarray, b: np.ndarray, is_revolute=None) -> float:
    """The largest per-joint difference between two configurations.

    Revolute joints are compared *wrapped*, because a joint at +pi and the same
    joint at -pi are the same place, and a continuous joint reaches the same
    pose at any number of whole turns -- without wrapping, every one of those
    would be reported as a different configuration.

    Prismatic joints are compared directly. Wrapping a linear axis would be
    nonsense in a way that quietly loses solutions: a rail at 0 m and the same
    rail at 6.28 m are not the same place, but modulo 2*pi they compare equal,
    and one of two genuinely different alternatives would be dropped.

    ``is_revolute`` defaults to all-revolute, which is what every arm without a
    rail or a gripper is.
    """
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if is_revolute is None:
        wrapped = np.angle(np.exp(1j * delta))
    else:
        revolute = np.asarray(is_revolute, dtype=bool)
        wrapped = np.where(revolute, np.angle(np.exp(1j * delta)), delta)
    return float(np.max(np.abs(wrapped)))


def is_distinct(
    q: np.ndarray, found, tolerance: float = DISTINCT_TOLERANCE, is_revolute=None
) -> bool:
    """True if ``q`` is not already in ``found``."""
    return all(
        joint_distance(q, other, is_revolute) >= tolerance for other in found
    )


def reach_error(chain: Chain, q: np.ndarray, goal: np.ndarray) -> tuple[float, float]:
    """How far the tool lands from ``goal``, as (metres, radians).

    Measured off the chain rather than taken from the solver's own report, so a
    solution is checked against the rig the user is looking at.

    Both halves, because the goal is a full pose. An earlier version measured
    position alone, on the reasoning that a tool reaching the point with a
    different twist is another configuration rather than a failure. That is
    wrong: where the position is reachable and the orientation is not, the
    least-squares solve returns a compromise that lands on the point pointing
    the wrong way, and reporting it as a way to reach the goal is a lie the
    user would only catch by looking at the robot.
    """
    actual = chain.forward(np.asarray(q))
    position = float(np.linalg.norm(actual[:3, 3] - goal[:3, 3]))
    # Angle of the residual rotation, via its trace. Clipped because a rotation
    # matrix that has drifted a hair outside SO(3) would otherwise put arccos
    # outside its domain and return nan, which compares false against every
    # tolerance and would silently keep the solution.
    relative = actual[:3, :3].T @ goal[:3, :3]
    cosine = (float(np.trace(relative)) - 1.0) / 2.0
    orientation = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return position, orientation


def collect(
    chain: Chain,
    goal: np.ndarray,
    solved,
    tolerance: float = REACH_TOLERANCE,
    orientation_tolerance: float = ORIENTATION_TOLERANCE,
):
    """Fold an iterable of candidate configurations into distinct solutions.

    ``solved`` yields configurations already put through a solver; this decides
    which of them count. Kept separate from the solving so the choosing can be
    tested without one.

    Ordered by how close each came, so the list reads best-first and the tie
    between two branches that both reached the goal is broken by something
    rather than by seed order.
    """
    solutions: list[Solution] = []
    for q in solved:
        q = np.asarray(q, dtype=float)
        position, orientation = reach_error(chain, q, goal)
        if position > tolerance or orientation > orientation_tolerance:
            continue
        if not is_distinct(q, [s.q for s in solutions], is_revolute=chain.is_revolute):
            continue
        solutions.append(Solution(q=q, position_error=position))
    return sorted(solutions, key=lambda s: s.position_error)
