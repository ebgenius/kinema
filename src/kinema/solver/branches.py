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

#: How far apart two configurations must be to count as different, in radians
#: on the joint that differs most. A tenth of a radian is a little under six
#: degrees: far wider than solver noise between two runs onto the same branch,
#: far narrower than the gap between genuinely different ones, which is
#: typically most of a half-turn.
DISTINCT_TOLERANCE = 0.1

#: How close the tool has to land for a solution to count as having reached the
#: goal, in metres. Generous next to the sub-micron a converged PyRoki solve
#: actually manages -- this is here to throw out the seeds that did not
#: converge at all, not to grade the ones that did.
REACH_TOLERANCE = 5e-4

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


def wrapped_distance(a: np.ndarray, b: np.ndarray) -> float:
    """The largest per-joint angle between two configurations, in radians.

    Wrapped, because a joint at +pi and the same joint at -pi are the same
    place. Continuous joints reach the same pose at any number of whole turns,
    and without wrapping every one of those would be reported as a different
    configuration.
    """
    return float(np.max(np.abs(np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b)))))))


def is_distinct(q: np.ndarray, found, tolerance: float = DISTINCT_TOLERANCE) -> bool:
    """True if ``q`` is not already in ``found``."""
    return all(wrapped_distance(q, other) >= tolerance for other in found)


def reach_error(chain: Chain, q: np.ndarray, goal: np.ndarray) -> float:
    """How far the tool lands from ``goal`` for configuration ``q``, in metres.

    Measured off the chain rather than taken from the solver's own report, so a
    solution is checked against the rig the user is looking at. Position only:
    a solution that reaches the point with the tool twisted is a different
    configuration, not a failed one, and the pose cost has already weighted
    orientation during the solve.
    """
    return float(np.linalg.norm(chain.forward(np.asarray(q))[:3, 3] - goal[:3, 3]))


def collect(chain: Chain, goal: np.ndarray, solved, tolerance: float = REACH_TOLERANCE):
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
        error = reach_error(chain, q, goal)
        if error > tolerance:
            continue
        if not is_distinct(q, [s.q for s in solutions]):
            continue
        solutions.append(Solution(q=q, position_error=error))
    return sorted(solutions, key=lambda s: s.position_error)
