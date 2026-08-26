"""Damped least squares IK -- the always-available fallback.

This backend exists so Kinema is never dead in the water. It needs only NumPy
and the rig itself, so it answers while JAX is still importing in the
background, on a machine where JAX failed to load, and whenever the user picks
"NumPy" explicitly for a lighter, more predictable solve.

The method is Levenberg-Marquardt on the pose error, better known in robotics
as damped least squares:

    dq = Jᵀ (J Jᵀ + λ² I)⁻¹ e

The damping term λ is what keeps this usable near singularities. An undamped
pseudo-inverse asks for enormous joint velocities as the Jacobian loses rank --
the classic wrist-flip where a robot spins wildly for a tiny tool movement.
Damping trades a little tracking accuracy for a bounded, smooth solution, and
λ is raised adaptively when a step fails to improve.

It is genuinely worse than PyRoki: joint limits are enforced by clamping rather
than as constraints, so a solution pressed against a limit is simply truncated
instead of the solver finding a different arm configuration that respects it.
"""

from __future__ import annotations

import numpy as np

from .base import POSITION_TOLERANCE, ROTATION_TOLERANCE, SolveResult
from .chain import Chain, pose_error

NAME = "NumPy"

#: Starting damping. Small enough not to slow ordinary tracking, large enough
#: to stay stable through a singularity.
LAMBDA_INITIAL = 0.05
LAMBDA_MAX = 10.0
#: Largest joint step per iteration (radians / metres). Prevents a single
#: iteration from throwing the arm across its workspace.
MAX_STEP = 0.4


def solve(
    chain: Chain,
    q_seed: np.ndarray,
    target: np.ndarray,
    *,
    max_iterations: int = 60,
    rest_pose: np.ndarray | None = None,
    rest_weight: float = 0.0,
) -> SolveResult:
    """Solve for a configuration putting the tool at ``target``.

    Args:
        chain: the kinematic chain to solve.
        q_seed: starting configuration -- normally the current pose, which is
            what makes successive viewport solves continuous rather than
            jumping between IK branches.
        target: desired 4x4 tool pose, in armature space.
        rest_pose / rest_weight: optional null-space bias pulling unused
            degrees of freedom toward a preferred posture.
    """
    q = chain.clamp(np.asarray(q_seed, dtype=float).copy())
    damping = LAMBDA_INITIAL
    identity = np.eye(6)

    current = chain.forward(q)
    error = pose_error(current, target)
    best_norm = float(np.linalg.norm(error))

    iterations = 0
    for step_index in range(1, max_iterations + 1):
        iterations = step_index
        position_error = float(np.linalg.norm(error[:3]))
        rotation_error = float(np.linalg.norm(error[3:]))
        if position_error < POSITION_TOLERANCE and rotation_error < ROTATION_TOLERANCE:
            break

        jacobian = chain.jacobian(q)
        # Solve (J Jᵀ + λ²I) y = e, then dq = Jᵀ y. Going through the 6x6
        # system is cheaper than the dof x dof one for any real robot arm.
        gram = jacobian @ jacobian.T + (damping**2) * identity
        try:
            y = np.linalg.solve(gram, error)
        except np.linalg.LinAlgError:
            damping = min(damping * 4.0, LAMBDA_MAX)
            continue
        step = jacobian.T @ y

        largest = float(np.max(np.abs(step))) if step.size else 0.0
        if largest > MAX_STEP:
            step *= MAX_STEP / largest

        candidate = chain.clamp(q + step)

        if rest_pose is not None and rest_weight > 0.0:
            # Null-space bias: move toward the rest pose only in directions
            # that do not disturb the tool, using the damped projector.
            null = np.eye(chain.dof) - jacobian.T @ np.linalg.solve(gram, jacobian)
            candidate = chain.clamp(
                candidate + rest_weight * (null @ (rest_pose - candidate))
            )

        candidate_error = pose_error(chain.forward(candidate), target)
        candidate_norm = float(np.linalg.norm(candidate_error))

        if candidate_norm < best_norm:
            # Step helped: accept it and trust the linearisation a bit more.
            q, error, best_norm = candidate, candidate_error, candidate_norm
            damping = max(damping * 0.7, LAMBDA_INITIAL * 0.1)
        else:
            # Step hurt: damp harder and try again from the same configuration.
            damping = min(damping * 2.5, LAMBDA_MAX)
            if damping >= LAMBDA_MAX:
                break

    position_error = float(np.linalg.norm(error[:3]))
    rotation_error = float(np.linalg.norm(error[3:]))
    return SolveResult(
        q=q,
        converged=(
            position_error < POSITION_TOLERANCE * 10
            and rotation_error < ROTATION_TOLERANCE * 10
        ),
        position_error=position_error,
        rotation_error=rotation_error,
        iterations=iterations,
        backend=NAME,
    )
