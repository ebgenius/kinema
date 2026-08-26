"""Common types for Kinema's IK backends.

Two backends implement the same tiny interface:

* :mod:`.numpy_backend` -- damped least squares. Always available, needs
  nothing but NumPy and the rig itself, and is what answers while JAX is still
  importing or when JAX is unavailable.
* :mod:`.pyroki_backend` -- PyRoki's nonlinear least squares. Handles joint
  limits as real constraints and behaves far better near singularities, which
  is the whole reason Kinema bundles it.

Keeping the interface this small is what makes the fallback honest: the panel
can switch backends mid-session and nothing else in the add-on has to care.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SolveResult:
    """Outcome of one IK solve."""

    q: np.ndarray
    converged: bool
    position_error: float
    rotation_error: float
    iterations: int
    backend: str

    @property
    def summary(self) -> str:
        return (
            f"{self.backend}: {'converged' if self.converged else 'no solution'} "
            f"({self.position_error * 1000:.2f} mm, "
            f"{np.degrees(self.rotation_error):.2f}°, {self.iterations} it)"
        )


#: Convergence thresholds. A tenth of a millimetre and a hundredth of a degree
#: are already well below what a render can show; tighter just burns iterations.
POSITION_TOLERANCE = 1e-4
ROTATION_TOLERANCE = 1e-4


class SolverError(RuntimeError):
    """A backend could not be prepared (missing dependency, bad rig)."""
