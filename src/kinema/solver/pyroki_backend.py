"""PyRoki-backed IK: the reason Kinema bundles a 118 MB dependency.

Blender's own IK and the damped-least-squares fallback both treat inverse
kinematics as "invert the Jacobian and hope". PyRoki poses it as a nonlinear
least-squares problem, which buys three things that matter for animating real
robots:

* **Joint limits are constraints, not clamps.** A damped-least-squares solver
  presses a joint against its stop and truncates the step; PyRoki finds a
  different arm configuration that satisfies the limits instead.
* **Singularities are penalised, not stumbled into.** The optional
  manipulability cost uses the Yoshikawa index, so the solver actively avoids
  the degenerate configurations where a wrist flips through half a turn to
  achieve a millimetre of tool motion.
* **Continuous joints keep their turns.** Seeding from the current pose means
  successive viewport solves stay on the same IK branch instead of jumping.

Everything here is lazy: importing JAX costs seconds, and the first solve pays
a JIT compile. Both are deliberately kept off Blender's startup path -- see
:mod:`kinema.runtime`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import POSITION_TOLERANCE, ROTATION_TOLERANCE, SolverError, SolveResult
from .chain import pose_error

NAME = "PyRoki"

#: Cost weights. Position is weighted above orientation because a tool that is
#: in the right place with a slightly wrong twist reads as correct, while the
#: reverse does not.
POSITION_WEIGHT = 50.0
ORIENTATION_WEIGHT = 10.0
#: Bias toward the seed, keeping successive viewport solves on the same IK
#: branch. Tuned empirically on UR5e over 20 random reachable targets: 0.001
#: holds 20/20 convergence at 0.0001 mm, while 0.01 starts to fight the pose
#: cost and pushes the 90th-percentile error up an order of magnitude.
REST_WEIGHT = 0.001
#: Manipulability weight. The Yoshikawa residual is 1/(manipulability + 1e-6),
#: which is *large* in ordinary configurations -- easily 10-20 before scaling.
#: It therefore needs a far smaller weight than the pose cost, not a comparable
#: one: at 0.02 it swamps the pose term entirely (14/20 converged, 0.32 mm
#: median), while 1e-4 keeps 20/20 at 0.0001 mm and still steers away from
#: singular configurations. Costs roughly 2.5x the solve time.
MANIPULABILITY_WEIGHT = 1e-4


@dataclass
class PyrokiSolver:
    """A compiled IK problem for one robot and one target link."""

    robot: object
    actuated_names: tuple[str, ...]
    target_link_index: int
    target_link_name: str
    #: Cached jitted solve, keyed by whether manipulability is enabled.
    _compiled: dict = field(default_factory=dict)

    @property
    def dof(self) -> int:
        return len(self.actuated_names)

    def _solver(self, avoid_singularities: bool):
        key = bool(avoid_singularities)
        if key in self._compiled:
            return self._compiled[key]

        from .. import runtime

        stack = runtime.load_solver_stack()
        if stack is None:
            raise SolverError(runtime.solver_error() or "JAX stack unavailable")

        jnp, jdc, jaxlie, jaxls, pk = (
            stack["jnp"], stack["jdc"], stack["jaxlie"], stack["jaxls"], stack["pyroki"]
        )

        @jdc.jit
        def _solve(robot, target_wxyz, target_position, link_index, seed):
            joint_var = robot.joint_var_cls(0)
            costs = [
                pk.costs.pose_cost(
                    robot,
                    joint_var,
                    jaxlie.SE3.from_rotation_and_translation(
                        jaxlie.SO3(target_wxyz), target_position
                    ),
                    link_index,
                    pos_weight=POSITION_WEIGHT,
                    ori_weight=ORIENTATION_WEIGHT,
                ),
                # A constraint, not a penalty: this is the headline difference
                # from the NumPy fallback.
                pk.costs.limit_constraint(robot, joint_var),
                # Keeps successive viewport solves on the same IK branch.
                pk.costs.rest_cost(joint_var, seed, REST_WEIGHT),
            ]
            if avoid_singularities:
                costs.append(
                    pk.costs.manipulability_cost(
                        robot, joint_var, link_index, MANIPULABILITY_WEIGHT
                    )
                )

            problem = jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
            solution = problem.analyze().solve(
                initial_vals=jaxls.VarValues.make([joint_var.with_value(seed)]),
                verbose=False,
                linear_solver="dense_cholesky",
                trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
            )
            return solution[joint_var]

        self._compiled[key] = (_solve, jnp)
        return self._compiled[key]

    def solve(
        self,
        q_seed: np.ndarray,
        target: np.ndarray,
        *,
        avoid_singularities: bool = True,
    ) -> np.ndarray:
        """Solve for the full actuated vector. ``target`` is a 4x4 in base frame."""
        solve_fn, jnp = self._solver(avoid_singularities)

        rotation = np.asarray(target[:3, :3], dtype=np.float64)
        quaternion = _matrix_to_wxyz(rotation)

        result = solve_fn(
            self.robot,
            jnp.array(quaternion, dtype=jnp.float32),
            jnp.array(target[:3, 3], dtype=jnp.float32),
            jnp.array(self.target_link_index, dtype=jnp.int32),
            jnp.array(q_seed, dtype=jnp.float32),
        )
        return np.asarray(result, dtype=np.float64)

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """Target-link pose for ``q``, as a 4x4 -- used to measure residuals."""
        from .. import runtime

        stack = runtime.load_solver_stack()
        poses = np.asarray(self.robot.forward_kinematics(stack["jnp"].array(q, dtype="float32")))
        return _wxyz_xyz_to_matrix(poses[self.target_link_index])

    def warm_up(self) -> None:
        """Trigger JIT compilation ahead of the first interactive solve."""
        identity = np.eye(4)
        identity[:3, 3] = (0.3, 0.0, 0.3)
        self.solve(np.zeros(self.dof), identity)


# --------------------------------------------------------------------------
# small conversions (kept in NumPy so they work before JAX is loaded)
# --------------------------------------------------------------------------
def _matrix_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Rotation matrix to a (w, x, y, z) quaternion, jaxlie's convention."""
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        # Pick the largest diagonal element for numerical stability.
        i = int(np.argmax(np.diag(rotation)))
        j, k = (i + 1) % 3, (i + 2) % 3
        scale = np.sqrt(1.0 + rotation[i, i] - rotation[j, j] - rotation[k, k]) * 2.0
        w = (rotation[k, j] - rotation[j, k]) / scale
        components = [0.0, 0.0, 0.0]
        components[i] = 0.25 * scale
        components[j] = (rotation[j, i] + rotation[i, j]) / scale
        components[k] = (rotation[k, i] + rotation[i, k]) / scale
        x, y, z = components
    quaternion = np.array([w, x, y, z], dtype=np.float64)
    return quaternion / (np.linalg.norm(quaternion) or 1.0)


def _wxyz_xyz_to_matrix(wxyz_xyz: np.ndarray) -> np.ndarray:
    """jaxlie's packed (w,x,y,z,x,y,z) SE3 representation to a 4x4."""
    w, x, y, z = wxyz_xyz[:4]
    matrix = np.eye(4)
    matrix[:3, :3] = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    matrix[:3, 3] = wxyz_xyz[4:7]
    return matrix


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------
def build(urdf, target_link_name: str) -> PyrokiSolver:
    """Compile a solver for ``urdf`` targeting ``target_link_name``."""
    from .. import runtime

    stack = runtime.load_solver_stack()
    if stack is None:
        raise SolverError(runtime.solver_error() or "JAX stack unavailable")

    robot = stack["pyroki"].Robot.from_urdf(urdf)
    names = tuple(robot.links.names)
    if target_link_name not in names:
        raise SolverError(
            f"link '{target_link_name}' is not in the robot ({len(names)} links)"
        )

    return PyrokiSolver(
        robot=robot,
        actuated_names=tuple(robot.joints.actuated_names),
        target_link_index=names.index(target_link_name),
        target_link_name=target_link_name,
    )


def measure(solver: PyrokiSolver, q: np.ndarray, target: np.ndarray) -> SolveResult:
    """Wrap a solved configuration in a :class:`SolveResult` with real errors."""
    error = pose_error(solver.forward_kinematics(q), target)
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
        iterations=0,  # jaxls runs a fixed internal schedule
        backend=NAME,
    )
