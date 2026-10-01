"""Trajectory optimisation: every frame of a job solved at once, within the robot's limits.

Generate Motion keys a job the way a robot program reads: joint moves
interpolate the joints, linear moves drive the tool along a line. Nothing in
that knows how fast a joint may turn or how hard it may speed up; the Motion
Check is where that shows. **Optimize Motion** takes the job as it plays and
solves all of its frames as one least-squares problem over the joint values:

- the tool at each waypoint, in the configuration it was taught in;
- on a linear move, the tool where the line puts it on every frame;
- every joint inside its range;
- every joint under its top speed, and its acceleration limit where it has one;
- as little acceleration and jerk as that leaves room for;
- otherwise, close to the job it started from.

It runs on PyRoki's solver, jaxls, over the PyRoki model the rig's IK already
uses, limits included (``manager.bone_limits``). Deliberately free of ``bpy``:
the job is read off the rig in ``ops/optimize.py``.

**Every limit is a cost, not a constraint.** jaxls meets a constraint by
raising a penalty on it pass after pass, and two things go wrong with that here:
- A job that can't keep within its limits in the frames it has is the common
  case for this tool, and there the penalty never stops rising. In float32 the
  cost soon grows past telling a better step from a worse one. On a job asking
  1.9 times a joint's top speed, every step was rejected and the job came back
  as it went in. Asked of one move in three, the waypoints ended 24 mm out.
- A constrained solve ends only once its constraints have settled, never on
  finding no better step, so one already finished ran on to the iteration cap:
  100 iterations, where the same job with its ranges as costs stopped at 11.

As costs:
- a joint's speed and acceleration cost :data:`LIMIT_WEIGHT` per whole limit
  past it, measured as the Motion Check measures them, and aim
  :data:`LIMIT_MARGIN` inside the limit. They land within it where the frames
  allow: 97.4% of a top speed, on a job that needed a joint near it;
- its range costs :data:`RANGE_WEIGHT` per radian or metre past it;
- where the frames don't allow, the job comes out over its limits by as little
  as the waypoints leave room for, with the waypoints within hundredths of a
  millimetre, and the Motion Check says by how much.

**One compile per robot and length.** A job is padded to the next of
:data:`BUCKETS`, with the padding held still, so jobs of 90 and 121 frames share
a kernel, and an edit to the waypoints only changes the arrays. A kernel holds
100-200 MB, so only :data:`KEEP` is kept.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace

import numpy as np

from .base import SolverError

#: Lengths a job is padded to, in frames. One compiled kernel per length used.
BUCKETS = (32, 64, 128, 256, 512, 1024, 2048)
#: Frames the robot stands still before the job and after it. The acceleration
#: at a frame is measured from the two either side, so the job's own first and
#: last frames are measured against a robot at rest, as playback shows it.
REST_FRAMES = 2

#: The tool's pose at a waypoint and along a line: per metre and per radian.
#: Stiff, since the limit costs pull against them wherever the frames are too
#: few: at 1000 per metre a waypoint gave 0.17 mm there, at 10000 0.002 mm.
POSITION_WEIGHT = 10000.0
ORIENTATION_WEIGHT = 2000.0
#: Smoothness, on acceleration (rad/s²) and jerk (rad/s³), as the spike tuned
#: them: enough to smooth what the limits leave free.
ACCELERATION_SMOOTHING = 0.02
JERK_SMOOTHING = 0.002
#: How hard each frame is pulled toward the job it started from, toward the
#: configuration taught at a waypoint, and to where a pinned joint is.
JOB_WEIGHT = 0.01
WAYPOINT_WEIGHT = 10.0
PIN_WEIGHT = 1000.0

#: What a joint's speed or acceleration costs past its limit, per whole limit
#: past it. Aimed 2% inside, on a job that needed a joint near its top speed:
#: at 1 it ran the joint at 100.0% of it, and to the iteration cap; at 3,
#: 98.4%, in 12 iterations.
LIMIT_WEIGHT = 3.0
#: How far inside each speed and acceleration limit the costs aim, as a
#: fraction of the limit.
LIMIT_MARGIN = 0.03
#: A limit for a joint that has none: never reached.
UNLIMITED = 1.0e4
#: What a joint costs past its range, per radian or metre: a millimetre or a
#: milliradian past costs what a tenth of a millimetre off a waypoint does.
RANGE_WEIGHT = 1000.0
#: Iterations at most. A job that has not settled by then is returned as it is.
MAX_ITERATIONS = 100
#: Compiled kernels kept, the most recently used.
KEEP = 1


def bucket(frames: int) -> int:
    """The padded length for a problem of ``frames`` frames."""
    for size in BUCKETS:
        if frames <= size:
            return size
    raise SolverError(
        f"a job of {frames} frames is too long to optimise; the most is "
        f"{BUCKETS[-1] - 2 * REST_FRAMES}"
    )


@dataclass(frozen=True)
class Problem:
    """A job to optimise, as arrays over its frames and PyRoki's actuated joints.

    ``targets`` are poses of the link PyRoki aims at, in the frame the solver
    works in; a frame with no target has weight 0 and any pose.
    """

    #: (frames, dof): the job as it plays, where the solve starts.
    q_init: np.ndarray
    #: (frames, 4, 4)
    targets: np.ndarray
    #: (frames,)
    position_weight: np.ndarray
    #: (frames,)
    orientation_weight: np.ndarray
    #: (frames, dof): what each joint is pulled toward, and how hard.
    prior: np.ndarray
    prior_weight: np.ndarray
    #: (dof,), rad/s or m/s, and rad/s² or m/s²; :data:`UNLIMITED` where none.
    speed_limit: np.ndarray
    acceleration_limit: np.ndarray
    #: Seconds per frame.
    dt: float
    link_index: int

    @property
    def frames(self) -> int:
        return int(len(self.q_init))


def at_rest(problem: Problem) -> tuple[Problem, slice]:
    """``problem`` with the robot still before and after it, padded to its bucket.

    Returns the padded problem and where the job's own frames sit in it. The
    frames added hold the job's first and last values, pinned, and ask nothing
    of the tool.
    """
    frames = problem.frames
    size = bucket(frames + 2 * REST_FRAMES)
    before, after = REST_FRAMES, size - frames - REST_FRAMES

    def held(array):
        return np.concatenate(
            [np.repeat(array[:1], before, axis=0), array, np.repeat(array[-1:], after, axis=0)]
        )

    def zero(array):
        return np.concatenate([np.zeros(before), array, np.zeros(after)])

    dof = problem.q_init.shape[1]
    weight = np.concatenate(
        [
            np.full((before, dof), PIN_WEIGHT),
            problem.prior_weight,
            np.full((after, dof), PIN_WEIGHT),
        ]
    )
    padded = replace(
        problem,
        q_init=held(problem.q_init),
        targets=held(problem.targets),
        position_weight=zero(problem.position_weight),
        orientation_weight=zero(problem.orientation_weight),
        prior=held(problem.prior),
        prior_weight=weight,
    )
    return padded, slice(before, before + frames)


# --------------------------------------------------------------------------
# the compiled solve
# --------------------------------------------------------------------------
@dataclass
class Pending:
    """A solve dispatched to JAX and not necessarily finished. Never blocks to ask."""

    q: object
    iterations: object
    own: slice

    def ready(self) -> bool:
        is_ready = getattr(self.q, "is_ready", None)
        return bool(is_ready()) if is_ready is not None else True

    def result(self) -> tuple[np.ndarray, int]:
        """The job's own frames, (frames, dof), and the iterations taken. Waits if needed."""
        q = np.asarray(self.q, dtype=np.float64)[self.own]
        return q, int(np.asarray(self.iterations))


class Optimizer:
    """A compiled trajectory solve for one robot model and one padded length."""

    def __init__(self, size: int, verbose: bool = False) -> None:
        self.size = size
        #: jaxls's own log of every iteration, for diagnosing a solve. Read when
        #: the kernel is traced.
        self.verbose = verbose
        #: Set once the first solve has been dispatched, which is what compiles.
        self.compiled = False
        self._fn = None

    def _kernel(self):
        if self._fn is not None:
            return self._fn
        from .. import runtime

        stack = runtime.load_solver_stack()
        if stack is None:
            raise SolverError(runtime.solver_error() or "JAX stack unavailable")
        jax, jnp, jdc, jaxlie, jaxls, pk = (
            stack["jax"], stack["jnp"], stack["jdc"], stack["jaxlie"],
            stack["jaxls"], stack["pyroki"],
        )
        speed_cost, acceleration_cost = _limit_costs(jaxls)

        @jdc.jit
        def solve(robot, q_init, targets, position_weight, orientation_weight,
                  prior, prior_weight, speed_limit, acceleration_limit, dt, link_index):
            frames = q_init.shape[0]
            joint = robot.joint_var_cls
            ids = np.arange(frames)
            a, j = ids[2:-2], ids[3:-3]
            # A leading axis of one: the robot is shared by every frame's cost.
            batched = jax.tree.map(lambda x: x[None] if hasattr(x, "shape") else x, robot)
            # Every limit is a cost, not a constraint: see the module's docstring.
            costs = [
                pk.costs.limit_cost(batched, joint(ids), RANGE_WEIGHT),
                speed_cost(
                    joint(ids[1:]), joint(ids[:-1]), dt, speed_limit[None], LIMIT_WEIGHT
                ),
                acceleration_cost(
                    joint(ids[:-2]), joint(ids[1:-1]), joint(ids[2:]), dt,
                    acceleration_limit[None], LIMIT_WEIGHT,
                ),
                pk.costs.five_point_acceleration_cost(
                    joint(a), joint(a + 2), joint(a + 1), joint(a - 1), joint(a - 2), dt,
                    ACCELERATION_SMOOTHING,
                ),
                pk.costs.five_point_jerk_cost(
                    joint(j + 3), joint(j + 2), joint(j + 1), joint(j - 1), joint(j - 2),
                    joint(j - 3), dt, JERK_SMOOTHING,
                ),
                pk.costs.rest_cost(joint(ids), prior, prior_weight),
                pk.costs.pose_cost(
                    batched, joint(ids), jaxlie.SE3(targets), link_index[None],
                    position_weight, orientation_weight,
                ),
            ]
            problem = jaxls.LeastSquaresProblem(costs=costs, variables=[joint(ids)])
            solution, summary = problem.analyze().solve(
                initial_vals=jaxls.VarValues.make([joint(ids).with_value(q_init)]),
                linear_solver="conjugate_gradient",
                termination=jaxls.TerminationConfig(max_iterations=MAX_ITERATIONS),
                verbose=self.verbose,
                return_summary=True,
            )
            return solution[joint(ids)], summary.iterations

        self._fn = (solve, jnp, jaxlie)
        return self._fn

    def dispatch(self, robot, problem: Problem) -> Pending:
        """Start solving ``problem``, padded, and return without waiting for it.

        The first call compiles, which does wait: seconds, or longer on a slow
        machine with nothing in the compile cache.
        """
        padded, own = at_rest(problem)
        if padded.frames != self.size:
            raise SolverError(f"a {padded.frames}-frame problem for a {self.size}-frame kernel")
        solve, jnp, jaxlie = self._kernel()
        targets = np.asarray(
            jaxlie.SE3.from_matrix(jnp.asarray(padded.targets, dtype=jnp.float32)).wxyz_xyz
        )
        q, iterations = solve(
            robot,
            jnp.asarray(padded.q_init, dtype=jnp.float32),
            jnp.asarray(targets, dtype=jnp.float32),
            jnp.asarray(padded.position_weight * POSITION_WEIGHT, dtype=jnp.float32),
            jnp.asarray(padded.orientation_weight * ORIENTATION_WEIGHT, dtype=jnp.float32),
            jnp.asarray(padded.prior, dtype=jnp.float32),
            jnp.asarray(padded.prior_weight, dtype=jnp.float32),
            jnp.asarray(padded.speed_limit * (1.0 - LIMIT_MARGIN), dtype=jnp.float32),
            jnp.asarray(padded.acceleration_limit * (1.0 - LIMIT_MARGIN), dtype=jnp.float32),
            jnp.asarray(padded.dt, dtype=jnp.float32),
            jnp.asarray(padded.link_index, dtype=jnp.int32),
        )
        self.compiled = True
        return Pending(q=q, iterations=iterations, own=own)


def speed_over(vals, var, previous, dt, limit, weight):
    """How far past its limit each joint's speed is, as a fraction of the limit.

    From the frame before, as the Motion Check measures a speed, so what is
    kept within here is what the check passes.
    """
    from jax import numpy as jnp

    speed = jnp.abs(vals[var] - vals[previous]) / dt
    return (jnp.maximum(0.0, speed / limit - 1.0) * weight).flatten()


def acceleration_over(vals, before, now, after, dt, limit, weight):
    """How far past its limit each joint's acceleration is, as a fraction of the limit.

    From the frames either side, as the Motion Check measures one. PyRoki's own
    residual reaches two frames each way, and a path within it can still fail
    the check.
    """
    from jax import numpy as jnp

    acceleration = jnp.abs(vals[before] - 2.0 * vals[now] + vals[after]) / dt**2
    return (jnp.maximum(0.0, acceleration / limit - 1.0) * weight).flatten()


_costs: dict = {}


def _limit_costs(jaxls):
    """The two limit costs, made once: jaxls orders a problem's costs by their functions."""
    if not _costs:
        _costs["speed"] = jaxls.Cost.factory(speed_over)
        _costs["acceleration"] = jaxls.Cost.factory(acceleration_over)
    return _costs["speed"], _costs["acceleration"]


# --------------------------------------------------------------------------
# the kernels kept
# --------------------------------------------------------------------------
#: (rig identity, link, padded length) -> (optimizer, rig name). The name is kept
#: so a rig rebuilt under it can have its kernels dropped by name.
_optimizers: OrderedDict = OrderedDict()


def optimizer(identity, rig_name: str, link: str, frames: int) -> Optimizer:
    """The optimizer for a job of ``frames`` frames on this rig's model, made if needed."""
    size = bucket(frames + 2 * REST_FRAMES)
    key = (identity, str(link), size)
    entry = _optimizers.get(key)
    if entry is not None:
        _optimizers.move_to_end(key)
        return entry[0]
    made = Optimizer(size)
    _optimizers[key] = (made, rig_name)
    while len(_optimizers) > KEEP:
        _optimizers.popitem(last=False)
    return made


def compiled_for(identity, link: str, frames: int) -> bool:
    """Whether a compiled kernel is kept for a job of ``frames`` frames on this model."""
    try:
        size = bucket(frames + 2 * REST_FRAMES)
    except SolverError:
        return False
    entry = _optimizers.get((identity, str(link), size))
    return entry is not None and entry[0].compiled


def forget(rig_name: str | None = None) -> None:
    """Drop the kept kernels: all of them, or those made for ``rig_name``."""
    if rig_name is None:
        _optimizers.clear()
        return
    for key, (_, name) in list(_optimizers.items()):
        if name == rig_name:
            del _optimizers[key]
