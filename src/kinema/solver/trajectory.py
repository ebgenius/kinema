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
- every capsule of the robot clear of every obstacle, given both;
- as little acceleration and jerk as that leaves room for;
- otherwise, close to the job it started from.

**Clearance is a cost too**, a hinge like the limits': nothing where a
capsule keeps far enough away, and a pull out where it doesn't. It measures
what the Motion Check measures, exactly, each capsule's segment against each
box, sphere or floor (:func:`segment_clearance`), placed by PyRoki's
kinematics on every frame. The pull is weak, :data:`COLLISION_WEIGHT`, and
levels off past :data:`SATURATION` deep, because it is only ever wanted where
nothing else is:
- Where there is room to go round, nothing much holds the arm where it was,
  and even a weak pull takes it all the way clear: on arm6, out of a post
  72 mm into the forearm to 12 mm clear, within every limit.
- Where there isn't, because a waypoint, a line or the job's pinned ends run
  into an obstacle, a strong pull wrecks the job instead of clearing it. At
  the weight of a waypoint, a post across a KR210's line moved the waypoints
  89 mm and took a joint to 136 times its acceleration limit. A weak one
  leaves the job close to what it was, the waypoints within 0.1 mm, and the
  Motion Check names the hit.
- Between the two, where going round would take more than a move's frames
  allow, it gets as close to clear as the limits leave room for.
- **A hit it can't avoid makes it hurry.** The cost is summed frame by
  frame, so where a capsule can't get clear, fewer frames near the obstacle
  are all that's left to lower it. The solver changes the move's timing: it
  winds back, crosses at full speed and passes the next waypoint before coming
  back. More frames don't help. On a KR120 at 24 fps, a joint move swinging
  past a 1.2 m cube: the arm went over the cube, but the turret's one capsule,
  then fitted as a cylinder with its ends rounded, reached 27 mm into it
  whenever the base faced it, while the turret's meshes stood 337 mm clear, and
  the base can't swing the other way round. At 100, 150, 200 and 300 frames the
  base still wound back by up to 0.43 rad and crossed at its top speed. Fitted
  as a capsule (``rig/collision.py``), the turret stands 227 mm clear. With
  Keep Clear By at 10, 20, 30 and 50 mm, on that one scene, the same move came
  out 12, 23, 31 and 52 mm clear within every limit, with neither the wind-up
  nor the overshoot. The base still holds back, then crosses faster than
  taught: at 50 mm it peaks at 0.36 of its speed limit, against 0.29 as taught.
  Where a capsule can't get clear at all, the hurry is all there: on arm6, a
  lug on joint 1 swept through a post winds back 0.28 rad and overshoots 0.07
  (issue #98). A start or an end pinned inside an obstacle does the same.
  Weighting the depth by how far a capsule moves that frame stopped the
  wind-up, but left the arm 350-410 mm in, or still crossing at twice the
  taught speed (issue #93).
Only with capsules and obstacles both: a kernel without them has no clearance
cost.

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

**Each step is solved closely enough to reach across a move.** Conjugate
gradient takes each step inexactly, stopping by default once it has cut its
residual to 1% of where it started. The pose costs dominate that residual, so
it stops once the waypoints are met. What eases a joint move into a linear
one is spread thinly over every frame of the move, and is all still in the
last 1%. So it was never taken: the steps shrank until jaxls took the solve
as finished, with the joint over its limit where the move meets the line. On
a KR210's 400-frame job at 24 fps, joint 3 was left at 3.3 times its
acceleration limit, a move with 150 frames was said to need 244, and the
waypoints were 0.2 mm out. At :data:`STEP_TOLERANCE` every move was within its
limits, at 26-78% of one, with the waypoints within 0.002 mm, in 3.2 s where
the inexact steps took 0.3.

**A solve that stops short is solved again.** The limit and clearance costs
are hinges, nothing on one side of a corner and a slope on the other. jaxls
predicts a step's gain from the slope where it stands, so a step that takes a
frame across a corner gains less than predicted, or costs, and is thrown away.
After enough of those its damping is at its ceiling and the steps it takes are
tiny, and it ends on the size of a step, the iteration cap, or no step left to
take, with the cost still falling. Two things answer that:
- each hinge's corner is rounded (:func:`hinge`): over :data:`CLEARANCE_BAND`
  short of a clearance, :data:`LIMIT_BAND` of a limit past it;
- a solve that ended on neither jaxls's cost rule nor its gradient rule, and
  cut the cost by more than :data:`RESTART_GAIN`, is solved again from where
  it ended, its damping reset, up to :data:`RESTARTS` times.

Each alone left jobs inside an obstacle. On a KR120 job past a 1.2 m cube,
Keep Clear By 10, 20, 30 and 50 mm came out 62 mm in, 22 mm clear, 34 mm in
and 52 mm clear in one solve, and 12, 23, 31 and 52 mm clear with both, the
solve taking 4-8 s where it took 2-4. Rounded but solved once, 20 mm
ended 45 mm in; solved again with plain corners, the arm6 crate in
``tests/integration/test_optimize.py`` ended 63 mm in. ``Pending.passes``
reads back how each pass ended: jaxls's own iteration count is one per
damping it tries, and its cost history holds the steps it threw away.

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
#: How far conjugate gradient cuts its residual for each step, at most, against
#: jaxls's 1e-2: see the module's docstring. Measured on arm6, with 150-frame
#: moves at 24 fps either side of a linear one: a joint was left at 2.6 times
#: its acceleration limit at 1e-2, 1.4 at 1e-3 and 1.6 at 1e-4. At 1e-5 every
#: move was within its limits, as an exact (dense) step has them; 1e-6 only
#: took three times as long.
STEP_TOLERANCE = 1e-5
#: Compiled kernels kept, the most recently used.
KEEP = 1
#: How many times a solve that stopped short is solved again from where it
#: stopped, at most, and the fraction of its cost the last pass has to have
#: cut to earn another. See the module's docstring.
RESTARTS = 4
RESTART_GAIN = 0.01
#: How far past a limit, as a fraction of it, and how far short of a
#: clearance, in metres, a hinge's corner is rounded over: see :func:`hinge`.
LIMIT_BAND = 0.01
CLEARANCE_BAND = 0.002

#: What coming closer to an obstacle than the robot is to keep costs, per metre
#: closer, before it levels off: a tenth of a whole limit over per 10 mm. See
#: the module's docstring. Measured on arm6 with a post 72 mm into the
#: forearm: at 30 and 300 it went round within every limit; at 10000, the
#: weight of a waypoint, it went round over them, and at 100 it stuck at
#: 62 mm in.
COLLISION_WEIGHT = 30.0
#: How much further than asked the collision cost aims, in metres, so the
#: robot lands at least as far away as asked where it can.
CLEARANCE_MARGIN = 0.002
#: Golden-section steps for where along a capsule an obstacle is closest:
#: each keeps 0.618 of the interval, so 24 take a 2 m capsule's to 0.02 mm.
SEARCH_STEPS = 24
#: Metres short past which the collision cost levels off, growing as the
#: logarithm of the depth: an obstacle half a metre into an arm that can't
#: leave it pulls little harder than one 20 mm in.
SATURATION = 0.02
#: Capsules and obstacles are padded to these counts, so a kernel serves a
#: scene until it outgrows one.
CAPSULE_BUCKETS = (8, 16, 32, 64, 128, 256)
OBSTACLE_BUCKETS = (4, 8, 16, 32, 64, 128, 256)
#: Obstacle shapes, by the names ``rig/collision.py`` gives them.
SHAPES = ("BOX", "SPHERE", "FLOOR")


def bucket(frames: int) -> int:
    """The padded length for a problem of ``frames`` frames, its rest frames included."""
    for size in BUCKETS:
        if frames <= size:
            return size
    # The user's job is what they know the length of: the rest frames are ours.
    raise SolverError(
        f"a job of {frames - 2 * REST_FRAMES} frames is too long to optimise; the most is "
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
    #: The robot's capsules and the obstacles to keep them clear of, or None
    #: to solve without: see :class:`Collision`.
    collision: Collision | None = None

    @property
    def frames(self) -> int:
        return int(len(self.q_init))


@dataclass(frozen=True)
class Collision:
    """The robot's capsules, the obstacles on each frame, and how far to keep between.

    Capsules ride on PyRoki's links: each is a segment in its link's frame,
    and the solver places it with the link on every frame. Obstacles are where
    the scene had them on each frame, in world space, as the Motion Check
    measures them (``rig/collision.py``).
    """

    #: (C,) the link each capsule rides on, as PyRoki numbers its links.
    link: np.ndarray
    #: (C, 3) each capsule's ends in its link's frame.
    a: np.ndarray
    b: np.ndarray
    #: (C,) radii in world space: the robot's scale taken in.
    radius: np.ndarray
    #: (frames, 4, 4) world space from the solver's frame on each frame: the
    #: robot's object and Root, which may be carried.
    placement: np.ndarray
    #: (O,) each obstacle's shape: :data:`SHAPES` numbers them.
    kind: np.ndarray
    #: (frames, O, 4, 4) each obstacle's frame in world space, rotation and
    #: centre: unscaled, as ``rig/collision.py``'s ``Obstacle.matrix``.
    frame: np.ndarray
    #: (frames, O, 3) a box's half-extents, or a sphere's radius first.
    size: np.ndarray
    #: (frames,) 1 on frames the robot has to keep clear on, 0 elsewhere.
    weight: np.ndarray
    #: Metres the robot is to keep from every obstacle.
    keep_clear: float

    @property
    def shape(self) -> tuple[int, int]:
        """Capsules and obstacles, padded: what a kernel is compiled for."""
        return padded(len(self.link), CAPSULE_BUCKETS), padded(len(self.kind), OBSTACLE_BUCKETS)


def padded(count: int, buckets: tuple) -> int:
    """``count`` capsules or obstacles, rounded up to the next of ``buckets``."""
    for size in buckets:
        if count <= size:
            return size
    raise SolverError(f"{count} is too many to keep clear: the most is {buckets[-1]}")


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
    collision = problem.collision
    if collision is not None:
        # Obstacles stand where they did on the job's ends; nothing is asked
        # of the robot there, pinned as it is.
        collision = replace(
            collision,
            placement=held(collision.placement),
            frame=held(collision.frame),
            size=held(collision.size),
            weight=zero(collision.weight),
        )
    padded_problem = replace(
        problem,
        q_init=held(problem.q_init),
        targets=held(problem.targets),
        position_weight=zero(problem.position_weight),
        orientation_weight=zero(problem.orientation_weight),
        prior=held(problem.prior),
        prior_weight=weight,
        collision=collision,
    )
    return padded_problem, slice(before, before + frames)


# --------------------------------------------------------------------------
# the compiled solve
# --------------------------------------------------------------------------
@dataclass
class Pending:
    """A solve dispatched to JAX and not necessarily finished. Never blocks to ask."""

    q: object
    iterations: object
    own: slice
    #: The kernel's account of the solve, for :meth:`passes`: the cost it
    #: started from, how many passes it took, and per pass the cost it ended
    #: on, which of jaxls's stopping rules fired, and its iterations.
    why: tuple | None = None

    def ready(self) -> bool:
        is_ready = getattr(self.q, "is_ready", None)
        return bool(is_ready()) if is_ready is not None else True

    def result(self) -> tuple[np.ndarray, int]:
        """The job's own frames, (frames, dof), and the iterations taken. Waits if needed.

        jaxls counts an iteration per damping it tries, taken or not: a count,
        not a measure of progress.
        """
        q = np.asarray(self.q, dtype=np.float64)[self.own]
        return q, int(np.asarray(self.iterations))

    def passes(self) -> list[Pass]:
        """Each pass of the solve, in order. Waits if needed."""
        if self.why is None:
            return []
        start, solves, costs, criteria, iterations = (np.asarray(x) for x in self.why)
        found, before = [], float(start)
        for index in range(int(solves)):
            cost_ended, gradient_ended, step_ended = (bool(x) for x in criteria[index])
            found.append(
                Pass(
                    before=before,
                    after=float(costs[index]),
                    iterations=int(iterations[index]),
                    settled=cost_ended or gradient_ended,
                    stopped_on_step=step_ended,
                )
            )
            before = min(before, float(costs[index]))
        return found


@dataclass(frozen=True)
class Pass:
    """One of jaxls's solves within a solve: see :data:`RESTARTS`."""

    #: The cost it started from and ended on.
    before: float
    after: float
    #: jaxls's iterations: one per damping tried.
    iterations: int
    #: Whether it ended on jaxls's cost or gradient rule: nothing left to gain.
    settled: bool
    #: Whether it ended on the size of a step, which a stall also ends on.
    stopped_on_step: bool


class Optimizer:
    """A compiled trajectory solve for one robot model and one padded length."""

    def __init__(self, size: int, shape: tuple[int, int] | None = None,
                 verbose: bool = False) -> None:
        self.size = size
        #: Capsules and obstacles, padded, or None for a kernel that keeps
        #: clear of nothing.
        self.shape = shape
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
        collision_cost = _collision_cost(jaxls)

        @jdc.jit
        def solve(robot, q_init, targets, position_weight, orientation_weight,
                  prior, prior_weight, speed_limit, acceleration_limit, dt, link_index,
                  collision):
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
            if collision is not None:
                # Per frame: the capsules and the robot's placement are shared.
                shared = {key: collision[key][None] for key in _SHARED}
                own = {key: collision[key] for key in _PER_FRAME}
                costs.append(collision_cost(batched, joint(ids), shared, own))
            problem = jaxls.LeastSquaresProblem(costs=costs, variables=[joint(ids)]).analyze()

            def cost_at(q):
                values = jaxls.VarValues.make([joint(ids).with_value(q)])
                return jnp.sum(problem.compute_residual_vector(values) ** 2)

            # Solved again from where it stopped while it stopped short: see
            # the module's docstring.
            def again(state):
                _, _, solves, gained, settled, _ = state
                return (solves == 0) | ((solves <= RESTARTS) & gained & ~settled)

            def once(state):
                q, cost, solves, _, _, why = state
                solution, summary = problem.solve(
                    initial_vals=jaxls.VarValues.make([joint(ids).with_value(q)]),
                    linear_solver=jaxls.ConjugateGradientConfig(tolerance_max=STEP_TOLERANCE),
                    termination=jaxls.TerminationConfig(max_iterations=MAX_ITERATIONS),
                    verbose=self.verbose,
                    return_summary=True,
                )
                found = solution[joint(ids)]
                # The cost where it ended, not jaxls's history of it, which
                # holds the steps it proposed and threw away too.
                reached = cost_at(found)
                cost_ended, gradient_ended, _ = summary.termination_criteria
                better = reached < cost
                why = (
                    why[0].at[solves].set(reached),
                    why[1].at[solves].set(summary.termination_criteria),
                    why[2].at[solves].set(summary.iterations),
                )
                return (
                    jnp.where(better, found, q),
                    jnp.minimum(reached, cost),
                    solves + 1,
                    reached < cost * (1.0 - RESTART_GAIN),
                    cost_ended | gradient_ended,
                    why,
                )

            start = cost_at(q_init)
            passes = RESTARTS + 1
            q, _, solves, _, _, why = jax.lax.while_loop(
                again,
                once,
                (
                    q_init, start, jnp.array(0), jnp.array(True), jnp.array(False),
                    (
                        jnp.full(passes, jnp.nan),
                        jnp.zeros((passes, 3), dtype=bool),
                        jnp.zeros(passes, dtype=jnp.int32),
                    ),
                ),
            )
            return q, jnp.sum(why[2]), (start, solves, *why)

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
        if (padded.collision.shape if padded.collision is not None else None) != self.shape:
            raise SolverError("the capsules or obstacles don't match this kernel")
        solve, jnp, jaxlie = self._kernel()
        targets = np.asarray(
            jaxlie.SE3.from_matrix(jnp.asarray(padded.targets, dtype=jnp.float32)).wxyz_xyz
        )
        q, iterations, why = solve(
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
            _collision_arrays(jnp, padded.collision),
        )
        self.compiled = True
        return Pending(q=q, iterations=iterations, own=own, why=why)


def hinge(jnp, over, band: float):
    """``over`` where it is positive, nothing where it isn't, the corner rounded over ``band``.

    A plain hinge's slope jumps from nothing to one at zero. jaxls steps by
    the slope where it stands, so a step that takes a frame across the corner
    costs what the step never saw coming, and is thrown away. Rounded, the
    slope rises from nothing to one across ``band``: a parabola up to it, then
    the line, half a band lower.
    """
    inside = jnp.clip(over, 0.0, band)
    return jnp.where(over < band, inside * inside / (2.0 * band), over - band / 2.0)


def speed_over(vals, var, previous, dt, limit, weight):
    """How far past its limit each joint's speed is, as a fraction of the limit.

    From the frame before, as the Motion Check measures a speed, so what is
    kept within here is what the check passes. Its corner is rounded over
    :data:`LIMIT_BAND`.
    """
    from jax import numpy as jnp

    speed = jnp.abs(vals[var] - vals[previous]) / dt
    return (hinge(jnp, speed / limit - 1.0, LIMIT_BAND) * weight).flatten()


def acceleration_over(vals, before, now, after, dt, limit, weight):
    """How far past its limit each joint's acceleration is, as a fraction of the limit.

    From the frames either side, as the Motion Check measures one. PyRoki's own
    residual reaches two frames each way, and a path within it can still fail
    the check. Its corner is rounded over :data:`LIMIT_BAND`.
    """
    from jax import numpy as jnp

    acceleration = jnp.abs(vals[before] - 2.0 * vals[now] + vals[after]) / dt**2
    return (hinge(jnp, acceleration / limit - 1.0, LIMIT_BAND) * weight).flatten()


#: Collision arrays every frame shares, and those with a value per frame.
_SHARED = ("link", "a", "b", "radius", "capsule_mask", "kind", "obstacle_mask", "aim")
_PER_FRAME = ("placement", "rotation", "centre", "size", "weight")


def _collision_arrays(jnp, collision: Collision | None) -> dict | None:
    """``collision`` padded to its kernel's shape, as the arrays the cost reads."""
    if collision is None:
        return None
    capsules, obstacles = collision.shape
    count, kinds = len(collision.link), len(collision.kind)

    def pad(array, size, axis=0):
        widths = [(0, 0)] * array.ndim
        widths[axis] = (0, size - array.shape[axis])
        return np.pad(array, widths)

    frame = np.asarray(collision.frame, dtype=float)
    floats = {
        "a": pad(np.asarray(collision.a, dtype=float), capsules),
        "b": pad(np.asarray(collision.b, dtype=float), capsules),
        "radius": pad(np.asarray(collision.radius, dtype=float), capsules),
        "capsule_mask": pad(np.ones(count), capsules),
        "placement": np.asarray(collision.placement, dtype=float),
        "obstacle_mask": pad(np.ones(kinds), obstacles),
        "aim": np.asarray(collision.keep_clear + CLEARANCE_MARGIN),
        "rotation": pad(frame[..., :3, :3], obstacles, axis=1),
        "centre": pad(frame[..., :3, 3], obstacles, axis=1),
        "size": pad(np.asarray(collision.size, dtype=float), obstacles, axis=1),
        "weight": np.asarray(collision.weight, dtype=float) * COLLISION_WEIGHT,
    }
    arrays = {key: jnp.asarray(value, dtype=jnp.float32) for key, value in floats.items()}
    arrays["link"] = jnp.asarray(pad(np.asarray(collision.link), capsules), dtype=jnp.int32)
    arrays["kind"] = jnp.asarray(pad(np.asarray(collision.kind), obstacles), dtype=jnp.int32)
    return arrays


def signed_distance(jnp, points, kind, rotation, centre, size):
    """Signed distance from ``points`` to obstacles, broadcast together: negative inside.

    Exact for each shape, as ``rig/collision.py`` measures it, and smooth
    enough to take a gradient of anywhere: no square root of zero.
    """
    local = jnp.einsum("...i,...ij->...j", points - centre, rotation)
    q = jnp.abs(local) - size
    box = _norm(jnp, jnp.maximum(q, 0.0)) + jnp.minimum(jnp.max(q, axis=-1), 0.0)
    sphere = _norm(jnp, local) - size[..., 0]
    floor = local[..., 2]
    return jnp.where(kind == 0, box, jnp.where(kind == 1, sphere, floor))


def _norm(jnp, vector):
    return jnp.sqrt(jnp.sum(vector * vector, axis=-1) + 1e-12)


def segment_clearance(jax, jnp, a, b, kind, rotation, centre, size):
    """Signed distance from each segment ``a``-``b`` to each obstacle.

    ``a`` and ``b`` are (C, 3); the obstacles' arrays have O leading. Returns
    (C, O). Every shape's distance is convex along a line, so a golden-section
    search finds where along the segment it is least, and the distance is
    taken there. Where that is comes of comparisons, which carry no gradient,
    and none is wanted: the closest point moving with the segment is a
    second-order change, so the gradient is the distance's own at that point.
    The search reads the segment with its gradient stopped all the same, which
    spares the solver differentiating every step of it.
    """
    a, b = a[:, None, :], b[:, None, :]
    kind, rotation, centre, size = kind[None], rotation[None], centre[None], size[None]
    fixed_a, fixed_b = jax.lax.stop_gradient(a), jax.lax.stop_gradient(b)

    def at(t, start, end):
        return signed_distance(
            jnp, start + t[..., None] * (end - start), kind, rotation, centre, size
        )

    ratio = (5.0**0.5 - 1.0) / 2.0
    shape = jnp.broadcast_shapes(a.shape[:-1], kind.shape)
    low, high = jnp.zeros(shape), jnp.ones(shape)
    left, right = high - ratio * (high - low), low + ratio * (high - low)

    def step(_, state):
        low, high, left, right, f_left, f_right = state
        # The least is left of `right` where f_left is the lower, else right of `left`.
        lower = f_left < f_right
        low = jnp.where(lower, low, left)
        high = jnp.where(lower, right, high)
        kept, f_kept = jnp.where(lower, left, right), jnp.where(lower, f_left, f_right)
        new = jnp.where(lower, high - ratio * (high - low), low + ratio * (high - low))
        f_new = at(new, fixed_a, fixed_b)
        left, f_left = jnp.where(lower, new, kept), jnp.where(lower, f_new, f_kept)
        right, f_right = jnp.where(lower, kept, new), jnp.where(lower, f_kept, f_new)
        return low, high, left, right, f_left, f_right

    state = (low, high, left, right, at(left, fixed_a, fixed_b), at(right, fixed_a, fixed_b))
    _, _, left, right, f_left, f_right = jax.lax.fori_loop(0, SEARCH_STEPS, step, state)
    # Where the least is at an end, the search closes in on it as on any other
    # point: within 0.618 ** 25 of the segment's length, 6 µm on a metre.
    return at(jnp.where(f_left < f_right, left, right), a, b)


def clearance_short(vals, robot, var, shared, own):
    """How much closer than it should be each capsule is to each obstacle, weighted.

    Zero where it keeps ``aim`` away: the robot is free to go anywhere else.
    """
    import jax
    import jaxlie
    from jax import numpy as jnp

    poses = jaxlie.SE3(robot.forward_kinematics(vals[var])[shared["link"]])
    placement = own["placement"]

    def world(points):
        return poses.apply(points) @ placement[:3, :3].T + placement[:3, 3]

    distance = segment_clearance(
        jax, jnp, world(shared["a"]), world(shared["b"]), shared["kind"],
        own["rotation"], own["centre"], own["size"],
    )
    short = hinge(jnp, shared["aim"] - (distance - shared["radius"][:, None]), CLEARANCE_BAND)
    # The same as the depth, near the surface; its logarithm, deep in.
    short = SATURATION * jnp.log1p(short / SATURATION)
    mask = shared["capsule_mask"][:, None] * shared["obstacle_mask"][None, :]
    return (short * mask * own["weight"]).flatten()


_costs: dict = {}


def _limit_costs(jaxls):
    """The two limit costs, made once: jaxls orders a problem's costs by their functions.

    Made again for another jaxls: a cost made by one module would be foreign to
    a problem built by the other. Disabling Kinema and enabling it again keeps
    the module it had, but nothing makes that a promise.
    """
    if _costs.get("jaxls") is not jaxls:
        _costs.clear()
        _costs["jaxls"] = jaxls
        _costs["speed"] = jaxls.Cost.factory(speed_over)
        _costs["acceleration"] = jaxls.Cost.factory(acceleration_over)
    return _costs["speed"], _costs["acceleration"]


def _collision_cost(jaxls):
    """The collision cost, made once per jaxls, as the limit costs are."""
    _limit_costs(jaxls)
    if "collision" not in _costs:
        _costs["collision"] = jaxls.Cost.factory(clearance_short)
    return _costs["collision"]


# --------------------------------------------------------------------------
# the kernels kept
# --------------------------------------------------------------------------
#: (rig identity, link, padded length) -> (optimizer, rig name). The name is kept
#: so a rig rebuilt under it can have its kernels dropped by name.
_optimizers: OrderedDict = OrderedDict()


def optimizer(identity, rig_name: str, link: str, frames: int,
              shape: tuple[int, int] | None = None) -> Optimizer:
    """The optimizer for a job of ``frames`` frames on this rig's model, made if needed.

    ``shape`` is the capsules and obstacles it keeps clear between, padded
    (:attr:`Collision.shape`), or None for none.
    """
    size = bucket(frames + 2 * REST_FRAMES)
    key = (identity, str(link), size, shape)
    entry = _optimizers.get(key)
    if entry is not None:
        _optimizers.move_to_end(key)
        return entry[0]
    made = Optimizer(size, shape)
    _optimizers[key] = (made, rig_name)
    while len(_optimizers) > KEEP:
        _optimizers.popitem(last=False)
    return made


def compiled_for(identity, link: str, frames: int,
                 shape: tuple[int, int] | None = None) -> bool:
    """Whether a compiled kernel is kept for a job of ``frames`` frames on this model."""
    try:
        size = bucket(frames + 2 * REST_FRAMES)
    except SolverError:
        return False
    entry = _optimizers.get((identity, str(link), size, shape))
    return entry is not None and entry[0].compiled


def forget(rig_name: str | None = None) -> None:
    """Drop the kept kernels: all of them, or those made for ``rig_name``.

    All of them takes the limit costs they were made from too.
    """
    if rig_name is None:
        _optimizers.clear()
        _costs.clear()
        return
    for key, (_, name) in list(_optimizers.items()):
        if name == rig_name:
            del _optimizers[key]
