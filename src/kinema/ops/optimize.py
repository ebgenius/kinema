"""Optimize Motion: the generated job solved again as one trajectory within its limits (#45).

Under Generate Motion in the Waypoints panel. It plays the job as Generate
Motion keyed it, solves every frame of it at once (``solver/trajectory.py``),
and keys the result: every free joint on every frame, with live IK keyed off.
A free joint is one on the chain to the TCP that isn't held. Then it checks
the job again, so the Motion Check shows what now plays.

**What it keeps.** The tool at each waypoint, in the configuration taught
there, and on each linear move's line. The first and last frames exactly, with
the robot at rest before and after, as playback has it. Held joints, and
joints the tool doesn't hang from, such as a gripper's, stay as the job has
them.

**What it changes.** The rest, within each joint's range, top speed and
acceleration limit, as smoothly as those leave room for. Not a linear move's
timing: the line's targets are where Generate Motion put the tool on each
frame, so it still runs along the line at a steady speed, and the joint moves
either side ease into it and out of it. A move that can't
keep within its limits in the frames it has comes out as close as it gets,
and the Motion Check says about how many frames it needs.

**Regenerating replaces it.** The keys are in the range Generate Motion owns,
so the next Generate Motion clears them as it clears its own. It optimises the
job only while that still follows the waypoints: after an edit, generate it
again first.

**Blender stays live.** From the panel it runs modal: the solve goes to JAX,
which works on threads of its own, and a timer looks for the result. Esc
abandons it, writing nothing. The first solve for a robot and a job length
compiles first, and that does hold Blender up: seconds, and longer on a slow
machine with nothing in the compile cache. Run from a script, the operator
waits for the result instead.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import bpy
import numpy as np
from bpy.types import Operator

from ..rig import builder, motion_check
from ..rig.waypoints import MOVE_LINEAR, WaypointError, spans
from ..solver import manager, trajectory
from ..solver.base import SolverError
from ..ui.panel import active_rig
from . import waypoints as wp
from .ik import (
    _key_switch_off,
    ik_switch_is_animated,
    key_joint_value,
    own_fcurve_containers,
    set_joint_value,
)
from .velocity import acceleration_limit_of, limit_of

#: Seconds between looks at whether a dispatched solve has finished.
POLL = 0.05

#: Rigs being optimised, by name. The panel shows them as busy, and they can't
#: be optimised twice at once.
_running: set[str] = set()


def optimizing(rig) -> bool:
    return rig is not None and rig.name in _running


class Refused(Exception):
    """Why a job can't be optimised, in words for the user."""


@dataclass
class Job:
    """One Optimize Motion, from reading the job to writing the result."""

    rig_name: str
    identity: object
    #: The waypoints as they were read, to check they haven't changed since.
    signature: str
    #: The robot as it was read, likewise: see :func:`model_signature`.
    model: tuple
    first: int
    last: int
    problem: trajectory.Problem
    robot: object
    optimizer: trajectory.Optimizer
    #: (joint bone name, column in PyRoki's actuated vector) for every joint
    #: the solve may move: the result is keyed on these and no others.
    free: list[tuple[str, int]]
    #: Moves the Motion Check flagged before, or None if it wasn't current.
    flagged_before: int | None
    pending: trajectory.Pending | None = None
    #: Set once keys start being written: from then on, a failure still has to
    #: end in an undo step, which only a finished operator gets.
    writing: bool = False
    compile_seconds: float = 0.0
    solve_started: float = 0.0


# --------------------------------------------------------------------------
# reading the job
# --------------------------------------------------------------------------
def prepare(context, rig) -> Job:
    """Everything needed to optimise ``rig``'s job, read off it. Raises :class:`Refused`.

    Plays the job once, as the Motion Check does, for where it starts from.
    Writes nothing.
    """
    try:
        plan = spans(rig.kinema_waypoints)
    except WaypointError as exc:
        raise Refused(str(exc)) from exc
    signature = wp.job_signature(rig)
    if not rig.kinema_generated_job:
        raise Refused("Generate Motion first: there's no job to optimize yet")
    if rig.kinema_generated_job != signature:
        raise Refused(
            "The waypoints have changed since the job was generated: Generate "
            "Motion again, then optimize"
        )
    solver, pyroki = _model(rig)
    first, last = plan[0].start_frame, plan[-1].end_frame
    try:
        optimizer = trajectory.optimizer(
            manager.rig_identity(rig), rig.name, solver.link_target[0], last - first + 1
        )
    except SolverError as exc:
        raise Refused(str(exc)) from exc

    samples = wp.play_job(context, rig, plan)
    if samples is None:
        raise Refused("This rig has no TCP to aim the tool by")
    problem, free = assemble(rig, plan, samples, solver, pyroki, 1.0 / wp.scene_fps(context.scene))

    flagged_before = None
    if len(rig.kinema_motion_check) and rig.kinema_motion_check_job == signature:
        flagged_before = sum(1 for check in rig.kinema_motion_check if motion_check.problems(check))
    return Job(
        rig_name=rig.name,
        identity=manager.rig_identity(rig),
        signature=signature,
        model=model_signature(rig),
        first=first,
        last=last,
        problem=problem,
        robot=pyroki.robot,
        optimizer=optimizer,
        free=free,
        flagged_before=flagged_before,
    )


#: The bone properties a joint's limits are read from.
_LIMIT_PROPS = (
    builder.PROP_LOWER, builder.PROP_UPPER, builder.PROP_VELOCITY, builder.PROP_ACCELERATION
)


def model_signature(rig) -> tuple:
    """What a solve is set up from besides the waypoints, to tell whether it still holds.

    The joints, which of them are held, their limits, and the TCP; and where
    the bones the solve doesn't move are: each held joint, and Root, which
    places the whole robot. Blender stays live while a solve runs, so any of
    them can change before its result lands, and keying that result would then
    write a motion solved for another robot, or for one standing elsewhere.
    """
    bones = builder.joint_bones(rig)
    joints = tuple(
        (
            pose_bone.name,
            bool(getattr(pose_bone, "kinema_ik_hold", False)),
            tuple(_number(pose_bone.bone.get(key)) for key in _LIMIT_PROPS),
        )
        for pose_bone in bones
    )
    unmoved = [pose_bone for pose_bone in bones if getattr(pose_bone, "kinema_ik_hold", False)]
    root = rig.pose.bones.get(builder.ROOT_BONE)
    if root is not None:
        unmoved.append(root)
    placed = tuple((pose_bone.name, _placement(rig, pose_bone)) for pose_bone in unmoved)
    tcp = rig.data.bones.get(wp._tool_bone(rig))
    if tcp is None:
        return joints, placed, None
    parent = tcp.parent.name if tcp.parent is not None else ""
    pose = tuple(_number(value) for row in tcp.matrix_local for value in row)
    return joints, placed, (tcp.name, parent, pose)


def _placement(rig, pose_bone) -> tuple:
    """Where a bone the solve doesn't move is, as the solve read it.

    Its keys, where it has them: playback moves it on every frame, so its pose
    changes with the current frame while nothing the solve read did. Otherwise
    its pose, which playback leaves where it is.
    """
    prefix = pose_bone.path_from_id() + "."
    keys = tuple(
        (curve.data_path, curve.array_index, curve.mute)
        + tuple(
            _number(value)
            for point in curve.keyframe_points
            for vector in (point.co, point.handle_left, point.handle_right)
            for value in vector
        )
        for container in own_fcurve_containers(rig)
        for curve in container
        if curve.data_path.startswith(prefix)
    )
    if keys:
        return keys
    return tuple(_number(value) for row in pose_bone.matrix_basis for value in row)


def _number(value) -> float | None:
    return None if value is None else round(float(value), 9)


def _model(rig):
    """The rig's chain to its TCP, and the PyRoki model for it. Raises :class:`Refused`.

    Aimed at the TCP whatever IK's tip is: the waypoints are poses of the tool.
    """
    solver = manager.build_solver(rig, wp.ik_bone_name(rig), wp._tool_bone(rig))
    if solver is None or solver.link_target is None:
        raise Refused("This rig has no chain from its base to its TCP to optimize")
    pyroki = solver.pyroki(rig)
    if pyroki is None:
        reason = solver.pyroki_error or "the PyRoki solver isn't available"
        raise Refused(f"Optimize Motion needs the PyRoki solver: {reason}")
    return solver, pyroki


def assemble(rig, plan, samples, solver, pyroki, dt) -> tuple[trajectory.Problem, list]:
    """The trajectory problem for ``plan``, starting from the job as ``samples`` played it.

    Returns the problem and the joints it may move, as :attr:`Job.free`.
    """
    first, last = plan[0].start_frame, plan[-1].end_frame
    frames = range(first, last + 1)
    played = {sample.frame: sample.q for sample in samples}
    bones = builder.joint_bones(rig)
    column = {name: index for index, name in enumerate(pyroki.actuated_names)}
    columns = [column.get(pose_bone.name) for pose_bone in bones]
    dof = len(column)

    def full(values) -> np.ndarray:
        """Values in the rig's joint order, spread over PyRoki's actuated vector."""
        out = np.zeros(dof)
        for value, index in zip(values, columns, strict=False):
            if index is not None:
                out[index] = value
        return out

    q_job = np.array([full(played[frame]) for frame in frames])

    # The arm's own joints move; held ones, and any the tool doesn't hang from,
    # stay where the job has them.
    held = dict(zip(solver.chain.bone_names, manager.held_mask(rig, solver.chain), strict=True))
    free = [
        (name, column[name])
        for name in solver.chain.bone_names
        if name in column and not held[name]
    ]
    moving = np.zeros(dof, dtype=bool)
    moving[[index for _, index in free]] = True

    prior = q_job.copy()
    prior_weight = np.where(moving, trajectory.JOB_WEIGHT, trajectory.PIN_WEIGHT)
    prior_weight = np.repeat(prior_weight[None], len(frames), axis=0)
    taught = {}
    for span in plan:
        for waypoint in (span.start, span.end):
            taught[int(waypoint.frame)] = waypoint
    for frame, waypoint in taught.items():
        if waypoint.dof != len(bones):
            continue  # taught with other joints: its values don't line up
        row = frame - first
        prior[row, moving] = full(list(waypoint.q)[: waypoint.dof])[moving]
        prior_weight[row, moving] = trajectory.WAYPOINT_WEIGHT
    # The job's own ends, exactly: the frames either side play them.
    prior[0], prior[-1] = q_job[0], q_job[-1]
    prior_weight[0] = prior_weight[-1] = trajectory.PIN_WEIGHT

    # The tool: at every waypoint, and on every frame of a linear move.
    to_solver = np.linalg.inv(manager.root_pose(rig))
    _, correction = solver.link_target
    targets = np.repeat(np.eye(4)[None], len(frames), axis=0)
    weight = np.zeros(len(frames))

    def aim(frame, pose) -> None:
        targets[frame - first] = to_solver @ np.array(pose, dtype=float) @ correction
        weight[frame - first] = 1.0

    for span in plan:
        start, end = wp.pose_of(span.start), wp.pose_of(span.end)
        aim(span.start_frame, start)
        aim(span.end_frame, end)
        if span.move == MOVE_LINEAR:
            # The tool's line, carried to the link afterwards: the link's own
            # origin doesn't travel in a straight line when the tool does.
            for frame in range(span.start_frame + 1, span.end_frame):
                fraction = (frame - span.start_frame) / span.frames
                aim(frame, wp._between(start, end, fraction))

    # The limits the Motion Check measures against: the bones'.
    speed = np.full(dof, trajectory.UNLIMITED)
    acceleration = np.full(dof, trajectory.UNLIMITED)
    for pose_bone, index in zip(bones, columns, strict=True):
        if index is None:
            continue
        for limits, limit in (
            (speed, limit_of(pose_bone)),
            (acceleration, acceleration_limit_of(pose_bone)),
        ):
            if limit is not None:
                limits[index] = limit

    problem = trajectory.Problem(
        q_init=q_job,
        targets=targets,
        position_weight=weight,
        orientation_weight=weight.copy(),
        prior=prior,
        prior_weight=prior_weight,
        speed_limit=speed,
        acceleration_limit=acceleration,
        dt=float(dt),
        link_index=int(pyroki.target_link_index),
    )
    return problem, free


# --------------------------------------------------------------------------
# solving and writing
# --------------------------------------------------------------------------
def dispatch(job: Job) -> None:
    """Hand the solve to JAX. Compiles first if this kernel never has, which waits."""
    compiled = job.optimizer.compiled
    started = time.perf_counter()
    try:
        job.pending = job.optimizer.dispatch(job.robot, job.problem)
    except SolverError as exc:
        raise Refused(f"Optimize Motion failed: {exc}") from exc
    if not compiled:
        job.compile_seconds = time.perf_counter() - started
    job.solve_started = time.perf_counter()


def finish(context, job: Job) -> tuple[bool, str, str]:
    """Key the solved job on its rig and check it.

    Returns whether anything was written, and a report: its kind and message.
    """
    rig = bpy.data.objects.get(job.rig_name)
    if (
        rig is None
        or not builder.is_kinema_rig(rig)
        or manager.rig_identity(rig) != job.identity
        or wp.job_signature(rig) != job.signature
        or model_signature(rig) != job.model
    ):
        return (
            False, "WARNING",
            "The job or the robot changed while it was being optimized, so nothing was written",
        )
    q, iterations = job.pending.result()
    seconds = time.perf_counter() - job.solve_started
    frames = range(job.first, job.last + 1)
    job.writing = True
    write_keys(context, rig, frames, q, job.free)

    plan = spans(rig.kinema_waypoints)
    samples = wp.play_job(context, rig, plan)
    checks = wp.check_job(context, rig, plan, samples=samples)
    if checks is not None:
        joints, fps = wp.check_joints(rig), wp.scene_fps(context.scene)
        for index, check in enumerate(checks):
            check.frames_needed = motion_check.frames_needed(
                check, samples, joints, fps, index == 0, index == len(checks) - 1
            )
    flagged = wp.store_check(rig, checks)

    moves = f"{len(plan)} move{'s' if len(plan) != 1 else ''}"
    message = f"Optimized {moves} over frames {job.first}-{job.last}"
    detail = f"{iterations} iterations, {seconds:.1f} s"
    if job.compile_seconds:
        detail += f", after compiling for {job.compile_seconds:.0f} s"
    message += f" ({detail})"
    if flagged:
        message += f"; {flagged} still over a limit, in the Motion Check"
    else:
        message += "; within every limit"
    if job.flagged_before:
        message += f" (was {job.flagged_before} flagged)"
    return True, ("WARNING" if flagged else "INFO"), message


def write_keys(context, rig, frames, q, free) -> None:
    """Key the free joints on every frame of the job, and live IK off over it.

    Their keys in the range go first, so nothing of the job before is left
    between them. Held joints and joints off the chain keep their keys.
    """
    first, last = frames[0], frames[-1]
    bones = [(rig.pose.bones[name], index) for name, index in free]
    paths = {
        (pose_bone.path_from_id(_channel(pose_bone)), 1): None for pose_bone, _ in bones
    }
    from .. import handlers

    original = context.scene.frame_current
    with handlers.suspended():
        for container in own_fcurve_containers(rig):
            for curve in list(container):
                if (curve.data_path, curve.array_index) not in paths:
                    continue
                for point in reversed(list(curve.keyframe_points)):
                    if first <= point.co[0] <= last:
                        curve.keyframe_points.remove(point)
        for frame, row in zip(frames, q, strict=True):
            for pose_bone, index in bones:
                set_joint_value(pose_bone, row[index])
                key_joint_value(pose_bone, frame)
        _key_switch_off(rig, first, last)
        if not ik_switch_is_animated(rig):
            # Nothing keys the switch, so it stays as set: off.
            rig.kinema_ik_enabled = False
    context.scene.frame_set(original)


def _channel(pose_bone) -> str:
    prismatic = pose_bone.bone.get(builder.PROP_JOINT_TYPE, "revolute") == "prismatic"
    return "location" if prismatic else "rotation_euler"


# --------------------------------------------------------------------------
# the operator
# --------------------------------------------------------------------------
class KINEMA_OT_optimize_motion(Operator):
    bl_idname = "kinema.optimize_motion"
    bl_label = "Optimize Motion"
    bl_description = (
        "Solve the generated job again as one smooth motion within every joint's "
        "range, speed and acceleration limits, keeping the waypoints and the lines. "
        "Keys every free joint, on the chain to the TCP and not held, on every frame"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        rig = active_rig(context)
        return rig is not None and len(rig.kinema_waypoints) > 1 and not optimizing(rig)

    def execute(self, context: bpy.types.Context) -> set[str]:
        """From a script: solve, and wait for the result."""
        rig = active_rig(context)
        try:
            job = prepare(context, rig)
            dispatch(job)
        except Refused as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return self._finish(context, job)

    def invoke(self, context: bpy.types.Context, event) -> set[str]:
        """From the panel: solve while Blender stays live, and key the result when it lands."""
        rig = active_rig(context)
        try:
            self._job = prepare(context, rig)
        except Refused as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        _running.add(self._job.rig_name)
        window_manager = context.window_manager
        self._timer = window_manager.event_timer_add(POLL, window=context.window)
        window_manager.modal_handler_add(self)
        # Shown before the compile, which holds Blender up, so it says why.
        _status(
            context,
            "Optimizing motion... Esc to cancel"
            if self._job.optimizer.compiled
            else "Compiling the optimizer for this robot and job length...",
        )
        return {"RUNNING_MODAL"}

    def modal(self, context: bpy.types.Context, event) -> set[str]:
        job = self._job
        if event.type == "ESC" and event.value == "PRESS":
            self._end(context)
            self.report({"INFO"}, "Optimize Motion cancelled; nothing was written")
            return {"CANCELLED"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        # Whatever goes wrong, the operator ends: left running, it would keep the
        # rig marked as optimizing, and the button gone, until the file reloads.
        if job.pending is None:
            try:
                dispatch(job)
            except Exception as exc:  # noqa: BLE001 - a JAX error, as well as a refusal
                self._end(context)
                failed = str(exc) if isinstance(exc, Refused) else f"Optimize Motion failed: {exc}"
                self.report({"ERROR"}, failed)
                return {"CANCELLED"}
            _status(context, "Optimizing motion... Esc to cancel")
            return {"RUNNING_MODAL"}
        if not job.pending.ready():
            return {"PASS_THROUGH"}
        self._end(context)
        return self._finish(context, job)

    def _finish(self, context, job: Job) -> set[str]:
        """Key the result, ending in an undo step if anything was written.

        Blender only makes an undo step for an operator that finishes. One that
        fails after it has started writing keys still finishes, so Ctrl+Z puts
        the job back, rather than cancelling with half of it written.
        """
        try:
            written, kind, message = finish(context, job)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Optimize Motion failed: {exc}")
            return {"FINISHED"} if job.writing else {"CANCELLED"}
        self.report({kind}, message)
        return {"FINISHED"} if written else {"CANCELLED"}

    def cancel(self, context: bpy.types.Context) -> None:
        """Blender ending the operator itself, as when its window closes."""
        self._end(context)

    def _end(self, context) -> None:
        if getattr(self, "_timer", None) is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        _running.discard(self._job.rig_name)
        _status(context, None)
        _redraw(context)


def _status(context, text) -> None:
    workspace = getattr(context, "workspace", None)
    if workspace is not None:
        workspace.status_text_set(text)


def _redraw(context) -> None:
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def forget() -> None:
    """Nothing is being optimised in a new file."""
    _running.clear()


classes = (KINEMA_OT_optimize_motion,)
