"""Put off PyRoki's compile until an edit is done, and solve on NumPy meanwhile.

Some edits change what PyRoki has to be compiled for -- an external axis changes
the model, a TCP moved onto another joint changes the link it aims at -- and a
compile is tens of seconds. Every change in the Adjust Last Operation panel runs
the operator again, so paying it inside the operator paid it on every nudge of a
field. Instead the operator defers the rig: live IK solves it on NumPy, which needs
no compile, and once the edit has settled the compile is paid once, behind a wait
cursor, without touching the pose.

Only live viewport solves are deferred. A bake, Find Solutions, playback or a
render builds PyRoki when it needs it: those mean the edit is done, and solving
them on NumPy would quietly give a different answer. See
``solver.manager.RigSolver.solve``.

An edit has settled when its operator is no longer the last registered one -- its
redo panel has given way to something else -- or when nothing has changed for
:data:`SETTLE_QUIET` seconds, since editing properties registers no operator.
Timers, not threads: an extension may not start threads.
"""

from __future__ import annotations

import time

import bpy

from ..rig import builder
from ..solver import manager

#: Operators whose redo panel is still an edit in progress. Each module adds its own.
ADJUSTING: set[str] = set()
#: Seconds between looks at whether the edit has settled.
SETTLE_POLL = 0.5
#: Seconds without a further change after which the edit counts as done anyway.
SETTLE_QUIET = 10.0
#: Rigs whose compile is waiting for the edit to settle, by name.
_waiting: set[str] = set()
#: time.monotonic() of the last deferring edit, redo-panel re-runs included.
_last_change = 0.0


def adjustable(*idnames: str) -> None:
    """Count these operators' redo panels as edits still in progress."""
    ADJUSTING.update(idnames)


def defer(rig) -> bool:
    """Solve ``rig`` live on NumPy until the edit settles, then compile PyRoki once.

    Call it before the edit, not after: an edit ends on a depsgraph update, and live
    IK would compile on it. Returns whether this started the wait -- False if an
    earlier edit's is still running -- so an edit that is then refused can
    :func:`withdraw` its own and leave the earlier one alone.
    """
    global _last_change
    started = not manager.deferred(rig.name)
    manager.defer(rig.name)
    _waiting.add(rig.name)
    _last_change = time.monotonic()
    if not bpy.app.timers.is_registered(compile_when_settled):
        bpy.app.timers.register(compile_when_settled, first_interval=SETTLE_POLL)
    return started


def withdraw(rig) -> None:
    """Undo :func:`defer` for an edit that did not happen after all."""
    manager.release(rig.name)
    _waiting.discard(rig.name)


def still_adjusting(window_manager) -> bool:
    """Whether the last registered operation is an edit whose panel is still open."""
    operators = getattr(window_manager, "operators", None)
    return bool(operators) and operators[-1].bl_idname in ADJUSTING


def forget() -> None:
    """Drop every pending compile: a new file shares no rig with the old one.

    A waiting rig is looked up by name, and the next file may well hold a different
    robot under it.
    """
    _waiting.clear()
    if bpy.app.timers.is_registered(compile_when_settled):
        bpy.app.timers.unregister(compile_when_settled)


def interface_locked() -> bool:
    return bool(getattr(bpy.context.window_manager, "is_interface_locked", False))


def settled() -> bool:
    """Whether the edit is done: its panel has given way, or it has sat unchanged."""
    if not still_adjusting(bpy.context.window_manager):
        return True
    return time.monotonic() - _last_change >= SETTLE_QUIET


def compile_when_settled() -> float | None:
    """Timer: release the waiting rigs once the edit has settled, and compile them.

    Not while a job has the interface locked -- a render with Lock Interface on:
    Blender forbids timers touching data then.
    """
    if interface_locked() or not settled():
        return SETTLE_POLL
    for name in list(_waiting):
        _waiting.discard(name)
        manager.release(name)
        rig = bpy.data.objects.get(name)
        if rig is not None and builder.is_kinema_rig(rig):
            compile_now(rig)
    return None


def compile_now(rig) -> None:
    """Pay PyRoki's compile for ``rig`` now, if live IK will want it, moving nothing.

    Not a solve of the IK goal, as Add IK Target's warm-up is: that one has just put
    the goal on the tool, while here the goal may be anywhere -- left behind while
    the arm was posed by hand with live IK off -- and a timer that fires ten seconds
    after the user stopped must not drag the robot to it. With live IK off, nothing:
    the first solve pays, if one ever comes.
    """
    ik_name = rig.get(builder.PROP_IK_BONE)
    if not ik_name or ik_name not in rig.pose.bones or not rig.kinema_ik_enabled:
        return
    if getattr(rig, "kinema_solver_mode", manager.MODE_PYROKI) != manager.MODE_PYROKI:
        return
    solver = manager.get_solver(rig, ik_name)
    if solver is None:
        return
    window = next(iter(bpy.context.window_manager.windows), None)
    if window is not None:
        window.cursor_set("WAIT")
    try:
        solver.compile(rig)
    finally:
        if window is not None:
            window.cursor_set("DEFAULT")
