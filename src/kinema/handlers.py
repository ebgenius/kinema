"""Live IK: solve whenever the user moves an IK target.

This is what makes a Kinema rig feel like an ordinary Blender IK setup -- drag
the control, the arm follows -- while a nonlinear solver runs underneath.

Driving pose bones from ``depsgraph_update_post`` is a well-known way to hang
Blender, because writing to the scene from inside a depsgraph callback triggers
another depsgraph update, which calls the handler again. Two guards prevent
that, and both are needed:

1. A re-entrancy flag, so the write we cause cannot re-enter the solve.
2. A cached copy of each IK target's matrix, so an update we did *not* cause
   still does nothing unless the target actually moved. Without this the
   handler would re-solve on every unrelated scene change, and the write from
   guard 1 would schedule one more pass every time.

There is also a time budget. A rig whose solves run over the user's configured
limit has some updates dropped, so a 30-DoF humanoid degrades to a laggy but
usable viewport rather than freezing it -- while still solving at least one
update in five, so it keeps tracking and its timing gets re-measured.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import bpy
import numpy as np
from bpy.app.handlers import persistent

from .rig import builder
from .solver import manager

#: Guards against the handler re-entering itself via its own scene writes.
_solving = False
#: rig name -> (tip bone, IK target matrix) we last solved for. The tip is part
#: of the key because it is keyframable: handing the goal from the wrist to the
#: elbow mid-shot changes the chain while the goal matrix stays exactly where
#: it was, and comparing the matrix alone would call that "nothing moved".
_last_target: dict[str, tuple[str, np.ndarray]] = {}
#: rig name -> seconds the last solve took.
_last_duration: dict[str, float] = {}
#: rig name -> consecutive live updates skipped for being over budget.
_skipped: dict[str, int] = {}
#: Solve at least one update in every _MAX_SKIPS + 1, so a slow rig still
#: tracks (laggily) and its timing is re-measured rather than assumed.
_MAX_SKIPS = 4


def _np4(matrix) -> np.ndarray:
    return np.array([[matrix[r][c] for c in range(4)] for r in range(4)])


def _live_rigs(scene: bpy.types.Scene) -> list[bpy.types.Object]:
    """Rigs in this scene with live IK switched on."""
    return [
        obj
        for obj in scene.objects
        if obj.type == "ARMATURE"
        and obj.get(builder.PROP_IS_RIG)
        and getattr(obj, "kinema_ik_enabled", False)
        and getattr(obj, "kinema_solver_mode", manager.MODE_PYROKI) != manager.MODE_OFF
    ]


def _budget_seconds() -> float:
    from .prefs import get_prefs

    prefs = get_prefs()
    return (prefs.solve_timeout_ms / 1000.0) if prefs else 0.033


def _over_budget(rig_name: str, budget: float) -> bool:
    """Throttle a rig whose solves are too slow, without latching it off.

    Skipping purely on "the last solve was slow" is a trap: once skipped, the
    measurement never refreshes, so a single slow solve -- a JIT recompile, a
    frame where the machine was busy -- would disable live IK for the rest of
    the session. Instead, drop at most ``_MAX_SKIPS`` updates in a row and then
    solve anyway, which both keeps the viewport responsive and re-measures.
    """
    if _last_duration.get(rig_name, 0.0) <= budget:
        _skipped[rig_name] = 0
        return False
    if _skipped.get(rig_name, 0) >= _MAX_SKIPS:
        _skipped[rig_name] = 0
        return False
    _skipped[rig_name] = _skipped.get(rig_name, 0) + 1
    return True


def solve_rig(rig: bpy.types.Object, *, force: bool = False) -> bool:
    """Solve one rig if its IK target moved. Returns True if it wrote anything."""
    ik_bone = rig.get(builder.PROP_IK_BONE)
    if not ik_bone or ik_bone not in rig.pose.bones:
        return False

    target = _np4(rig.pose.bones[ik_bone].matrix)
    tip = manager.tip_bone(rig)
    previous = _last_target.get(rig.name)
    if (
        not force
        and previous is not None
        and previous[0] == tip
        and np.allclose(target, previous[1], atol=1e-7)
    ):
        return False

    solver = manager.get_solver(rig, ik_bone)
    if solver is None:
        return False

    first_solve = solver.solve_count == 0
    started = time.perf_counter()
    result = solver.solve(rig, getattr(rig, "kinema_solver_mode", manager.MODE_PYROKI))
    elapsed = time.perf_counter() - started

    # The first PyRoki solve for a rig pays JAX's JIT compilation -- on the
    # order of ten seconds, once. Recording that as "the solve time" would
    # convince the budget check that this rig is hopelessly slow and throttle
    # every later update, when in fact warm solves run in ~10 ms.
    if not first_solve:
        _last_duration[rig.name] = elapsed

    if result is None:
        return False

    # Record the goal we actually solved for, not the one we may have been
    # asked for a moment ago, so the next update compares against reality.
    _last_target[rig.name] = (tip, target)
    return True


@persistent
def on_depsgraph_update(scene: bpy.types.Scene, depsgraph=None) -> None:
    global _solving
    if _solving:
        return

    rigs = _live_rigs(scene)
    if not rigs:
        return

    budget = _budget_seconds()
    _solving = True
    try:
        for rig in rigs:
            if _over_budget(rig.name, budget):
                continue
            solve_rig(rig)
    finally:
        _solving = False


@persistent
def on_frame_change(scene: bpy.types.Scene, depsgraph=None) -> None:
    """Re-solve during playback and rendering.

    Frame changes are not skipped on the time budget: a slow render frame is
    far better than a wrong one.
    """
    global _solving
    if _solving:
        return
    rigs = _live_rigs(scene)
    if not rigs:
        return
    _solving = True
    try:
        for rig in rigs:
            solve_rig(rig, force=True)
    finally:
        _solving = False


@persistent
def on_load_post(_dummy=None) -> None:
    """A freshly opened file shares nothing with the previous one."""
    _last_target.clear()
    _last_duration.clear()
    _skipped.clear()
    manager.invalidate()


@contextmanager
def suspended():
    """Stop the live handlers solving for the duration of the block.

    Baking needs this and cannot get it by switching ``kinema_ik_enabled`` off:
    that property is animatable too, so a rig with a curve on it has the value
    restored by every ``frame_set`` before the frame-change handler runs, and
    the handler would solve each frame that the bake is about to solve itself.

    Reuses the re-entrancy flag rather than adding a second one, because it
    wants exactly what that flag already means: whatever is happening to the
    scene right now, it is not the handlers' business.
    """
    global _solving
    previous = _solving
    _solving = True
    try:
        yield
    finally:
        _solving = previous


def last_solve_ms(rig: bpy.types.Object) -> float | None:
    duration = _last_duration.get(rig.name)
    return duration * 1000.0 if duration is not None else None


def reset(rig_name: str | None = None) -> None:
    """Forget cached targets so the next update re-solves."""
    if rig_name is None:
        _last_target.clear()
        _last_duration.clear()
        _skipped.clear()
    else:
        _last_target.pop(rig_name, None)
        _last_duration.pop(rig_name, None)
        _skipped.pop(rig_name, None)


_HANDLERS = (
    (bpy.app.handlers.depsgraph_update_post, on_depsgraph_update),
    (bpy.app.handlers.frame_change_post, on_frame_change),
    (bpy.app.handlers.load_post, on_load_post),
)


def register_handlers() -> None:
    for collection, function in _HANDLERS:
        if function not in collection:
            collection.append(function)


def unregister_handlers() -> None:
    for collection, function in _HANDLERS:
        if function in collection:
            collection.remove(function)
    reset()
    manager.invalidate()
