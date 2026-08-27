"""Kinema -- animation-ready robot rigs in Blender.

Import any robot from the robot_descriptions catalog (or a local URDF/MJCF),
get a single clean armature with one 1-DoF bone per joint, and drive it with
PyRoki IK that behaves like an ordinary Blender IK control.

Extension metadata lives in ``blender_manifest.toml``; there is deliberately no
``bl_info`` here, since Blender 4.2+ extensions take their metadata from the
manifest and a stale ``bl_info`` would silently disagree with it.
"""

from __future__ import annotations

import bpy

from . import handlers, prefs, runtime
from .ops import ik, import_dae, import_robot, pose
from .ui import panel

# panel first: ops/pose imports helpers from it, and registration order
# decides which classes exist when Blender resolves parent panels.
_MODULES = (prefs, panel, import_dae, import_robot, pose, ik)


def _warm_up_solver() -> None:
    """Kick off a background import of the solver stack, if enabled.

    Runs from a one-shot timer rather than directly in ``register()``: during
    Blender startup the add-on is registered on the main thread before the UI
    exists, and preferences are not reliably readable that early.
    """
    preferences = prefs.get_prefs()
    if preferences is None or not preferences.warm_up_on_startup:
        return
    runtime.warm_up_async(debug=preferences.debug_logging)


def _deferred_start() -> None:
    preferences = prefs.get_prefs()
    if preferences is not None:
        # Same reason as _warm_up_solver: preferences are not reliably readable
        # during register(), and this has to land before the first description
        # module resolves its REPOSITORY_PATH.
        prefs.apply_cache_dir(preferences.cache_dir)
    _warm_up_solver()
    return None  # unregister the timer


def register() -> None:
    runtime.ensure_vendor_path()
    runtime.configure_jax_env()

    for module in _MODULES:
        for cls in module.classes:
            bpy.utils.register_class(cls)

    for module in _MODULES:
        if hasattr(module, "register_props"):
            module.register_props()

    handlers.register_handlers()
    bpy.app.timers.register(_deferred_start, first_interval=0.1)


def unregister() -> None:
    if bpy.app.timers.is_registered(_deferred_start):
        bpy.app.timers.unregister(_deferred_start)

    handlers.unregister_handlers()

    for module in reversed(_MODULES):
        if hasattr(module, "unregister_props"):
            module.unregister_props()

    for module in reversed(_MODULES):
        for cls in reversed(module.classes):
            bpy.utils.unregister_class(cls)

    runtime.unload_solver_stack()
    runtime.remove_vendor_path()
