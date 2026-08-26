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

from . import prefs, runtime
from .ops import import_dae
from .ui import panel

_MODULES = (prefs, import_dae, panel)


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

    bpy.app.timers.register(_deferred_start, first_interval=0.1)


def unregister() -> None:
    if bpy.app.timers.is_registered(_deferred_start):
        bpy.app.timers.unregister(_deferred_start)

    for module in reversed(_MODULES):
        if hasattr(module, "unregister_props"):
            module.unregister_props()

    for module in reversed(_MODULES):
        for cls in reversed(module.classes):
            bpy.utils.unregister_class(cls)

    runtime.unload_solver_stack()
    runtime.remove_vendor_path()
