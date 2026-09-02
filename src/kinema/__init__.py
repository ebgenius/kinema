"""Kinema -- animation-ready robot rigs in Blender.

Import a robot description from disk -- URDF, xacro or MJCF -- get a single
clean armature with one 1-DoF bone per joint, and drive it with PyRoki IK that
behaves like an ordinary Blender IK control. A bundled catalogue of 186 robots
says where to find one; Kinema downloads nothing itself.

Extension metadata lives in ``blender_manifest.toml``; there is deliberately no
``bl_info`` here, since Blender 4.2+ extensions take their metadata from the
manifest and a stale ``bl_info`` would silently disagree with it.
"""

from __future__ import annotations

import bpy

from . import handlers, prefs, runtime
from .ops import attach, ik, import_dae, import_robot, pose
from .ui import panel

# panel first: ops/pose imports helpers from it, and registration order
# decides which classes exist when Blender resolves parent panels.
_MODULES = (prefs, panel, import_dae, import_robot, pose, ik, attach)


def register() -> None:
    runtime.ensure_vendor_path()

    for module in _MODULES:
        for cls in module.classes:
            bpy.utils.register_class(cls)

    for module in _MODULES:
        if hasattr(module, "register_props"):
            module.register_props()

    handlers.register_handlers()


def unregister() -> None:
    handlers.unregister_handlers()

    for module in reversed(_MODULES):
        if hasattr(module, "unregister_props"):
            module.unregister_props()

    for module in reversed(_MODULES):
        for cls in reversed(module.classes):
            bpy.utils.unregister_class(cls)

    runtime.unload_solver_stack()
    runtime.remove_vendor_path()
