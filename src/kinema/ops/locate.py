"""Locate Description: point a rig at its description again (#92).

A rig keeps the path of the description it was imported from, and the solver
rebuilds its PyRoki model from that file when it needs one. Open the .blend on
another machine, or move the description's folder, and the path no longer
exists: live IK falls back to NumPy, and Optimize Motion is refused.

This finds it again. The chosen file is read as an import reads it -- with the
rig's xacro arguments, and ``package://`` and meshes resolved from where the
file now is -- and accepted only if it describes the same robot
(``rig/same_robot.py``). Accepting it drops the cached solver and Optimize's
kernels, so the next solve builds from the new file. Nothing else in the file
changes: waypoints, keys, capsules, limits and the TCP stay as they are; a
description whose limits differ is accepted, and the rig keeps its own.

A rig with an external axis describes itself from its bones, and a rig linked
from a library is the library's to fix: neither has anything to locate here.
"""

from __future__ import annotations

import os

import bpy
from bpy.props import StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper

from ..rig import builder
from ..ui.panel import active_rig

#: The sources a description file stands behind: URDF or xacro, and MJCF.
FILE_KINDS = ("file", "mjcf")


def description_missing(rig) -> bool:
    """Whether ``rig`` was imported from a file that isn't where it was."""
    if rig is None or rig.get(builder.PROP_SOURCE_KIND) not in FILE_KINDS:
        return False
    source = rig.get(builder.PROP_SOURCE)
    if not source or os.path.isfile(str(source)):
        return False
    from ..solver import rig_model

    return not rig_model.has_external_axes(rig)


def relink(rig, path: str) -> tuple[bool, str]:
    """Point ``rig`` at the description at ``path``, if it describes the same robot.

    Returns whether it did, and what to report: what differs, where it doesn't.
    """
    from ..io import loader
    from ..io.xacro_args import parse_args
    from ..prefs import package_search_paths
    from ..rig.same_robot import compare
    from ..solver import manager, rig_model

    if rig.library is not None:
        return False, (
            f"'{rig.name}' is linked from a library: locate its description in the "
            "library's own file"
        )
    if rig_model.has_external_axes(rig):
        return False, (
            "This robot has an external axis, so the solver describes it from its own "
            "bones: there is no description file to locate"
        )
    result = loader.load_file(
        path,
        xacro_args=parse_args(rig.get(builder.PROP_XACRO_ARGS, "")),
        extra_search_paths=package_search_paths(),
    )
    if result.error:
        return False, result.error
    if result.source is None:
        return False, f"'{path}' isn't a description file Kinema can read again"
    try:
        ours = rig_model.model_from_rig(rig)
    except ValueError as exc:
        return False, f"Couldn't read the robot this rig was built from: {exc}"
    comparison = compare(ours, result.model)
    if not comparison.same_robot:
        shown = comparison.differences[:4]
        more = len(comparison.differences) - len(shown)
        return False, (
            f"Not the robot this rig was built from: {'; '.join(shown)}"
            + (f"; and {more} more" if more > 0 else "")
            + ". A changed robot is imported again, not located"
        )

    kind, value = result.source
    rig[builder.PROP_SOURCE_KIND] = kind
    rig[builder.PROP_SOURCE] = value
    # The solver and Optimize's kernels were built, or refused, from the file
    # that was missing: the next solve builds from this one.
    manager.invalidate(rig.name)
    message = f"Located the description of '{rig.get(builder.PROP_ROBOT_NAME, rig.name)}'"
    if comparison.limits:
        shown = comparison.limits[:3]
        more = len(comparison.limits) - len(shown)
        message += (
            ". Its limits differ, and the rig keeps its own: " + "; ".join(shown)
            + (f"; and {more} more" if more > 0 else "")
        )
    return True, message


class KINEMA_OT_locate_description(Operator, ImportHelper):
    bl_idname = "kinema.locate_description"
    bl_label = "Locate Description"
    bl_description = (
        "Point this robot at its description file again, where the one it was "
        "imported from is missing. The file must describe the same robot"
    )
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".urdf"
    filter_glob: StringProperty(default="*.urdf;*.xacro;*.xml", options={"HIDDEN"})

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return active_rig(context) is not None

    def invoke(self, context: bpy.types.Context, event) -> set[str]:
        rig = active_rig(context)
        source = rig.get(builder.PROP_SOURCE) if rig is not None else None
        if source:
            # Named as it was: the browser shows what to look for.
            self.filepath = os.path.basename(str(source))
        return super().invoke(context, event)

    def execute(self, context: bpy.types.Context) -> set[str]:
        rig = active_rig(context)
        done, message = relink(rig, self.filepath)
        self.report({"INFO"} if done else {"ERROR"}, message)
        return {"FINISHED"} if done else {"CANCELLED"}


classes = (KINEMA_OT_locate_description,)
