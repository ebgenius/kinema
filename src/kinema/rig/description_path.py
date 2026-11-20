"""Where a rig's description file is, kept relative to the .blend too (#92).

A rig keeps the absolute path of the description it was imported from
(``kinema_source``), and the solver rebuilds its model from that file. Move the
project folder, or open it on another machine, and the absolute path points at
where the file was. So the path is also kept relative to the .blend
(``kinema_source_relative``, ``//`` first, as Blender writes the paths of its
images), where the description sits in the .blend's folder or under it: a
project moved as a whole finds its robot again, with nothing to locate.

The relative path is the one tried first, and the absolute path is the fallback,
so a .blend moved on its own, out of its project folder, still finds the file
where it was. A description in another folder altogether, or on another drive,
is kept absolute only: a ``..`` path would find whatever file of that name sits
beside the .blend's new place, which needn't be this robot's.

Both are written when the file is saved, against the file it is saved as, so a
rig imported before the .blend was first saved has one too, and Save As keeps it
pointing at the same description from the new place. The absolute path is
brought up to date at the same time, where the relative one found the file:
Kinema 0.6 and earlier read only that. Blender's *Default To: Relative Paths*
preference turns the relative path off, as it does for images.

:func:`relative_to` needs no Blender, so it is tested without one.
"""

from __future__ import annotations

import os

#: The path relative to the .blend, ``//`` first, where the description is in the
#: .blend's folder or under it. Rigs saved before it was kept have none.
PROP_SOURCE_RELATIVE = "kinema_source_relative"

#: The sources a description file stands behind: URDF or xacro, and MJCF.
FILE_KINDS = ("file", "mjcf")


def relative_to(path: str, blend_path: str) -> str | None:
    """``path`` relative to the .blend at ``blend_path``, as Blender writes one, where
    it is in the .blend's folder or under it; None anywhere else, or unsaved."""
    if not blend_path:
        return None
    try:
        relative = os.path.relpath(path, os.path.dirname(blend_path))
    except ValueError:  # Another drive.
        return None
    if relative == os.pardir or relative.startswith(os.pardir + os.sep) or os.path.isabs(relative):
        return None
    return "//" + relative


def _blend_of(rig) -> str:
    """The .blend ``rig`` is saved in: its library's, if it is linked from one."""
    import bpy

    if rig.library is not None:
        return bpy.path.abspath(rig.library.filepath)
    return bpy.data.filepath


def source_path(rig) -> str | None:
    """Where ``rig``'s description is: by the path relative to its .blend, where that
    finds the file, else the absolute path. None for a rig with no source."""
    from . import builder

    source = rig.get(builder.PROP_SOURCE)
    if not source:
        return None
    relative = rig.get(PROP_SOURCE_RELATIVE)
    blend = _blend_of(rig) if relative else ""
    if blend and str(relative).startswith("//"):
        found = os.path.normpath(
            os.path.join(os.path.dirname(blend), str(relative)[2:].replace("\\", os.sep))
        )
        if os.path.isfile(found):
            return found
    return str(source)


def remember(rig, blend_path: str) -> None:
    """Keep ``rig``'s description path relative to ``blend_path`` as well, where it
    can be, and the absolute one up to date. A description that is missing is left
    as it is recorded, for Locate Description to find."""
    import bpy

    from . import builder

    if rig.library is not None or rig.get(builder.PROP_SOURCE_KIND) not in FILE_KINDS:
        return
    path = source_path(rig)
    if path is None or not os.path.isfile(path):
        return
    path = os.path.abspath(path)
    if rig.get(builder.PROP_SOURCE) != path:
        rig[builder.PROP_SOURCE] = path
    relative = (
        relative_to(path, blend_path)
        if bpy.context.preferences.filepaths.use_relative_paths
        else None
    )
    if relative is None:
        if PROP_SOURCE_RELATIVE in rig:
            del rig[PROP_SOURCE_RELATIVE]
    elif rig.get(PROP_SOURCE_RELATIVE) != relative:
        rig[PROP_SOURCE_RELATIVE] = relative


def remember_all(blend_path: str) -> None:
    """:func:`remember` every rig in the file, against ``blend_path``."""
    import bpy

    from . import builder

    for obj in bpy.data.objects:
        if builder.is_kinema_rig(obj):
            remember(obj, blend_path)
