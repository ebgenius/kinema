"""The quick bar: the buttons used all the time, above the Kinema tab's panels (#103).

Which buttons, and in what order, is the user's to choose in the add-on's
preferences; :data:`DEFAULT` is what they start with. Laid out two to a row,
with the first spanning the bar where there is an odd number of them.

Each button is a key into :data:`BUTTONS`: what the preferences list it as, and
how it draws. A key the add-on no longer knows -- from a newer Kinema's
preferences, say -- is skipped rather than drawn wrong.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

#: The buttons a new install shows, in order.
DEFAULT = ("LIVE_IK", "REST_POSE", "GRAPH_EDITOR", "CONFIGURATIONS", "MOTION_CHECK")


@dataclass(frozen=True)
class Button:
    """One button the quick bar can show: its name in the preferences, what it is
    for, its icon there, and how it draws into a layout for a rig."""

    label: str
    description: str
    icon: str
    draw: Callable


def _operator(idname: str, text: str, icon: str):
    def draw(layout, _context, _rig) -> None:
        layout.operator(idname, text=text, icon=icon)

    return draw


def _live_ik(layout, _context, rig) -> None:
    from ..ops.waypoints import ik_bone_name

    if ik_bone_name(rig) is None:
        layout.operator("kinema.add_ik", text="Add IK Target", icon="CON_KINEMATIC")
        return
    layout.prop(rig, "kinema_ik_enabled", text="Live IK", toggle=True, icon="CON_KINEMATIC")


def _configurations(layout, _context, rig) -> None:
    """Step through the configurations found for this pose, or find them.

    Once the target has moved, they were found for a pose it has left: the one
    picked is still named, without how many there were, and the middle searches
    again from where the arm stands, the arrows search again and step.
    """
    from ..ops import ik as ik_ops

    count = ik_ops.solution_count(rig)
    if not count:
        layout.operator("kinema.find_solutions", text="Configurations", icon="VIEWZOOM")
        return
    current = min(int(rig.kinema_active_solution) + 1, count)
    text = f"Config {current}" if ik_ops.solutions_are_stale(rig) else f"Config {current}/{count}"
    row = layout.row(align=True)
    row.operator("kinema.apply_solution", text="", icon="TRIA_LEFT").step = -1
    row.operator("kinema.find_solutions", text=text)
    row.operator("kinema.apply_solution", text="", icon="TRIA_RIGHT").step = 1


BUTTONS: dict[str, Button] = {
    "LIVE_IK": Button(
        "Live IK", "Turn live IK on and off, or add an IK target", "CON_KINEMATIC", _live_ik
    ),
    "REST_POSE": Button(
        "Rest Pose",
        "Put every joint back at zero",
        "LOOP_BACK",
        _operator("kinema.reset_pose", "Rest Pose", "LOOP_BACK"),
    ),
    "GRAPH_EDITOR": Button(
        "Graph Editor",
        "Open the joints' curves",
        "GRAPH",
        _operator("kinema.graph_editor", "Graph Editor", "GRAPH"),
    ),
    "CONFIGURATIONS": Button(
        "Configurations",
        "Find the arm's other ways to the pose, and step through them",
        "VIEWZOOM",
        _configurations,
    ),
    "MOTION_CHECK": Button(
        "Motion Check",
        "Play the job through and check every move",
        "CHECKMARK",
        _operator("kinema.check_motion", "Motion Check", "CHECKMARK"),
    ),
    "KEY_JOINTS": Button(
        "Key All",
        "Key every joint on this frame",
        "DECORATE_KEYFRAME",
        _operator("kinema.key_joints", "Key All", "DECORATE_KEYFRAME"),
    ),
    "RECORD_WAYPOINT": Button(
        "Record Waypoint",
        "Teach a waypoint where the tool is",
        "KEYFRAME_HLT",
        _operator("kinema.add_waypoint", "Record", "KEYFRAME_HLT"),
    ),
    "GENERATE": Button(
        "Generate Motion",
        "Key the job through its motions",
        "TRACKING",
        _operator("kinema.generate_motion", "Generate", "TRACKING"),
    ),
    "OPTIMIZE": Button(
        "Optimize Motion",
        "Solve the generated job within the robot's limits",
        "IPO_EASE_IN_OUT",
        _operator("kinema.optimize_motion", "Optimize", "IPO_EASE_IN_OUT"),
    ),
    "SNAP_IK": Button(
        "Snap to Tool",
        "Put the IK target where the tool is",
        "SNAP_ON",
        _operator("kinema.snap_ik", "Snap to Tool", "SNAP_ON"),
    ),
    "BAKE_IK": Button(
        "Bake IK",
        "Bake what live IK plays to keyframes",
        "RENDER_ANIMATION",
        _operator("kinema.bake_ik", "Bake IK", "RENDER_ANIMATION"),
    ),
}


def rows(keys) -> list[list[str]]:
    """``keys``, the known ones, laid out two to a row: the first alone where
    there is an odd number of them, spanning the bar."""
    keys = [key for key in keys if key in BUTTONS]
    first = [[keys[0]]] if len(keys) % 2 else []
    rest = keys[len(first) :]
    return first + [rest[index : index + 2] for index in range(0, len(rest), 2)]


def draw(layout, context, rig, keys) -> None:
    """The quick bar for ``rig``, with the buttons ``keys`` names.

    Each button in a column of its own, the two of a row split down the middle,
    apart: an aligned row shares its width out by what each button holds, so one
    with arrows took more than its half, and pressed against its neighbour.
    """
    column = layout.column()
    column.scale_y = 1.2
    for row_keys in rows(keys):
        if len(row_keys) == 1:
            BUTTONS[row_keys[0]].draw(column.row(), context, rig)
            continue
        halves = column.split(factor=0.5)
        for key in row_keys:
            BUTTONS[key].draw(halves.column(), context, rig)
