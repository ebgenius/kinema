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

#: The buttons a new install shows, in order: Live IK spanning the bar, then
#: Configurations beside Auto Config, Rest Pose beside Snap to Tool, and Motion
#: Check beside Graph Editor.
DEFAULT = (
    "LIVE_IK",
    "CONFIGURATIONS",
    "AUTO_CONFIG",
    "REST_POSE",
    "SNAP_IK",
    "MOTION_CHECK",
    "GRAPH_EDITOR",
)


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
    _toggle(layout, rig, "kinema_ik_enabled", "Live IK")


def _auto_configuration(layout, _context, rig) -> None:
    _toggle(layout, rig, "kinema_auto_configuration", "Auto Config")


def _toggle(layout, rig, name: str, text: str) -> None:
    """A switch that plays, as the Target (IK) panel's Live IK shows it: Play
    while it is on, running, and Pause while it is off."""
    on = bool(getattr(rig, name))
    layout.prop(rig, name, text=text, toggle=True, icon="PLAY" if on else "PAUSE")


def _snap_to_tool(layout, _context, rig) -> None:
    """Put the IK target back on the tool: only to be pressed while it is off it."""
    from ..ops import ik as ik_ops

    row = layout.row()
    row.enabled = ik_ops.goal_off_tool(rig)
    row.operator("kinema.snap_ik", text="Snap to Tool", icon="SNAP_ON")


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
    "AUTO_CONFIG": Button(
        "Auto Config",
        "Find another configuration when live IK can't reach the target in this one",
        "AUTO",
        _auto_configuration,
    ),
    "SNAP_IK": Button(
        "Snap to Tool",
        "Put the IK target where the tool is: pressable while the target is off it",
        "SNAP_ON",
        _snap_to_tool,
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


def status(context, rig) -> tuple[str, str, str]:
    """What to say under the buttons, as (text, icon, dismiss): the one thing most
    worth knowing about the robot shown, and what to do about it, and the operator
    that dismisses it, where it can be. Always one line, so the panels under the
    bar don't move as it changes."""
    from ..ops import ik as ik_ops
    from ..ops.waypoints import ik_bone_name
    from .panel import is_selected

    if not is_selected(context, rig):
        # Kept while something else is selected: said here, above every panel,
        # so none is read as the selected object's.
        return ("Not selected: the robot last shown", "RESTRICT_SELECT_ON", "")
    if ik_bone_name(rig) is None:
        return ("Add an IK target to pose the tool by it", "INFO", "")
    if not rig.kinema_ik_enabled:
        return ("Live IK paused: pose the joints, or turn it on", "INFO", "")
    if ik_ops.goal_off_tool(rig):
        if rig.kinema_auto_configuration and ik_ops.auto_found_nothing(rig):
            # Searched, and no configuration reaches it: out of the robot's reach.
            return ("The target is out of reach", "ERROR", "")
        if rig.kinema_auto_configuration:
            return ("The tool can't reach the target", "ERROR", "")
        return ("Can't reach the target: try Config or Snap to Tool", "ERROR", "")
    if ik_ops.switched_since_recorded(rig):
        # Kept until the next waypoint is recorded, or dismissed: a linear move from
        # the last one can't cross the switch, and Generate would refuse it.
        return (
            "Config changed: linear move can't cross it",
            "ERROR",
            "kinema.dismiss_config_change",
        )
    return ("Following the target", "CHECKMARK", "")


def draw(layout, context, rig, keys) -> None:
    """The quick bar for ``rig``, with the buttons ``keys`` names, in a box, and
    what to say about the robot under them.

    Each button in a column of its own, the two of a row split down the middle,
    apart: an aligned row shares its width out by what each button holds, so one
    with arrows took more than its half, and pressed against its neighbour.
    """
    box = layout.box()
    column = box.column()
    column.scale_y = 1.2
    for row_keys in rows(keys):
        if len(row_keys) == 1:
            BUTTONS[row_keys[0]].draw(column.row(), context, rig)
            continue
        halves = column.split(factor=0.5)
        for key in row_keys:
            BUTTONS[key].draw(halves.column(), context, rig)
    text, icon, dismiss = status(context, rig)
    row = box.row()
    row.label(text=text, icon=icon)
    if dismiss:
        # Seen, and let be: the usual line comes back.
        row.operator(dismiss, text="", icon="X", emboss=False)
