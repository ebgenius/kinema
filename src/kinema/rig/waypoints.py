"""Turning a set of taught waypoints into the spans between them.

Deliberately free of ``bpy``: deciding what happens between two waypoints is
ordering and arithmetic, so it can be tested in a plain venv the way
``kinematics.py`` is. Everything Blender-shaped -- the property group, the
operators, the keyframes -- lives in ``ops/waypoints.py``.

The one idea worth stating plainly:

    **The timeline is the ordering.** A waypoint carries a frame, not a
    position in a list. That is how a Blender user already thinks about when
    things happen, and it means dragging a key in the dope sheet reorders the
    job without touching the panel.

A *span* is the motion that arrives at a waypoint, so the move type belongs to
the waypoint being arrived at rather than to the one being left. That reads the
way robot programs do -- ``MoveL(pick)`` says how to get *to* ``pick`` -- and it
means the first waypoint needs no move type at all.
"""

from __future__ import annotations

from dataclasses import dataclass

#: How the robot arrives at a waypoint.
MOVE_JOINT = "JOINT"
MOVE_LINEAR = "LINEAR"
MOVE_TYPES = (MOVE_JOINT, MOVE_LINEAR)


class WaypointError(ValueError):
    """A set of waypoints that cannot be turned into motion."""


@dataclass(frozen=True)
class Span:
    """One move: from ``start`` at ``start_frame`` to ``end`` at ``end_frame``.

    ``start`` and ``end`` are whatever the caller passed in -- property group
    items in Blender, plain objects in the tests. This module only reads their
    ``frame`` and ``move``.
    """

    start: object
    end: object
    move: str

    @property
    def start_frame(self) -> int:
        return int(self.start.frame)

    @property
    def end_frame(self) -> int:
        return int(self.end.frame)

    @property
    def frames(self) -> int:
        return self.end_frame - self.start_frame


def ordered(waypoints) -> list:
    """Waypoints in time order.

    Sorted by frame, ties broken by the order they were taught. Python's sort
    is stable, so passing the collection in its own order is enough to make
    that second rule hold without recording anything extra.
    """
    return sorted(waypoints, key=lambda item: int(item.frame))


def spans(waypoints) -> list[Span]:
    """The moves between consecutive waypoints, in time order.

    Raises :class:`WaypointError` rather than guessing when the list cannot
    describe a motion: two waypoints on one frame would ask the robot to be in
    two places at once, and a single waypoint has nothing to move between.
    """
    items = ordered(waypoints)
    if len(items) < 2:
        raise WaypointError(
            "At least two waypoints are needed to generate motion; "
            f"this rig has {len(items)}"
        )

    for previous, current in zip(items, items[1:], strict=False):
        if int(previous.frame) == int(current.frame):
            raise WaypointError(
                f"'{previous.name}' and '{current.name}' are both on frame "
                f"{int(current.frame)}; give them different frames"
            )

    return [
        Span(start=previous, end=current, move=_move_of(current))
        for previous, current in zip(items, items[1:], strict=False)
    ]


def _move_of(waypoint) -> str:
    move = str(getattr(waypoint, "move", MOVE_JOINT) or MOVE_JOINT)
    if move not in MOVE_TYPES:
        raise WaypointError(
            f"'{waypoint.name}' has an unknown move type '{move}'; "
            f"expected one of {', '.join(MOVE_TYPES)}"
        )
    return move


def frame_range(waypoints) -> tuple[int, int]:
    """First and last frame the waypoints cover."""
    items = ordered(waypoints)
    if not items:
        raise WaypointError("This rig has no waypoints")
    return int(items[0].frame), int(items[-1].frame)
