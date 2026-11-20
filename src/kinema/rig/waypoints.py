"""Turning a robot's motions into the spans of its job.

Deliberately free of ``bpy``: deciding what happens between two waypoints is
ordering and arithmetic, so it can be tested in a plain venv the way
``kinematics.py`` is. Everything Blender-shaped -- the property groups, the
operators, the keyframes -- lives in ``ops/waypoints.py``.

**A waypoint is a pose; a motion is a move between two of them.** A motion
names its start and end waypoint, how it moves -- joint or linear -- and its
own start and end frame, as a strip in Blender's NLA has. So one waypoint can
start one motion and end another, and a motion can be moved or retimed without
touching the poses. :func:`plan` lays the motions out in time:

- the job runs in frame order, not list order;
- a gap where the next motion starts at the waypoint the last one ended on is
  a hold;
- a gap where it starts somewhere else gets a joint move across it, and a
  note saying so;
- an overlap is refused, and so is a jump with no frames to make it in.

Every span from the first frame to the last is a move, a hold or an added
joint move, so whatever reads the job -- generation, the Motion Check,
Optimize Motion -- sees it whole.

Files saved before motions kept the frame and the move on the waypoint, the
move belonging to the waypoint arrived at, as ``MoveL(pick)`` says how to get
*to* ``pick``. :func:`spans` still reads them that way, to convert them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: How a motion moves.
MOVE_JOINT = "JOINT"
MOVE_LINEAR = "LINEAR"
MOVE_TYPES = (MOVE_JOINT, MOVE_LINEAR)

#: What made a span: one of the motions, or :func:`plan` filling a gap.
MADE_MOTION = "MOTION"
MADE_HOLD = "HOLD"
MADE_BRIDGE = "BRIDGE"


class WaypointError(ValueError):
    """A set of waypoints or motions that cannot be turned into motion."""


class MotionsOverlap(WaypointError):
    """Two motions of one job overlap in time: nothing is generated until they don't."""


@dataclass(frozen=True)
class Span:
    """One move: from ``start`` at ``start_frame`` to ``end`` at ``end_frame``.

    ``start`` and ``end`` are whatever the caller passed in -- property group
    items in Blender, plain objects in the tests. The frames are ``first`` and
    ``last`` where given, as :func:`plan` gives them, and otherwise the
    waypoints' own ``frame``, as :func:`spans` reads a file saved before
    motions. ``name`` is what the Motion Check calls the span; ``made`` says
    whether a motion made it, or :func:`plan` did to fill a gap.
    """

    start: object
    end: object
    move: str
    first: int | None = None
    last: int | None = None
    name: str = ""
    made: str = MADE_MOTION

    @property
    def start_frame(self) -> int:
        return int(self.start.frame if self.first is None else self.first)

    @property
    def end_frame(self) -> int:
        return int(self.end.frame if self.last is None else self.last)

    @property
    def frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def label(self) -> str:
        """What the Motion Check calls this span: its motion's name, or the waypoint it reaches."""
        return self.name or str(self.end.name)


@dataclass(frozen=True)
class Plan:
    """A job laid out in time: its spans, first to last, and what was added to it."""

    spans: list
    #: One line per joint move :func:`plan` added between two motions.
    notes: list
    #: Motions retimed from the NLA editor, as ops/strips.py's ``catch_up``
    #: found them: planned at their new frames, which an operator writes into
    #: the Motions list once nothing will refuse it.
    retimes: list = field(default_factory=list)


def plan(motions, waypoints) -> Plan:
    """The job ``motions`` make, as spans from its first frame to its last.

    ``motions`` are whatever the caller passes -- property group items in
    Blender, plain objects in the tests -- with a ``name``, a ``start`` and an
    ``end`` waypoint id, a ``move``, and a ``frame_start`` and ``frame_end``.
    ``waypoints`` have a ``uid`` and a ``name``. See the module's docstring
    for what fills a gap. An overlap raises :class:`MotionsOverlap`, and
    anything else that can't be laid out a :class:`WaypointError`.
    """
    by_id = {str(waypoint.uid): waypoint for waypoint in waypoints}
    laid = []
    for motion in motions:
        start, end = by_id.get(str(motion.start)), by_id.get(str(motion.end))
        for which, waypoint in (("start", start), ("end", end)):
            if waypoint is None:
                raise WaypointError(
                    f"Motion '{motion.name}' has no {which} waypoint: pick one, or "
                    f"remove the motion"
                )
        first, last = int(motion.frame_start), int(motion.frame_end)
        if last <= first:
            raise WaypointError(
                f"Motion '{motion.name}' ends on frame {last}, not after it starts "
                f"on frame {first}"
            )
        laid.append((first, last, motion, start, end))
    if not laid:
        raise WaypointError(
            "There are no motions to generate: add one between two waypoints"
        )
    # By start frame, ties in list order: the job runs in time, as the NLA does.
    laid.sort(key=lambda item: item[0])

    out, notes = [], []
    for index, (first, last, motion, start, end) in enumerate(laid):
        if index:
            before_first, before_last, before, _, arrived = laid[index - 1]
            if first < before_last:
                raise MotionsOverlap(
                    f"Motions '{before.name}' (frames {before_first}-{before_last}) "
                    f"and '{motion.name}' (frames {first}-{last}) overlap: move or "
                    f"shorten one of them"
                )
            if arrived is start:
                if first > before_last:
                    out.append(
                        Span(arrived, start, MOVE_JOINT, before_last, first,
                             f"hold at {start.name}", MADE_HOLD)
                    )
            elif first == before_last:
                raise WaypointError(
                    f"'{motion.name}' starts at '{start.name}' on the frame "
                    f"'{before.name}' ends at '{arrived.name}': leave frames between "
                    f"them for the robot to move in, or start it at '{arrived.name}'"
                )
            else:
                out.append(
                    Span(arrived, start, MOVE_JOINT, before_last, first,
                         f"to {start.name}", MADE_BRIDGE)
                )
                notes.append(
                    f"'{motion.name}' starts at '{start.name}', not where "
                    f"'{before.name}' ends, so a joint move was added over frames "
                    f"{before_last}-{first}: check its speed"
                )
        out.append(
            Span(start, end, _move_of(motion), first, last, str(motion.name), MADE_MOTION)
        )
    return Plan(spans=out, notes=notes)


def ordered(waypoints) -> list:
    """Waypoints in time order, as a file saved before motions has them.

    Sorted by frame, ties broken by the order they were taught. Python's sort
    is stable, so passing the collection in its own order is enough to make
    that second rule hold without recording anything extra.
    """
    return sorted(waypoints, key=lambda item: int(item.frame))


def spans(waypoints) -> list[Span]:
    """The moves between consecutive waypoints, in time order, as a file saved
    before motions has them: each waypoint's ``frame`` and ``move``.

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


def _move_of(item) -> str:
    move = str(getattr(item, "move", MOVE_JOINT) or MOVE_JOINT)
    if move not in MOVE_TYPES:
        raise WaypointError(
            f"'{item.name}' has an unknown move type '{move}'; "
            f"expected one of {', '.join(MOVE_TYPES)}"
        )
    return move


def frame_range(waypoints) -> tuple[int, int]:
    """First and last frame the waypoints cover, in a file saved before motions."""
    items = ordered(waypoints)
    if not items:
        raise WaypointError("This rig has no waypoints")
    return int(items[0].frame), int(items[-1].frame)
