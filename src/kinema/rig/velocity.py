"""Joint speeds against their velocity limits: the part that needs no Blender.

A joint's speed at a frame is how far it moved since the frame before, times
the frame rate. The value one frame back comes from one of two places:

* **The joint's own curve**, when it has one and nothing else is writing the
  channel. Exact at any frame, however the user got there: a jump, a scrub or
  an edit in the graph editor all read the same.
* **What was seen**, when live IK drives the joint. Nothing on disk says where
  the solver put it a frame ago, so :class:`History` remembers the last two
  frames each rig was seen at. Playback and stepping visit neighbouring
  frames, so the answer is there; after a jump it is not, and the speed is
  unknown rather than guessed.

A joint with neither -- no curve, no solver -- holds its value from frame to
frame. It is measured against what was seen when that is to hand, which
catches anything outside Kinema animating it during playback, and reads as
still otherwise.

A gap of a few frames is measured as an average, because playback that drops
frames to keep up still wants a warning. An average can only under-read the
fastest frame inside the gap, never over-read it, so it can miss a spike but
never invent one.

Everything Blender-shaped -- reading channels, the handler, the panel, the
viewport overlay -- lives in ``ops/velocity.py`` and ``ui/overlay.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Largest gap, in frames, measured as an average rather than called unknown.
MAX_GAP = 3

#: Relative slack before a speed counts as over its limit. Pose channels are
#: float32, so a move keyed at exactly the limit reads back a hair either side.
TOLERANCE = 1e-4


@dataclass(frozen=True)
class Reading:
    """One joint's speed at the current frame, against its limit."""

    joint: str
    #: rad/s for a revolute joint, m/s for a prismatic one.
    limit: float
    #: Same units. None when there is nothing to measure against.
    speed: float | None
    prismatic: bool = False

    @property
    def ratio(self) -> float | None:
        """Speed as a fraction of the limit: 1.5 is half as fast again."""
        return None if self.speed is None else self.speed / self.limit

    @property
    def over(self) -> bool:
        return self.speed is not None and self.speed > self.limit * (1.0 + TOLERANCE)


def speed(now: float, before: float, frames: int, fps: float) -> float:
    """Average speed over ``frames`` frames, in units per second."""
    return abs(now - before) * fps / abs(frames)


def clamp(value: float, lower: float | None, upper: float | None) -> float:
    """``value`` inside the range, where there is one."""
    if lower is not None and value < lower:
        return lower
    if upper is not None and value > upper:
        return upper
    return value


class History:
    """The last two frames each rig was seen at, and its joint values there.

    Two, not one: seeing a rig again at the frame it is already on refreshes
    that frame and keeps the one before, so editing a pose on frame 11 is still
    measured against frame 10.
    """

    def __init__(self) -> None:
        self._current: dict[object, tuple[int, dict[str, float]]] = {}
        self._previous: dict[object, tuple[int, dict[str, float]]] = {}

    def observe(self, key, frame: int, values: dict[str, float]) -> None:
        current = self._current.get(key)
        if current is not None and current[0] != frame:
            self._previous[key] = current
        self._current[key] = (int(frame), dict(values))

    def before(self, key, frame: int) -> tuple[int, dict[str, float]] | None:
        """A neighbouring frame this rig was seen at, and what it held there.

        The nearer of the two records is taken, so after stepping back from 11
        to 10 the answer is 11 -- the speed across that one frame -- rather
        than something older.
        """
        found = None
        for record in (self._current.get(key), self._previous.get(key)):
            if record is None or record[0] == frame:
                continue
            gap = abs(frame - record[0])
            if gap <= MAX_GAP and (found is None or gap < abs(frame - found[0])):
                found = record
        return found

    def forget(self, key=None) -> None:
        if key is None:
            self._current.clear()
            self._previous.clear()
        else:
            self._current.pop(key, None)
            self._previous.pop(key, None)


def format_speed(value: float | None, *, prismatic: bool, degrees: bool) -> str:
    """A speed the way the panel shows it: in the scene's rotation unit."""
    if value is None:
        return "—"
    if prismatic:
        return f"{value:.3g} m/s"
    if degrees:
        shown = math.degrees(value)
        return f"{shown:.0f}°/s" if shown >= 10.0 else f"{shown:.1f}°/s"
    return f"{value:.2f} rad/s"
