"""Joint speeds and accelerations against their limits: the part that needs no Blender.

A joint's speed at a frame is how far it moved since the frame before, times
the frame rate. Its acceleration is how much that speed changed since the
frame before that: the second difference over the last two frames, times the
frame rate squared. The values one and two frames back come from one of two
places:

* **The joint's own curve**, when it has one and nothing else is writing the
  channel. Exact at any frame, however the user got there: a jump, a scrub or
  an edit in the graph editor all read the same.
* **What was seen**, when live IK drives the joint. Nothing on disk says where
  the solver put it a frame ago, so :class:`History` remembers the last three
  frames each rig was seen at. Playback and stepping visit neighbouring
  frames, so the answer is there; after a jump it is not, and the speed or
  acceleration is unknown rather than guessed.

Jerk and effort limits are kept on the joints too, and listed, but not checked.
Effort needs the robot's masses and inertias, which a rig does not carry. Jerk
would be a third difference, and between frames it can pass a limit J only
where the acceleration jumps by more than J / fps within one frame -- so one of
those frames reads over J / (2 fps). For limits like the Franka Panda's, 15
rad/s² and 7500 rad/s³, that is ten times the acceleration limit at 24 fps,
which is flagged already.

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


#: What a :class:`Reading` measures, and the power of time in its unit.
SPEED = "speed"
ACCELERATION = "acceleration"
ORDERS = {SPEED: 1, ACCELERATION: 2}


@dataclass(frozen=True)
class Reading:
    """One joint's speed or acceleration at the current frame, against its limit."""

    joint: str
    #: Per second for a speed, per second squared for an acceleration; radians
    #: for a revolute joint, metres for a prismatic one.
    limit: float
    #: Same units. None when there is nothing to measure against.
    speed: float | None
    prismatic: bool = False
    #: :data:`SPEED` or :data:`ACCELERATION`. ``speed`` holds the value either way.
    kind: str = SPEED

    @property
    def ratio(self) -> float | None:
        """The value as a fraction of the limit: 1.5 is half as much again."""
        return None if self.speed is None else self.speed / self.limit

    @property
    def over(self) -> bool:
        return self.speed is not None and self.speed > self.limit * (1.0 + TOLERANCE)


def speed(now: float, before: float, frames: int, fps: float) -> float:
    """Average speed over ``frames`` frames, in units per second."""
    return abs(now - before) * fps / abs(frames)


def acceleration(points, fps: float) -> float:
    """How fast the speed is changing through three ``(frame, value)`` points, per s².

    The second derivative of the parabola through them. With evenly spaced
    frames that is the usual second difference, ``(q0 - 2 q1 + q2) fps²``; the
    general form keeps a dropped frame from reading as a spike, and does not
    care which order the points come in.
    """
    (t0, q0), (t1, q1), (t2, q2) = ((frame / fps, value) for frame, value in points)
    return abs(2.0 * (
        q0 / ((t0 - t1) * (t0 - t2))
        + q1 / ((t1 - t0) * (t1 - t2))
        + q2 / ((t2 - t0) * (t2 - t1))
    ))


def clamp(value: float, lower: float | None, upper: float | None) -> float:
    """``value`` inside the range, where there is one."""
    if lower is not None and value < lower:
        return lower
    if upper is not None and value > upper:
        return upper
    return value


class History:
    """The last three frames each rig was seen at, and its joint values there.

    Seeing a rig again at the frame it is already on refreshes that frame and
    keeps the ones before, so editing a pose on frame 11 is still measured
    against frames 10 and 9. Three, because an acceleration needs two frames
    besides the current one.
    """

    DEPTH = 3

    def __init__(self) -> None:
        #: key -> [(frame, values)], most recently seen first, frames distinct.
        self._seen: dict[object, list[tuple[int, dict[str, float]]]] = {}

    def observe(self, key, frame: int, values: dict[str, float]) -> None:
        records = [r for r in self._seen.get(key, []) if r[0] != frame]
        self._seen[key] = [(int(frame), dict(values)), *records][: self.DEPTH]

    def _around(self, key, frame: int) -> list[tuple[int, dict[str, float]]]:
        """Records other than ``frame``'s, within reach of it, nearest first."""
        near = [
            record for record in self._seen.get(key, [])
            if record[0] != frame and abs(frame - record[0]) <= MAX_GAP
        ]
        return sorted(near, key=lambda record: abs(frame - record[0]))

    def before(self, key, frame: int) -> tuple[int, dict[str, float]] | None:
        """A neighbouring frame this rig was seen at, and what it held there.

        The nearest record is taken, so after stepping back from 11 to 10 the
        answer is 11 -- the speed across that one frame -- rather than something
        older.
        """
        near = self._around(key, frame)
        return near[0] if near else None

    def two_before(self, key, frame: int) -> list[tuple[int, dict[str, float]]] | None:
        """The two nearest other frames this rig was seen at, for an acceleration."""
        near = self._around(key, frame)
        return near[:2] if len(near) >= 2 else None

    def forget(self, key=None) -> None:
        if key is None:
            self._seen.clear()
        else:
            self._seen.pop(key, None)


_PER_SECOND = {1: "/s", 2: "/s²", 3: "/s³"}


def _figure(value: float) -> str:
    """Three significant figures, but a whole number from 100 up: 1500, not 1.5e+03."""
    return f"{value:.0f}" if abs(value) >= 100.0 else f"{value:.3g}"


def format_rate(value: float | None, *, prismatic: bool, degrees: bool, order: int = 1) -> str:
    """A speed (``order`` 1), acceleration (2) or jerk (3) as the panel shows it.

    In the scene's rotation unit for a revolute joint, metres for a prismatic one.
    """
    if value is None:
        return "—"
    unit = _PER_SECOND[order]
    if prismatic:
        return f"{_figure(value)} m{unit}"
    if degrees:
        shown = math.degrees(value)
        return f"{shown:.0f}°{unit}" if shown >= 10.0 else f"{shown:.1f}°{unit}"
    return f"{value:.0f} rad{unit}" if value >= 100.0 else f"{value:.2f} rad{unit}"


def format_effort(value: float | None, *, prismatic: bool) -> str:
    """A torque in N·m for a revolute joint, a force in N for a prismatic one."""
    if value is None:
        return "—"
    return f"{_figure(value)} N" if prismatic else f"{_figure(value)} N·m"
