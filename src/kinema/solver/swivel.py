"""Where the elbow of a redundant arm should be, for a swivel an animator turns.

A seven-axis arm holding its tool still can still swing its elbow. On an arm with
a spherical shoulder and a spherical wrist -- the iiwa, and near enough the Panda
-- that swing is exact geometry: the shoulder centre S is fixed by the base, the
wrist centre W is fixed by the tool, and the upper arm and forearm are rigid, so
the elbow can only be where a sphere of the upper arm's length about S meets a
sphere of the forearm's length about W. Two spheres meet in a circle, and the
circle's axis is the line S-W.

So a swivel is an angle on that circle, and this module turns the angle into a
point. That point is what makes the control honest. An earlier elbow target was a
free point the user dragged anywhere, usually somewhere the elbow could not reach,
and a soft cost pulling toward an unreachable point settles by giving up some of
the tool. A point *on the circle* can be reached exactly, so there is nothing to
trade.

On an arm whose links are offset from those centres the circle is only close, and
the elbow goal is then near-reachable rather than exactly so.

Deliberately free of ``bpy``, like ``solver/chain.py`` and ``solver/branches.py``,
so the geometry can be tested in a plain venv.
"""

from __future__ import annotations

import numpy as np

#: Below this length a vector is treated as having no direction. Metres for the
#: shoulder-to-wrist span, unit-free for a projected direction.
DEGENERATE = 1e-9

#: How far off the shoulder-to-wrist line the elbow goal stays however
#: stretched the arm is, as a fraction of the upper arm. See elbow_circle.
STRETCHED_SIDE = 0.05


def elbow_circle(
    shoulder: np.ndarray, wrist: np.ndarray, upper: float, fore: float
) -> tuple[np.ndarray, np.ndarray | None, float]:
    """The circle the elbow can sweep: (centre, unit axis, radius).

    ``upper`` is the shoulder-to-elbow length and ``fore`` the elbow-to-wrist
    length, both taken from the rest pose because they do not change.

    A wrist past the arm's reach is clamped to it. The true circle there is a
    point on the straight line, but the radius never drops below
    :data:`STRETCHED_SIDE` of the upper arm -- see the comment below for why a
    goal on the line is worse than one just off it. The axis is None only when
    shoulder and wrist coincide, which no real arm does.
    """
    shoulder = np.asarray(shoulder, dtype=float)
    span = np.asarray(wrist, dtype=float) - shoulder
    distance = float(np.linalg.norm(span))
    if distance < DEGENERATE:
        return shoulder, None, 0.0
    axis = span / distance

    reach = min(distance, upper + fore)
    # How far along S-W the circle's plane sits: the law of cosines, rearranged
    # for the foot of the elbow's perpendicular onto the axis.
    along = (upper * upper - fore * fore + reach * reach) / (2.0 * reach)
    radius = float(np.sqrt(max(upper * upper - along * along, 0.0)))
    # Never quite on the line. A goal on the line has no side, and a stretched
    # arm solved toward one can straighten through it and come out with the
    # elbow on the far side: dragging arm7's target just past reach flipped its
    # elbow 162 degrees. Held a hair off, on the knob's side, the elbow stays
    # where the animator put it. The price is a goal fractionally off the circle
    # when the arm is within a whisker of fully stretched -- which is where the
    # tool is at the edge of its reach anyway.
    radius = max(radius, STRETCHED_SIDE * upper)
    return shoulder + along * axis, axis, radius


def _perpendicular(vector: np.ndarray, axis: np.ndarray) -> np.ndarray | None:
    """``vector`` with its component along ``axis`` removed, normalised."""
    flat = vector - float(np.dot(vector, axis)) * axis
    length = float(np.linalg.norm(flat))
    return flat / length if length > DEGENERATE else None


def elbow_goal(
    shoulder: np.ndarray,
    wrist: np.ndarray,
    upper: float,
    fore: float,
    direction: np.ndarray,
    current_elbow: np.ndarray,
) -> np.ndarray:
    """The point on the elbow's circle that ``direction`` points at.

    ``direction`` is the swivel handle's knob, in the same space as the other
    points. Only its part perpendicular to S-W means anything -- the part along
    the axis cannot move the elbow round the circle -- so it is dropped.

    ``current_elbow`` is the fallback for a knob pointing straight along the
    axis, which leaves no side to choose: keep the elbow on whichever side it is
    already on rather than inventing one. If that is on the axis too, the arm is
    stretched straight and the centre is the only answer there is.
    """
    centre, axis, radius = elbow_circle(shoulder, wrist, upper, fore)
    if axis is None:
        return np.asarray(current_elbow, dtype=float)

    side = _perpendicular(np.asarray(direction, dtype=float), axis)
    if side is None:
        side = _perpendicular(np.asarray(current_elbow, dtype=float) - centre, axis)
    if side is None:
        return centre
    return centre + radius * side


def signed_angle(axis: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    """Right-handed angle about ``axis`` from ``start`` to ``end``, in radians.

    Used to set a new swivel so its knob already points at the elbow, which is
    what keeps adding one from moving the robot. Both vectors are flattened onto
    the plane perpendicular to the axis first; if either has no direction there,
    the angle is 0 rather than nan.
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / (float(np.linalg.norm(axis)) or 1.0)
    a = _perpendicular(np.asarray(start, dtype=float), axis)
    b = _perpendicular(np.asarray(end, dtype=float), axis)
    if a is None or b is None:
        return 0.0
    return float(np.arctan2(np.dot(np.cross(a, b), axis), np.dot(a, b)))
