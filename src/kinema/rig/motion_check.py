"""Checking a generated job against what it was taught: the part that needs no Blender.

Generate Motion writes keys, and nothing about a keyframe says whether a robot
can play it. The check steps through the job the way playback does, records
each frame's joint values and tool pose, and this module reduces them to one
line per move:

* a linear move's worst distance from its straight line, and how far the tool
  turned off the shortest rotation between its two ends;
* a joint that reached its limit;
* the fastest any joint moved, against its velocity limit;
* the largest jump any joint made between two frames, which is how a
  configuration flip shows up.

Deliberately free of ``bpy``, like ``rig/velocity.py``. Stepping the frames and
reading the rig live in ``ops/waypoints.py``.

Numbers that were not measured are negative and names that do not apply are
empty, rather than None: the same fields are stored on the rig as Blender
properties, which have no None, and :func:`problems` reads either.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .velocity import TOLERANCE as SPEED_TOLERANCE
from .waypoints import MOVE_LINEAR

#: How far the tool may stray from a linear move's line, in metres, before it
#: is reported. A tenth of a millimetre: the solver holds the line to microns
#: when it can hold it at all, so anything past this is the arm failing to.
LINE_TOLERANCE = 1e-4
#: The same for its orientation against the shortest turn between the ends.
TWIST_TOLERANCE = math.radians(0.1)
#: How close to a limit counts as on it: radians for a revolute joint, metres
#: for a prismatic one. A clamped solve lands exactly on the limit, so this only
#: has to be wider than float32 noise in the channels.
LIMIT_MARGIN = 1e-3
#: One frame's change that reads as a jump rather than as motion. A revolute
#: joint turning a radian in one frame is over 1300 degrees a second at 24 fps,
#: beyond any industrial wrist; a configuration flip is typically most of a
#: half-turn. A prismatic axis covering a quarter metre is 6 m/s.
JUMP_REVOLUTE = 1.0
JUMP_PRISMATIC = 0.25


@dataclass(frozen=True)
class Joint:
    """What the check needs to know about one joint bone."""

    name: str
    prismatic: bool = False
    lower: float | None = None
    upper: float | None = None
    #: rad/s or m/s; None when the description gives none.
    velocity: float | None = None


@dataclass(frozen=True)
class Sample:
    """One frame of the job as it played."""

    frame: int
    #: Joint values as the rig shows them, in the order of the joints list.
    q: np.ndarray
    #: Tool pose in rig space, 4x4.
    tool: np.ndarray


@dataclass
class MoveCheck:
    """One move's findings. Field for field what is stored on the rig."""

    name: str
    move: str
    start_frame: int
    end_frame: int
    #: Metres; negative on a joint move, which has no line to keep to.
    line_error: float = -1.0
    #: Radians; negative on a joint move.
    twist_error: float = -1.0
    limit_joint: str = ""
    speed_joint: str = ""
    #: Fastest speed as a fraction of that joint's limit; 0 when none has one.
    speed_ratio: float = 0.0
    jump_joint: str = ""
    jump: float = 0.0
    jump_prismatic: bool = False
    jump_frame: int = 0


def check_move(
    name: str,
    move: str,
    start_frame: int,
    end_frame: int,
    start_pose: np.ndarray,
    end_pose: np.ndarray,
    samples: list[Sample],
    joints: list[Joint],
    fps: float,
) -> MoveCheck:
    """Findings for the move from ``start_frame`` to ``end_frame``, both inclusive.

    ``samples`` may cover more than the move; only its own frames are read.
    """
    own = sorted(
        (s for s in samples if start_frame <= s.frame <= end_frame),
        key=lambda s: s.frame,
    )
    result = MoveCheck(name, move, int(start_frame), int(end_frame))
    if not own:
        return result

    if move == MOVE_LINEAR:
        result.line_error, result.twist_error = _line_errors(
            own, start_frame, end_frame, start_pose, end_pose
        )

    values = np.array([s.q for s in own], dtype=float)
    result.limit_joint = _limit_joint(values, joints)

    frames = np.array([s.frame for s in own])
    for (a, b), (qa, qb) in zip(
        zip(frames, frames[1:], strict=False),
        zip(values, values[1:], strict=False),
        strict=False,
    ):
        gap = int(b - a)
        if gap <= 0:
            continue
        for index, joint in enumerate(joints):
            change = abs(float(qb[index] - qa[index]))
            if joint.velocity:
                ratio = change * fps / gap / joint.velocity
                if ratio > result.speed_ratio:
                    result.speed_ratio, result.speed_joint = ratio, joint.name
            if change / gap > result.jump:
                result.jump = change / gap
                result.jump_joint = joint.name
                result.jump_prismatic = joint.prismatic
                result.jump_frame = int(b)
    return result


def problems(check) -> list[str]:
    """What is wrong with one move, in words; empty when nothing is.

    Reads the fields of a :class:`MoveCheck` or anything shaped like one -- the
    copy stored on the rig included.
    """
    found = []
    if check.line_error > LINE_TOLERANCE:
        found.append(f"{check.line_error * 1000:.1f} mm off the line")
    if check.twist_error > TWIST_TOLERANCE:
        found.append(f"turned {math.degrees(check.twist_error):.1f}° off")
    if check.limit_joint:
        found.append(f"{check.limit_joint} at its limit")
    if check.speed_ratio > 1.0 + SPEED_TOLERANCE:
        found.append(
            f"{check.speed_joint} at {check.speed_ratio * 100:.0f}% of its speed limit"
        )
    threshold = JUMP_PRISMATIC if check.jump_prismatic else JUMP_REVOLUTE
    if check.jump > threshold:
        amount = (
            f"{check.jump * 1000:.0f} mm"
            if check.jump_prismatic
            else f"{math.degrees(check.jump):.0f}°"
        )
        found.append(f"{check.jump_joint} jumps {amount} at frame {check.jump_frame}")
    return found


# --------------------------------------------------------------------------
# measuring
# --------------------------------------------------------------------------
def _line_errors(
    samples: list[Sample],
    start_frame: int,
    end_frame: int,
    start_pose: np.ndarray,
    end_pose: np.ndarray,
) -> tuple[float, float]:
    """Worst distance from the segment, and worst angle off the shortest turn.

    Distance to the *segment*, not to where the tool should be at that frame:
    the claim a linear move makes is about the path. The turn is compared at the
    frame's own fraction of the move, since orientation has no path to measure
    against otherwise.
    """
    a = np.asarray(start_pose, dtype=float)[:3, 3]
    b = np.asarray(end_pose, dtype=float)[:3, 3]
    segment = b - a
    length_squared = float(segment @ segment)
    q_start = quaternion(np.asarray(start_pose)[:3, :3])
    q_end = quaternion(np.asarray(end_pose)[:3, :3])
    span = max(end_frame - start_frame, 1)

    worst_line = worst_twist = 0.0
    for sample in samples:
        point = np.asarray(sample.tool, dtype=float)[:3, 3]
        if length_squared > 0.0:
            along = float(np.clip((point - a) @ segment / length_squared, 0.0, 1.0))
        else:
            along = 0.0
        worst_line = max(worst_line, float(np.linalg.norm(point - (a + along * segment))))

        expected = slerp(q_start, q_end, (sample.frame - start_frame) / span)
        worst_twist = max(
            worst_twist, angle_between(quaternion(np.asarray(sample.tool)[:3, :3]), expected)
        )
    return worst_line, worst_twist


def _limit_joint(values: np.ndarray, joints: list[Joint]) -> str:
    """The joint that came closest to a limit, if any reached it."""
    closest, name = LIMIT_MARGIN, ""
    for index, joint in enumerate(joints):
        column = values[:, index]
        margins = []
        if joint.lower is not None:
            margins.append(float(np.min(column - joint.lower)))
        if joint.upper is not None:
            margins.append(float(np.min(joint.upper - column)))
        if margins and min(margins) < closest:
            closest, name = min(margins), joint.name
    return name


# --------------------------------------------------------------------------
# rotations, in (w, x, y, z)
# --------------------------------------------------------------------------
def quaternion(rotation: np.ndarray) -> np.ndarray:
    """A unit quaternion for a rotation matrix, with w >= 0."""
    r = np.asarray(rotation, dtype=float)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
        )
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(max(1.0 + r[i, i] - r[j, j] - r[k, k], 0.0)) * 2.0
        q = np.zeros(4)
        q[0] = (r[k, j] - r[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (r[j, i] + r[i, j]) / s
        q[1 + k] = (r[k, i] + r[i, k]) / s
    q = q / (np.linalg.norm(q) or 1.0)
    return -q if q[0] < 0.0 else q


def slerp(a: np.ndarray, b: np.ndarray, fraction: float) -> np.ndarray:
    """Spherical interpolation the short way round."""
    b = -b if float(a @ b) < 0.0 else b
    cosine = float(np.clip(a @ b, -1.0, 1.0))
    if cosine > 0.9995:
        out = a + fraction * (b - a)
        return out / np.linalg.norm(out)
    angle = math.acos(cosine)
    return (
        math.sin((1.0 - fraction) * angle) * a + math.sin(fraction * angle) * b
    ) / math.sin(angle)


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """The rotation angle taking one orientation to the other, in radians."""
    return 2.0 * math.acos(min(abs(float(a @ b)), 1.0))
