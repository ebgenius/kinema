"""Whether a description describes the robot a rig was built from (#92).

A rig keeps the path of the description it was imported from, and the solver
rebuilds its model from that file. Moved to another machine, the file is gone,
and Locate Description points the rig at it again -- but only at a file that
describes the same robot. Driving a rig with another robot's kinematics would
solve every pose wrong and say nothing.

The same robot here is: the same moving joints, by name and type, each moving
a link by the same name; each one's frame in the same place and turned the same
way, with the robot at rest, so its axis is too; and the same chain, each joint
moved by the same one before it. A difference in any of those is a changed
robot, which is a re-import. The links' names and frames matter beyond the
joints': the rig keeps a correction per link, and its IK aims at a link by its
name, so a frame turned about its joint's axis, or a link renamed, would leave
the solver aiming somewhere else.

Limits are another matter. They are kept on the rig from the import, and the
Motion Check and Optimize read them there, so a description whose limits were
since updated still drives the rig as it was: those differences are listed,
for the user to know, and change nothing.

Pure numpy over :class:`~kinema.rig.kinematics.RobotModel`, so it is tested
without Blender.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: How far a joint's frame may sit from where it was, in metres: well under
#: anything a description changes on purpose, well over float noise.
POSITION_TOLERANCE = 1e-4
#: How far its axis, or its frame, may turn: in radians, about 0.006 degrees --
#: for the axis, as the length of the difference between two unit vectors.
AXIS_TOLERANCE = 1e-4
#: Limits that differ by less are the same, in radians or metres (per second).
LIMIT_TOLERANCE = 1e-6

#: The limits compared, with how a report names each.
LIMITS = (
    ("lower", "lower limit"), ("upper", "upper limit"), ("velocity", "speed limit"),
    ("effort", "effort limit"),
)


@dataclass
class Comparison:
    """What sets a description apart from a rig's robot."""

    #: What makes it another robot, or this one changed: any of these refuses it.
    differences: list[str] = field(default_factory=list)
    #: Limits that differ, which the rig keeps its own of.
    limits: list[str] = field(default_factory=list)

    @property
    def same_robot(self) -> bool:
        return not self.differences


def _listed(names) -> str:
    names = [f"'{name}'" for name in names]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _limit(value) -> str:
    return "none" if value is None else f"{value:.4g}"


def compare(rig_model, described) -> Comparison:
    """How ``described`` differs from ``rig_model``, the model rebuilt from a rig."""
    result = Comparison()
    ours = {j.name: j for j in rig_model.actuated_joints}
    theirs = {j.name: j for j in described.actuated_joints}

    missing = [name for name in ours if name not in theirs]
    extra = [name for name in theirs if name not in ours]
    if missing:
        result.differences.append(f"it has no joint {_listed(missing)}")
    if extra:
        result.differences.append(f"it has a joint {_listed(extra)} the rig hasn't")

    our_frames, their_frames = rig_model.joint_frames(), described.joint_frames()
    our_owner = rig_model.nearest_actuated_ancestor()
    their_owner = described.nearest_actuated_ancestor()
    for name, joint in ours.items():
        other = theirs.get(name)
        if other is None:
            continue
        if joint.joint_type != other.joint_type:
            result.differences.append(
                f"'{name}' is {other.joint_type} there, {joint.joint_type} on the rig"
            )
            continue
        if joint.child_link != other.child_link:
            result.differences.append(
                f"'{name}' moves '{other.child_link}' there, '{joint.child_link}' on the rig"
            )
        ours_at, theirs_at = our_frames[name], their_frames[name]
        moved = float(np.linalg.norm(ours_at[:3, 3] - theirs_at[:3, 3]))
        if moved > POSITION_TOLERANCE:
            result.differences.append(f"'{name}' sits {moved * 1000:.1f} mm from the rig's")
        axis = _unit(ours_at[:3, :3] @ joint.axis)
        their_axis = _unit(theirs_at[:3, :3] @ other.axis)
        if float(np.linalg.norm(axis - their_axis)) > AXIS_TOLERANCE:
            turned = np.degrees(np.arccos(np.clip(float(axis @ their_axis), -1.0, 1.0)))
            result.differences.append(f"'{name}' turns about an axis {turned:.1f}° off the rig's")
        else:
            # The axis can agree while the frame is turned about it.
            relative = ours_at[:3, :3].T @ theirs_at[:3, :3]
            cosine = np.clip((float(np.trace(relative)) - 1.0) / 2.0, -1.0, 1.0)
            if float(np.arccos(cosine)) > AXIS_TOLERANCE:
                result.differences.append(
                    f"'{name}''s frame is turned {np.degrees(np.arccos(cosine)):.1f}° "
                    f"from the rig's"
                )
        before, their_before = our_owner.get(joint.parent_link), their_owner.get(other.parent_link)
        if before != their_before:
            result.differences.append(
                f"'{name}' follows {_named(their_before)} there, {_named(before)} on the rig"
            )
        for attribute, label in LIMITS:
            a, b = getattr(joint, attribute), getattr(other, attribute)
            if (a is None) != (b is None) or (
                a is not None and abs(float(a) - float(b)) > LIMIT_TOLERANCE
            ):
                result.limits.append(f"'{name}' {label} {_limit(b)}, the rig's {_limit(a)}")
    return result


def _named(joint: str | None) -> str:
    return "the base" if joint is None else f"'{joint}'"


def _unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(vector))
    return vector / length if length > 0 else vector
