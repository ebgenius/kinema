"""The robot model as the rig describes it, for a rig the description no longer covers.

PyRoki solves on a URDF, and normally that is the file the rig was imported from.
An external axis breaks that: a rail or a spindle added in Blender is in no
description on disk, and a BEFORE axis has moved the whole robot besides. So once a
rig has one, the model PyRoki gets is written from the rig itself.

That is possible because a Kinema rig describes itself. Every joint bone records
its URDF joint type, limits, the link it moves and the fixed transform from the
bone's frame to that link's (``builder.PROP_LINK_CORRECTION``), so

    link frame at rest  = bone.matrix_local @ correction
    joint origin        = parent link frame⁻¹ @ link frame
    joint axis          = the bone's +Y, expressed in the link frame

and the tree is the bone hierarchy with everything but joint bones skipped. Joints
are named after their bones, which is how the manager maps its chain onto PyRoki's
joint vector anyway, and the links after the ones each bone records -- so the TCP's
link, the elbow's link and every correction the manager already holds still apply.

What is lost is what the rig never had: fixed joints are folded into the links they
carry. A mimic joint's bone is written back as a ``<mimic>`` of the actuated joint
at the end of its chain of mimics, with the multipliers and offsets folded into one:
the form PyRoki takes it in.

The root link is the armature's origin, which is where the manager expresses every
goal. Collapsing the robot's own base link into it is exact: nothing moves the base
link relative to the armature except the external axes, which are joints here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..rig.kinematics import JointSpec, LinkSpec, RobotModel

#: The root link: the armature's own origin.
WORLD_LINK = "kinema_world"

#: A joint bone's axis of motion, in bone coordinates. Guaranteed by the builder.
BONE_AXIS = np.array([0.0, 1.0, 0.0])


@dataclass
class BoneRecord:
    """The facts one joint bone carries, as plain values."""

    name: str
    #: The nearest ancestor that is also a joint bone; None when it hangs off Root.
    parent: str | None
    #: bone.matrix_local: the bone's rest frame in armature space.
    rest: np.ndarray
    #: builder.PROP_LINK_CORRECTION: bone rest frame -> link rest frame.
    correction: np.ndarray
    joint_type: str
    child_link: str
    lower: float | None = None
    upper: float | None = None
    velocity: float | None = None
    acceleration: float | None = None
    jerk: float | None = None
    effort: float | None = None
    #: For a mimic joint's bone: the joint bone it follows, and how.
    mimic_joint: str | None = None
    mimic_multiplier: float = 1.0
    mimic_offset: float = 0.0

    @property
    def link_frame(self) -> np.ndarray:
        return self.rest @ self.correction


def model_from_records(name: str, records: list[BoneRecord]) -> RobotModel:
    """A RobotModel whose zero configuration is the rig's rest pose."""
    by_name = {record.name: record for record in records}
    links = {WORLD_LINK: LinkSpec(WORLD_LINK)}
    joints = []
    for record in records:
        parent = by_name.get(record.parent) if record.parent else None
        parent_frame = parent.link_frame if parent is not None else np.eye(4)
        origin = np.linalg.inv(parent_frame) @ record.link_frame
        # The bone's +Y in the link frame. The correction is a rotation for every
        # bone the builder makes -- bone heads sit on the link origin -- so this
        # is the whole of it.
        axis = record.correction[:3, :3].T @ BONE_AXIS
        joints.append(
            JointSpec(
                name=record.name,
                joint_type=record.joint_type,
                parent_link=parent.child_link if parent is not None else WORLD_LINK,
                child_link=record.child_link,
                origin=origin,
                axis=axis / np.linalg.norm(axis),
                lower=record.lower,
                upper=record.upper,
                velocity=record.velocity,
                acceleration=record.acceleration,
                jerk=record.jerk,
                effort=record.effort,
                mimic_joint=record.mimic_joint,
                mimic_multiplier=record.mimic_multiplier,
                mimic_offset=record.mimic_offset,
            )
        )
        links[record.child_link] = LinkSpec(record.child_link)
    return RobotModel(name=name, links=links, joints=joints, root_link=WORLD_LINK)


# --------------------------------------------------------------------------
# reading the rig
# --------------------------------------------------------------------------
def has_external_axes(rig) -> bool:
    """Whether any joint bone on ``rig`` is an external axis.

    Anything without bones has none, so a caller holding only a rig's stored
    source -- as the reload tests do -- goes on to reload it.
    """
    from ..rig import builder

    bones = getattr(getattr(rig, "data", None), "bones", None) or ()
    return any(builder.PROP_EXTERNAL in bone for bone in bones)


def records_from_rig(rig) -> list[BoneRecord]:
    """One record per joint bone, and per mimic joint's bone, in rig order."""
    from ..rig import builder

    def moves(bone) -> bool:
        return builder.PROP_JOINT_NAME in bone or builder.PROP_MIMIC_JOINT in bone

    # A mimic bone records the URDF joint it follows, a mimic or not; its record,
    # the bone. Among the robot's own joints only: an external axis is never a
    # mimic's leader, and one added later can be called what a renamed joint was.
    bone_of_joint = {
        str(bone.get(builder.PROP_JOINT_NAME, bone.get(builder.PROP_MIMIC_JOINT))): bone.name
        for bone in rig.data.bones
        if moves(bone) and builder.PROP_EXTERNAL not in bone
    }

    def matrix(values) -> np.ndarray:
        return np.array([float(v) for v in values]).reshape(4, 4)

    def as_array(m) -> np.ndarray:
        return np.array([[m[r][c] for c in range(4)] for r in range(4)])

    def optional(bone, key) -> float | None:
        return float(bone[key]) if key in bone else None

    records = []
    for bone in rig.data.bones:
        if not moves(bone):
            continue
        stored = bone.get(builder.PROP_LINK_CORRECTION)
        if stored is None or len(stored) != 16:
            raise ValueError(f"'{bone.name}' records no link frame")
        parent = bone.parent
        while parent is not None and not moves(parent):
            parent = parent.parent
        leader = bone.get(builder.PROP_MIMIC_OF)
        if leader is not None and str(leader) not in bone_of_joint:
            raise ValueError(f"'{bone.name}' follows '{leader}', which the rig has no bone for")
        records.append(
            BoneRecord(
                name=bone.name,
                parent=parent.name if parent is not None else None,
                rest=as_array(bone.matrix_local),
                correction=matrix(stored),
                joint_type=str(bone.get(builder.PROP_JOINT_TYPE, "revolute")),
                child_link=str(bone.get(builder.PROP_CHILD_LINK, bone.name)),
                lower=optional(bone, builder.PROP_LOWER),
                upper=optional(bone, builder.PROP_UPPER),
                velocity=optional(bone, builder.PROP_VELOCITY),
                acceleration=optional(bone, builder.PROP_ACCELERATION),
                jerk=optional(bone, builder.PROP_JERK),
                effort=optional(bone, builder.PROP_EFFORT),
                mimic_joint=bone_of_joint[str(leader)] if leader is not None else None,
                mimic_multiplier=float(bone.get(builder.PROP_MIMIC_MULTIPLIER, 1.0)),
                mimic_offset=float(bone.get(builder.PROP_MIMIC_OFFSET, 0.0)),
            )
        )
    return _folded(records)


def _folded(records: list[BoneRecord]) -> list[BoneRecord]:
    """``records`` with each mimic following an actuated joint: a mimic of a mimic
    folded down the chain, ``m * (m' * q + o') + o``.

    A limit partway down a chain is lost, as it is in URDF itself, which has no
    way to say it: the rig's drivers keep it, and so do the NumPy chain's steps.
    """
    from dataclasses import replace

    by_name = {record.name: record for record in records}
    out = []
    for record in records:
        multiplier, offset = record.mimic_multiplier, record.mimic_offset
        leader = by_name.get(record.mimic_joint) if record.mimic_joint else None
        seen = {record.name}
        while leader is not None and leader.mimic_joint is not None:
            if leader.name in seen:
                raise ValueError(f"'{record.name}' follows a loop of mimics")
            seen.add(leader.name)
            offset = multiplier * leader.mimic_offset + offset
            multiplier = multiplier * leader.mimic_multiplier
            leader = by_name.get(leader.mimic_joint)
        if record.mimic_joint is not None:
            record = replace(
                record,
                mimic_joint=leader.name if leader is not None else record.mimic_joint,
                mimic_multiplier=multiplier,
                mimic_offset=offset,
            )
        out.append(record)
    return out


def model_from_rig(rig) -> RobotModel:
    from ..rig import builder

    name = str(rig.get(builder.PROP_ROBOT_NAME, rig.name))
    return model_from_records(name, records_from_rig(rig))
