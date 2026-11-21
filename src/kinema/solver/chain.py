"""A serial kinematic chain read straight off a Kinema rig.

The chain is derived from the armature, not from the URDF. That is deliberate:
a saved .blend must keep working when the original description is gone, moved,
or simply not downloaded on this machine. Every fact needed is already on the
rig -- bone rest matrices give the joint frames, the bone's local Y *is* the
joint axis by construction, and limits and joint type are custom properties.

Because Kinema builds bones so that a single local-Y channel equals the joint
value, forward kinematics here is exactly what Blender itself computes:

    T_i = T_{i-1} · A_i · M_i(q_i)

where ``A_i`` is the rest transform of bone *i* relative to bone *i-1*, and
``M_i`` is a rotation about local Y (revolute) or a translation along local Y
(prismatic). The class is plain NumPy with no ``bpy`` import beyond
construction, so the maths is testable outside Blender.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Bone-local axis of motion. Guaranteed by the rig builder.
LOCAL_AXIS = np.array([0.0, 1.0, 0.0])


def rotation_about_y(angle: float) -> np.ndarray:
    cos, sin = np.cos(angle), np.sin(angle)
    matrix = np.eye(4)
    matrix[0, 0] = cos
    matrix[0, 2] = sin
    matrix[2, 0] = -sin
    matrix[2, 2] = cos
    return matrix


def translation_along_y(distance: float) -> np.ndarray:
    matrix = np.eye(4)
    matrix[1, 3] = distance
    return matrix


@dataclass(frozen=True)
class MimicLink:
    """A mimic joint's bone on the way to the tool: it moves, but by a driver from
    another joint rather than as a degree of freedom of its own.

    A parallelogram in a palletizer is one: the wrist below it is carried by a
    joint whose value follows another's. Leaving it out would solve a mechanism
    the rig doesn't have.
    """

    #: The chain index of the joint it comes after, -1 before the first.
    after: int
    #: Its rest transform relative to the frame before it: that joint's, or the
    #: mimic's before it in the same gap.
    rest_relative: np.ndarray
    is_revolute: bool
    #: The chain index of the joint it follows, or -1 for one off the chain, whose
    #: value is ``fixed`` for the solve.
    leader: int
    multiplier: float
    offset: float
    fixed: float = 0.0
    lower: float = -np.inf
    upper: float = np.inf

    def value(self, q: np.ndarray) -> float:
        raw = self.multiplier * float(q[self.leader]) + self.offset if self.leader >= 0 else self.fixed
        return float(np.clip(raw, self.lower, self.upper))

    def moving(self, q: np.ndarray) -> bool:
        """Whether it moves with its leader here, rather than held at a limit."""
        if self.leader < 0:
            return False
        raw = self.multiplier * float(q[self.leader]) + self.offset
        return self.lower <= raw <= self.upper

    def motion(self, q: np.ndarray) -> np.ndarray:
        value = self.value(q)
        return rotation_about_y(value) if self.is_revolute else translation_along_y(value)


@dataclass
class Chain:
    """A root-to-tool chain of 1-DoF joints."""

    #: Bone names, base first.
    bone_names: list[str]
    #: True for revolute joints, False for prismatic.
    is_revolute: np.ndarray
    #: Rest transform of each joint relative to the previous one (n, 4, 4).
    rest_relative: np.ndarray
    #: Constant transform from the last joint's frame to the tool frame.
    tool_offset: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    #: True where the joint is genuinely bounded (continuous joints are not).
    limited: np.ndarray
    #: Mimic joints' bones between the joints, in chain order. Not degrees of
    #: freedom: each moves by its leader's value.
    mimics: tuple[MimicLink, ...] = ()

    @property
    def dof(self) -> int:
        return len(self.bone_names)

    def joint_motion(self, q: np.ndarray) -> np.ndarray:
        """Per-joint local motion matrices for configuration ``q``."""
        return np.stack(
            [
                rotation_about_y(float(value)) if revolute
                else translation_along_y(float(value))
                for value, revolute in zip(q, self.is_revolute, strict=True)
            ]
        )

    def _walk(self, q: np.ndarray) -> tuple[np.ndarray, list]:
        """Every joint frame plus the tool, as (n+1, 4, 4), and each mimic's frame."""
        motions = self.joint_motion(q)
        out = np.empty((self.dof + 1, 4, 4))
        mimic_frames = []
        current = np.eye(4)
        pending = list(self.mimics)

        def through(after: int, current):
            while pending and pending[0].after == after:
                mimic = pending.pop(0)
                current = current @ mimic.rest_relative @ mimic.motion(q)
                mimic_frames.append((mimic, current))
            return current

        current = through(-1, current)
        for index in range(self.dof):
            current = current @ self.rest_relative[index] @ motions[index]
            out[index] = current
            current = through(index, current)
        out[self.dof] = current @ self.tool_offset
        return out, mimic_frames

    def frames(self, q: np.ndarray) -> np.ndarray:
        """World transform of every joint frame, plus the tool, as (n+1, 4, 4)."""
        return self._walk(q)[0]

    def forward(self, q: np.ndarray) -> np.ndarray:
        """Tool pose for configuration ``q``."""
        return self.frames(q)[-1]

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Geometric Jacobian of the tool, as (6, dof): linear rows then angular.

        Standard construction: for a revolute joint the tool's linear velocity
        is ``axis x (p_tool - p_joint)`` and its angular velocity is the axis;
        for a prismatic joint the linear part is the axis and there is no
        angular part.
        """
        frames, mimic_frames = self._walk(q)
        tool_position = frames[-1][:3, 3]
        jacobian = np.zeros((6, self.dof))

        def column(frame, revolute: bool) -> np.ndarray:
            axis = frame[:3, :3] @ LOCAL_AXIS
            if revolute:
                return np.concatenate([np.cross(axis, tool_position - frame[:3, 3]), axis])
            return np.concatenate([axis, np.zeros(3)])

        for index in range(self.dof):
            jacobian[:, index] = column(frames[index], bool(self.is_revolute[index]))
        # A mimic moves the tool as its leader turns, scaled: its column adds to
        # the leader's. Held at a limit, it doesn't move.
        for mimic, frame in mimic_frames:
            if mimic.moving(q):
                jacobian[:, mimic.leader] += mimic.multiplier * column(frame, mimic.is_revolute)
        return jacobian

    def clamp(self, q: np.ndarray) -> np.ndarray:
        """Clamp into joint limits, leaving unlimited joints untouched."""
        return np.where(self.limited, np.clip(q, self.lower, self.upper), q)


def pose_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Six-vector twist taking ``current`` to ``target``.

    The rotation part is the axis-angle of the relative rotation, expressed in
    world axes so it lines up with the geometric Jacobian's angular rows.
    """
    error = np.empty(6)
    error[:3] = target[:3, 3] - current[:3, 3]

    relative = target[:3, :3] @ current[:3, :3].T
    # Axis-angle from a rotation matrix. The angle from both its sine (the skew
    # part's length) and cosine (the trace), not the cosine alone: near zero an
    # arccos of the trace is only as good as the square root of the matrix's
    # error, and a goal read off Blender's float32 matrices pushed it past 1,
    # reading a turn of 4e-4 rad as none.
    skew = 0.5 * np.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ]
    )
    sin_angle = float(np.linalg.norm(skew))
    cos_angle = float((np.trace(relative) - 1.0) * 0.5)
    angle = float(np.arctan2(sin_angle, cos_angle))
    if angle < 1e-12:
        error[3:] = 0.0
        return error
    if angle > np.pi - 1e-6:
        # Near 180 degrees: recover the axis from the symmetric part instead.
        axis = np.sqrt(np.clip(np.diag(relative) * 0.5 + 0.5, 0.0, None))
        largest = int(np.argmax(axis))
        axis = axis / (np.linalg.norm(axis) or 1.0)
        # Sign is ambiguous at exactly pi; pick one consistently.
        if relative[(largest + 1) % 3, largest] < 0:
            axis = -axis
        error[3:] = axis * angle
        return error

    error[3:] = skew / sin_angle * angle
    return error


# --------------------------------------------------------------------------
# Building a chain from a rig
# --------------------------------------------------------------------------
def chain_bones(armature_object, tip_bone_name: str) -> list:
    """The joint bones from the root down to ``tip_bone_name``, root first.

    Which joints a solver aimed at that tip drives, without building one: no
    matrices, so it is cheap enough for a draw callback. Empty when the tip is
    missing or has no joint above it.
    """
    from ..rig import builder

    # Walk up to the root, collecting only bones that represent joints.
    ancestry = []
    node = armature_object.data.bones.get(tip_bone_name)
    while node is not None:
        ancestry.append(node)
        node = node.parent
    ancestry.reverse()
    return [b for b in ancestry if builder.PROP_JOINT_NAME in b]


def chain_from_rig(armature_object, tip_bone_name: str) -> Chain | None:
    """Extract the chain of joint bones from the root down to ``tip_bone_name``.

    Returns None when the tip has no joint bones above it -- an unrigged or
    partially built armature -- rather than raising, because this is called
    from UI code that must stay usable.
    """
    import numpy as np

    from ..rig import builder

    tip = armature_object.data.bones.get(tip_bone_name)
    joints = chain_bones(armature_object, tip_bone_name)
    if not joints:
        return None

    def as_array(matrix) -> np.ndarray:
        return np.array([[matrix[r][c] for c in range(4)] for r in range(4)])

    # Every bone that moves on the way down, joint or mimic, root first: each
    # one's rest relative to the one before it, whichever kind that is.
    ancestry = []
    node = tip
    while node is not None:
        ancestry.append(node)
        node = node.parent
    ancestry.reverse()
    moving = [
        b for b in ancestry
        if builder.PROP_JOINT_NAME in b or builder.PROP_MIMIC_JOINT in b
    ]
    index_of = {b.name: i for i, b in enumerate(joints)}
    # A mimic records its leader's URDF joint name; the chain is by bone. An
    # external axis is never one, and could have taken a renamed joint's name.
    bone_of_joint = {
        str(b[builder.PROP_JOINT_NAME]): b.name
        for b in armature_object.data.bones
        if builder.PROP_JOINT_NAME in b and builder.PROP_EXTERNAL not in b
    }

    rest_relative = np.empty((len(joints), 4, 4))
    mimics = []
    previous = np.eye(4)
    after = -1
    for bone in moving:
        here = as_array(bone.matrix_local)
        relative = np.linalg.inv(previous) @ here
        previous = here
        if bone.name in index_of:
            after = index_of[bone.name]
            rest_relative[after] = relative
            continue
        leader_bone = bone_of_joint.get(str(bone.get(builder.PROP_MIMIC_OF, "")))
        leader = index_of.get(leader_bone, -1)
        multiplier = float(bone.get(builder.PROP_MIMIC_MULTIPLIER, 1.0))
        offset = float(bone.get(builder.PROP_MIMIC_OFFSET, 0.0))
        fixed = 0.0
        if leader < 0:
            # Its leader is off the chain, so not solved for: held where it is.
            source = armature_object.pose.bones.get(leader_bone) if leader_bone else None
            if source is not None:
                prismatic = source.bone.get(builder.PROP_JOINT_TYPE) == "prismatic"
                value = float(source.location[1] if prismatic else source.rotation_euler[1])
                # Where the leader stands, through its limit, as the driver reads it.
                value = min(
                    max(value, float(source.bone.get(builder.PROP_LOWER, -np.inf))),
                    float(source.bone.get(builder.PROP_UPPER, np.inf)),
                )
                fixed = multiplier * value + offset
        mimics.append(
            MimicLink(
                after=after,
                rest_relative=relative,
                is_revolute=bone.get(builder.PROP_JOINT_TYPE, "revolute") != "prismatic",
                leader=leader,
                multiplier=multiplier,
                offset=offset,
                fixed=fixed,
                lower=float(bone.get(builder.PROP_LOWER, -np.inf)),
                upper=float(bone.get(builder.PROP_UPPER, np.inf)),
            )
        )

    tip_matrix = as_array(tip.matrix_local)
    tool_offset = np.linalg.inv(previous) @ tip_matrix

    is_revolute = np.array(
        [b.get(builder.PROP_JOINT_TYPE, "revolute") != "prismatic" for b in joints]
    )
    lower = np.array([float(b.get(builder.PROP_LOWER, -np.inf)) for b in joints])
    upper = np.array([float(b.get(builder.PROP_UPPER, np.inf)) for b in joints])
    limited = np.array([builder.PROP_LOWER in b for b in joints])

    return Chain(
        bone_names=[b.name for b in joints],
        is_revolute=is_revolute,
        rest_relative=rest_relative,
        tool_offset=tool_offset,
        lower=lower,
        upper=upper,
        limited=limited,
        mimics=tuple(mimics),
    )


def read_configuration(armature_object, chain: Chain) -> np.ndarray:
    """Current joint values, read from the pose bones' driving channels."""
    pose = armature_object.pose
    values = np.zeros(chain.dof)
    for index, name in enumerate(chain.bone_names):
        pose_bone = pose.bones[name]
        values[index] = (
            pose_bone.rotation_euler[1]
            if chain.is_revolute[index]
            else pose_bone.location[1]
        )
    return values


def displayed_configuration(
    armature_object, chain: Chain, raw: np.ndarray, only: np.ndarray | None = None
) -> np.ndarray:
    """Joint values as the rig shows them: channels with the constraint stack applied.

    ``read_configuration`` reads the channels, which is what an animator keys and
    what a solve seeds from. It is not always what is on screen. A joint with a
    limit constraint is clamped after its channel, so a rail dragged to 1.55 m
    against a 1.5 m limit reads 1.55 while the carriage stands at 1.5 -- and a
    joint the solver must hold where it stands has to be held at the 1.5.

    Taken from each bone's evaluated matrix relative to its evaluated parent,
    which is the composition Blender itself uses, so any constraint counts, not
    only Kinema's own limit. A revolute value keeps the channel's turn count: a
    continuous joint wound past half a turn is not unwound. ``only`` restricts
    the work to a mask of joints; the rest come back as ``raw``.
    """
    pose = armature_object.pose
    values = np.array(raw, dtype=float)
    for index, name in enumerate(chain.bone_names):
        if only is not None and not only[index]:
            continue
        pose_bone = pose.bones[name]
        frame = pose_bone.bone.matrix_local
        if pose_bone.parent is not None:
            parent = pose_bone.parent
            frame = parent.matrix @ parent.bone.matrix_local.inverted() @ frame
        local = frame.inverted() @ pose_bone.matrix
        if chain.is_revolute[index]:
            angle = float(np.arctan2(local[0][2], local[2][2]))
            values[index] = values[index] + float(np.angle(np.exp(1j * (angle - values[index]))))
        else:
            values[index] = float(local[1][3])
    return values


def write_configuration(
    armature_object, chain: Chain, q: np.ndarray, skip: np.ndarray | None = None
) -> None:
    """Write joint values back onto the pose bones, leaving any ``skip`` joints alone."""
    pose = armature_object.pose
    for index, name in enumerate(chain.bone_names):
        if skip is not None and skip[index]:
            continue
        pose_bone = pose.bones[name]
        if chain.is_revolute[index]:
            pose_bone.rotation_mode = "YXZ"
            pose_bone.rotation_euler[1] = float(q[index])
        else:
            pose_bone.location[1] = float(q[index])
