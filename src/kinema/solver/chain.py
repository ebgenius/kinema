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


def _rotation_about_y(angle: float) -> np.ndarray:
    cos, sin = np.cos(angle), np.sin(angle)
    matrix = np.eye(4)
    matrix[0, 0] = cos
    matrix[0, 2] = sin
    matrix[2, 0] = -sin
    matrix[2, 2] = cos
    return matrix


def _translation_along_y(distance: float) -> np.ndarray:
    matrix = np.eye(4)
    matrix[1, 3] = distance
    return matrix


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

    @property
    def dof(self) -> int:
        return len(self.bone_names)

    def joint_motion(self, q: np.ndarray) -> np.ndarray:
        """Per-joint local motion matrices for configuration ``q``."""
        return np.stack(
            [
                _rotation_about_y(float(value)) if revolute
                else _translation_along_y(float(value))
                for value, revolute in zip(q, self.is_revolute, strict=True)
            ]
        )

    def frames(self, q: np.ndarray) -> np.ndarray:
        """World transform of every joint frame, plus the tool, as (n+1, 4, 4)."""
        motions = self.joint_motion(q)
        out = np.empty((self.dof + 1, 4, 4))
        current = np.eye(4)
        for index in range(self.dof):
            current = current @ self.rest_relative[index] @ motions[index]
            out[index] = current
        out[self.dof] = current @ self.tool_offset
        return out

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
        frames = self.frames(q)
        tool_position = frames[-1][:3, 3]

        jacobian = np.zeros((6, self.dof))
        for index in range(self.dof):
            frame = frames[index]
            axis = frame[:3, :3] @ LOCAL_AXIS
            if self.is_revolute[index]:
                jacobian[:3, index] = np.cross(axis, tool_position - frame[:3, 3])
                jacobian[3:, index] = axis
            else:
                jacobian[:3, index] = axis
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
    # Axis-angle from a rotation matrix, guarding the degenerate cases where
    # the usual formula divides by zero.
    cos_angle = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))
    if angle < 1e-9:
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

    axis = np.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ]
    ) / (2.0 * np.sin(angle))
    error[3:] = axis * angle
    return error


# --------------------------------------------------------------------------
# Building a chain from a rig
# --------------------------------------------------------------------------
def chain_from_rig(armature_object, tip_bone_name: str) -> Chain | None:
    """Extract the chain of joint bones from the root down to ``tip_bone_name``.

    Returns None when the tip has no joint bones above it -- an unrigged or
    partially built armature -- rather than raising, because this is called
    from UI code that must stay usable.
    """
    import numpy as np

    from ..rig import builder

    bones = armature_object.data.bones
    tip = bones.get(tip_bone_name)
    if tip is None:
        return None

    # Walk up to the root, collecting only bones that represent joints.
    ancestry = []
    node = tip
    while node is not None:
        ancestry.append(node)
        node = node.parent
    ancestry.reverse()

    joints = [b for b in ancestry if builder.PROP_JOINT_NAME in b]
    if not joints:
        return None

    def as_array(matrix) -> np.ndarray:
        return np.array([[matrix[r][c] for c in range(4)] for r in range(4)])

    rest = [as_array(b.matrix_local) for b in joints]
    rest_relative = np.empty((len(joints), 4, 4))
    rest_relative[0] = rest[0]
    for index in range(1, len(joints)):
        rest_relative[index] = np.linalg.inv(rest[index - 1]) @ rest[index]

    tip_matrix = as_array(tip.matrix_local)
    tool_offset = np.linalg.inv(rest[-1]) @ tip_matrix

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


def write_configuration(armature_object, chain: Chain, q: np.ndarray) -> None:
    """Write joint values back onto the pose bones."""
    pose = armature_object.pose
    for index, name in enumerate(chain.bone_names):
        pose_bone = pose.bones[name]
        if chain.is_revolute[index]:
            pose_bone.rotation_mode = "YXZ"
            pose_bone.rotation_euler[1] = float(q[index])
        else:
            pose_bone.location[1] = float(q[index])
