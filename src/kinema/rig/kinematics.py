"""Robot kinematics as a Blender-free intermediate representation.

Everything Blender-specific lives in ``builder.py``. This module turns a parsed
URDF into a plain tree of links and joints plus the zero-configuration forward
kinematics, so the maths that decides where every bone goes can be tested with
``uv run pytest`` and no Blender at all.

The central idea, and the reason Kinema's armatures differ from every other
URDF importer's:

    A URDF joint rotates about an arbitrary 3-vector ``axis``. A Blender bone
    rotates about its own local **Y**. So build each bone with its Y axis
    *aligned to the joint axis*, and a single rotation channel becomes exactly
    the joint value -- a true 1-DoF control with real limits.

The cost is that the bone can no longer also point at its child link, which is
what importers usually optimise for. That trade is made deliberately: bone
*direction* is cosmetic and recoverable with a custom shape, whereas a bone
whose rotation does not correspond to a joint value is not a rig an animator
can use.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: URDF joint types that contribute a degree of freedom.
ACTUATED_TYPES = frozenset({"revolute", "continuous", "prismatic"})
#: Types Kinema understands at all. floating/planar are rejected explicitly
#: rather than silently mis-rigged.
SUPPORTED_TYPES = ACTUATED_TYPES | {"fixed"}


class UnsupportedJointError(ValueError):
    """A URDF used a joint type Kinema cannot represent as a single bone."""


@dataclass
class JointSpec:
    """One URDF joint, with its origin already resolved to a 4x4 matrix."""

    name: str
    joint_type: str
    parent_link: str
    child_link: str
    #: Child frame relative to the parent link frame, at zero configuration.
    origin: np.ndarray
    #: Unit axis of motion, expressed in the joint's own frame.
    axis: np.ndarray
    lower: float | None = None
    upper: float | None = None
    #: URDF <mimic>: this joint follows another. Recorded so the rig can lock
    #: the bone and drive it, instead of exposing a control that does nothing.
    mimic_joint: str | None = None
    mimic_multiplier: float = 1.0
    mimic_offset: float = 0.0

    @property
    def is_actuated(self) -> bool:
        return self.joint_type in ACTUATED_TYPES and self.mimic_joint is None

    @property
    def is_revolute(self) -> bool:
        return self.joint_type in {"revolute", "continuous"}

    @property
    def has_limits(self) -> bool:
        # "continuous" joints are explicitly unbounded -- that is the whole
        # point of the type, and clamping them would break multi-turn spins.
        return (
            self.joint_type != "continuous"
            and self.lower is not None
            and self.upper is not None
            and self.upper > self.lower
        )


@dataclass
class LinkSpec:
    """One URDF link and the visual meshes attached to it."""

    name: str
    #: (mesh_path_or_None, primitive_or_None, transform 4x4, scale 3, material)
    visuals: list[VisualSpec] = field(default_factory=list)


@dataclass
class VisualSpec:
    """A single <visual> element resolved to absolute paths and matrices."""

    #: Absolute path to a mesh file, or None for a geometric primitive.
    mesh_path: str | None
    #: ("box"|"cylinder"|"sphere", params) when this is a primitive.
    primitive: tuple[str, dict] | None
    #: Placement relative to the owning link's frame.
    origin: np.ndarray
    #: URDF <mesh scale>. Doosan and many ROS packages ship millimetre meshes
    #: and correct for it here rather than in the file's own <unit> tag.
    scale: np.ndarray
    name: str | None = None
    material_color: tuple[float, float, float, float] | None = None


@dataclass
class RobotModel:
    name: str
    links: dict[str, LinkSpec]
    joints: list[JointSpec]
    root_link: str

    @property
    def actuated_joints(self) -> list[JointSpec]:
        return [j for j in self.joints if j.is_actuated]

    def joints_by_parent(self) -> dict[str, list[JointSpec]]:
        table: dict[str, list[JointSpec]] = {}
        for joint in self.joints:
            table.setdefault(joint.parent_link, []).append(joint)
        return table

    def link_frames(self) -> dict[str, np.ndarray]:
        """World transform of every link at zero configuration.

        Zero configuration is the rig's rest pose. Because a Blender bone's
        rest matrix composes with its parent exactly the way URDF joint
        transforms do, building bones from these frames makes the armature's
        rest pose identical to URDF FK at q = 0.
        """
        frames = {self.root_link: np.eye(4)}
        by_parent = self.joints_by_parent()

        # Iterative walk: robot trees are shallow but recursion depth is not
        # worth risking on a 200-link humanoid.
        stack = [self.root_link]
        while stack:
            parent = stack.pop()
            for joint in by_parent.get(parent, ()):
                frames[joint.child_link] = frames[parent] @ joint.origin
                stack.append(joint.child_link)
        return frames

    def joint_frames(self) -> dict[str, np.ndarray]:
        """World transform of each joint's own frame at zero configuration.

        A joint's frame coincides with its child link's frame in URDF, so the
        bone head sits at the child link origin and the axis is expressed in
        that same frame.
        """
        link_frames = self.link_frames()
        return {j.name: link_frames[j.child_link] for j in self.joints}

    def nearest_actuated_ancestor(self) -> dict[str, str | None]:
        """Map each link to the actuated joint that ultimately moves it.

        Fixed joints get no bone -- they would be inert clutter in an
        animator's rig. Their transform is instead folded into whatever the
        fixed chain carries (meshes, tool frames), so the link still lands in
        exactly the right place with fewer controls on screen.
        """
        owner: dict[str, str | None] = {self.root_link: None}
        by_parent = self.joints_by_parent()

        stack = [self.root_link]
        while stack:
            parent = stack.pop()
            for joint in by_parent.get(parent, ()):
                owner[joint.child_link] = (
                    joint.name if joint.is_actuated else owner.get(parent)
                )
                stack.append(joint.child_link)
        return owner


# --------------------------------------------------------------------------
# Matrix helpers
# --------------------------------------------------------------------------
def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis XYZ (roll-pitch-yaw) to a 3x3 rotation matrix."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def make_transform(xyz, rpy) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = rpy_to_matrix(*(rpy if rpy is not None else (0.0, 0.0, 0.0)))
    matrix[:3, 3] = xyz if xyz is not None else (0.0, 0.0, 0.0)
    return matrix


def normalize(vector: np.ndarray, fallback=(0.0, 0.0, 1.0)) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        # A zero axis is malformed URDF; Z keeps the rig buildable.
        return np.asarray(fallback, dtype=float)
    return vector / norm


# --------------------------------------------------------------------------
# Building the model from yourdfpy
# --------------------------------------------------------------------------
def _origin_of(element) -> np.ndarray:
    origin = getattr(element, "origin", None)
    if origin is None:
        return np.eye(4)
    origin = np.asarray(origin, dtype=float)
    if origin.shape == (4, 4):
        return origin
    return np.eye(4)


def _build_material_map(urdf) -> dict[str, object]:
    """Name -> Material for robot-level ``<material>`` declarations.

    URDF lets a material be defined once at robot level and then referenced by
    name from a ``<visual>``. yourdfpy faithfully reproduces that: the visual's
    own material has ``color=None`` and only carries the name, so the colour
    has to be looked up or every such link imports untinted.
    """
    materials = getattr(getattr(urdf, "robot", None), "materials", None) or ()
    return {m.name: m for m in materials if getattr(m, "name", None)}


def _material_color(visual, material_map) -> tuple[float, float, float, float] | None:
    material = getattr(visual, "material", None)
    if material is None:
        return None
    color = getattr(material, "color", None)
    rgba = getattr(color, "rgba", None) if color is not None else None
    if rgba is None:
        target = material_map.get(getattr(material, "name", None))
        rgba = getattr(getattr(target, "color", None), "rgba", None)
    if rgba is None:
        return None
    values = [float(v) for v in rgba]
    while len(values) < 4:
        values.append(1.0)
    return tuple(values[:4])


def _visual_specs(link, material_map, mesh_resolver) -> list[VisualSpec]:
    specs: list[VisualSpec] = []
    for visual in getattr(link, "visuals", ()) or ():
        geometry = getattr(visual, "geometry", None)
        if geometry is None:
            continue
        origin = _origin_of(visual)
        color = _material_color(visual, material_map)
        name = getattr(visual, "name", None)

        mesh = getattr(geometry, "mesh", None)
        if mesh is not None and getattr(mesh, "filename", None):
            scale = getattr(mesh, "scale", None)
            scale = (
                np.asarray(scale, dtype=float)
                if scale is not None
                else np.ones(3, dtype=float)
            )
            specs.append(
                VisualSpec(
                    mesh_path=mesh_resolver(mesh.filename),
                    primitive=None,
                    origin=origin,
                    scale=scale,
                    name=name,
                    material_color=color,
                )
            )
            continue

        for kind in ("box", "cylinder", "sphere"):
            shape = getattr(geometry, kind, None)
            if shape is None:
                continue
            params: dict[str, object] = {}
            if kind == "box":
                params["size"] = [float(v) for v in getattr(shape, "size", (1, 1, 1))]
            elif kind == "cylinder":
                params["radius"] = float(getattr(shape, "radius", 0.5))
                params["length"] = float(getattr(shape, "length", 1.0))
            else:
                params["radius"] = float(getattr(shape, "radius", 0.5))
            specs.append(
                VisualSpec(
                    mesh_path=None,
                    primitive=(kind, params),
                    origin=origin,
                    scale=np.ones(3),
                    name=name,
                    material_color=color,
                )
            )
            break
    return specs


def model_from_urdf(urdf, mesh_resolver=None) -> RobotModel:
    """Build a :class:`RobotModel` from a ``yourdfpy.URDF``.

    Args:
        urdf: a loaded ``yourdfpy.URDF``.
        mesh_resolver: maps a URDF mesh filename (often ``package://...``) to an
            absolute path. Defaults to identity, which is enough for URDFs whose
            meshes are already absolute or relative to the cwd.
    """
    resolver = mesh_resolver or (lambda name: name)

    joints: list[JointSpec] = []
    for joint in urdf.robot.joints:
        joint_type = str(joint.type)
        if joint_type not in SUPPORTED_TYPES:
            raise UnsupportedJointError(
                f"joint '{joint.name}' has unsupported type '{joint_type}'; "
                f"Kinema supports {sorted(SUPPORTED_TYPES)}"
            )

        limit = getattr(joint, "limit", None)
        mimic = getattr(joint, "mimic", None)
        axis = getattr(joint, "axis", None)

        joints.append(
            JointSpec(
                name=joint.name,
                joint_type=joint_type,
                parent_link=joint.parent,
                child_link=joint.child,
                origin=_origin_of(joint),
                # URDF defaults the axis to (1, 0, 0) when omitted.
                axis=normalize(axis if axis is not None else (1.0, 0.0, 0.0)),
                lower=float(limit.lower) if limit is not None and limit.lower is not None else None,
                upper=float(limit.upper) if limit is not None and limit.upper is not None else None,
                mimic_joint=getattr(mimic, "joint", None) if mimic is not None else None,
                mimic_multiplier=float(getattr(mimic, "multiplier", None) or 1.0)
                if mimic is not None else 1.0,
                mimic_offset=float(getattr(mimic, "offset", None) or 0.0)
                if mimic is not None else 0.0,
            )
        )

    material_map = _build_material_map(urdf)
    links = {
        link.name: LinkSpec(
            name=link.name, visuals=_visual_specs(link, material_map, resolver)
        )
        for link in urdf.robot.links
    }

    children = {j.child_link for j in joints}
    roots = [name for name in links if name not in children]
    if not roots:
        raise ValueError("URDF has no root link (the joint graph contains a cycle)")
    # A well-formed URDF has exactly one root; extra roots mean disconnected
    # sub-trees, which we attach to the first so nothing is silently dropped.
    root = getattr(urdf, "base_link", None) or roots[0]
    if root not in links:
        root = roots[0]

    return RobotModel(
        name=str(getattr(urdf.robot, "name", None) or "robot"),
        links=links,
        joints=joints,
        root_link=root,
    )
