"""Read MuJoCo MJCF into Kinema's URDF-shaped intermediate representation.

Sixty of the 186 robots in the robot_descriptions catalog ship MJCF only, so
supporting it roughly doubles Kinema's reach. Producing a
:class:`~..rig.kinematics.RobotModel` -- the same IR the URDF path builds --
means the rig builder, solver and panels all work unchanged.

**Why not just bundle ``mujoco``?** It would be more faithful, but the wheel is
~50 MB per platform on top of an already 118 MB payload, for a parser we only
need the kinematic subset of. If ``mujoco`` happens to be importable it is not
used either -- one code path is easier to trust than two.

MJCF and URDF disagree in ways that matter:

* **Joints live on bodies, not between them.** A body may carry zero joints
  (rigidly attached), one, or several. URDF has exactly one joint per link, so
  multi-joint bodies are expanded into a chain of massless intermediate links.
* **A joint has its own ``pos`` inside the body.** MuJoCo applies a joint as
  T(p)·R(axis, q)·T(-p), pivoting about ``p`` while leaving the body frame
  where it was; a URDF joint's child frame *is* its pivot. Each joint's link is
  therefore placed on the pivot and a closing fixed joint carries the frame
  back by T(-p) onto MuJoCo's body frame.
* **Angles may be degrees.** ``<compiler angle="degree">`` is the MuJoCo
  default, the opposite of URDF's radians, and getting it wrong yields a robot
  whose joint limits are off by a factor of 57.

* **A floating base is a ``<freejoint>``.** Every legged model has one. It maps
  to a fixed attachment, because Blender's Root bone already *is* a freely
  movable control -- six IK joints there would be a control no animator wants.

Only the kinematic subset is read: bodies, joints, geoms, meshes and defaults.
Actuators, contacts, sensors and physics parameters are ignored by design.

Verified against MuJoCo's own ``mj_kinematics`` over all 60 MJCF descriptions
in the robot_descriptions catalog: 59 parse, and every body frame agrees to
8.9e-16. The remaining one (Cassie) uses a ``ball`` joint, which has no honest
single-axis bone equivalent.
"""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..rig.kinematics import (
    JointSpec,
    LinkSpec,
    RobotModel,
    VisualSpec,
    normalize,
)

#: MJCF joint type -> URDF joint type.
_JOINT_TYPES = {"hinge": "revolute", "slide": "prismatic"}
#: Attributes inherited through <default> classes, per element kind.
_DEFAULTABLE = ("joint", "geom", "mesh", "site", "body")


class MjcfError(RuntimeError):
    """An MJCF file could not be read as a kinematic tree."""


# --------------------------------------------------------------------------
# rotations
# --------------------------------------------------------------------------
def _quat_to_matrix(quat) -> np.ndarray:
    """MuJoCo quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    w, x, y, z = (float(v) for v in quat)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        return np.eye(3)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _axis_angle_to_matrix(axis, angle: float) -> np.ndarray:
    axis = normalize(np.asarray(axis, dtype=float))
    cos, sin = np.cos(angle), np.sin(angle)
    cross = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + sin * cross + (1 - cos) * (cross @ cross)


def _euler_to_matrix(angles, sequence: str) -> np.ndarray:
    """Intrinsic euler angles in the compiler's ``eulerseq`` order."""
    basis = {
        "x": lambda a: _axis_angle_to_matrix((1, 0, 0), a),
        "y": lambda a: _axis_angle_to_matrix((0, 1, 0), a),
        "z": lambda a: _axis_angle_to_matrix((0, 0, 1), a),
    }
    matrix = np.eye(3)
    for axis_name, angle in zip(sequence.lower(), angles, strict=False):
        matrix = matrix @ basis[axis_name](float(angle))
    return matrix


# --------------------------------------------------------------------------
# compiler options and defaults
# --------------------------------------------------------------------------
@dataclass
class Compiler:
    """The ``<compiler>`` settings that change how numbers are interpreted."""

    angle_in_degrees: bool = True  # MuJoCo's default, unlike URDF
    mesh_dir: str = ""
    euler_sequence: str = "xyz"

    def to_radians(self, value: float) -> float:
        return np.radians(value) if self.angle_in_degrees else value


@dataclass
class Defaults:
    """A ``<default>`` class: attributes inherited by elements naming it."""

    attributes: dict[str, dict[str, str]] = field(default_factory=dict)
    children: dict[str, Defaults] = field(default_factory=dict)

    def resolved(self, kind: str) -> dict[str, str]:
        return dict(self.attributes.get(kind, {}))

    def child(self, name: str) -> Defaults:
        return self.children.get(name, self)


def _merge_defaults(parent: Defaults, node: ElementTree.Element) -> Defaults:
    """Build a defaults class inheriting from ``parent``."""
    merged = Defaults(attributes=deepcopy(parent.attributes))
    for kind in _DEFAULTABLE:
        for element in node.findall(kind):
            merged.attributes.setdefault(kind, {})
            merged.attributes[kind].update(element.attrib)
    for child in node.findall("default"):
        name = child.get("class")
        if name:
            merged.children[name] = _merge_defaults(merged, child)
    # Nested classes remain reachable from the parent scope too.
    for name, value in parent.children.items():
        merged.children.setdefault(name, value)
    return merged


def _attributes(element, defaults: Defaults, kind: str, classes: dict) -> dict:
    """Element attributes with its ``class``'s defaults applied underneath."""
    class_name = element.get("class")
    source = classes.get(class_name, defaults) if class_name else defaults
    values = source.resolved(kind)
    values.update(element.attrib)
    return values


def _is_free(element: ElementTree.Element) -> bool:
    """True for a floating-base joint, in either of MJCF's two spellings."""
    return element.tag == "freejoint" or element.get("type", "hinge") == "free"


def _floats(text: str | None, count: int, default) -> np.ndarray:
    if not text:
        return np.array(default, dtype=float)
    parts = [float(v) for v in text.replace(",", " ").split()]
    if len(parts) < count:
        parts += list(np.asarray(default, dtype=float)[len(parts):])
    return np.array(parts[:count], dtype=float)


# --------------------------------------------------------------------------
# document loading
# --------------------------------------------------------------------------
def _load_tree(path: Path) -> ElementTree.Element:
    """Parse an MJCF file, splicing in every ``<include>`` recursively."""
    try:
        root = ElementTree.parse(path).getroot()
    except ElementTree.ParseError as exc:
        raise MjcfError(f"could not parse {path.name}: {exc}") from exc

    def splice(node: ElementTree.Element, base: Path) -> None:
        for include in list(node.findall("include")):
            target = include.get("file")
            index = list(node).index(include)
            node.remove(include)
            if not target:
                continue
            included = base / target
            if not included.is_file():
                raise MjcfError(f"included file not found: {included}")
            sub = _load_tree(included)
            for offset, child in enumerate(list(sub)):
                node.insert(index + offset, child)
        for child in node:
            splice(child, base)

    splice(root, path.parent)
    return root


def _read_compiler(root: ElementTree.Element) -> Compiler:
    compiler = Compiler()
    for element in root.findall("compiler"):
        angle = element.get("angle")
        if angle:
            compiler.angle_in_degrees = angle.strip().lower() != "radian"
        if element.get("meshdir"):
            compiler.mesh_dir = element.get("meshdir")
        if element.get("assetdir") and not element.get("meshdir"):
            compiler.mesh_dir = element.get("assetdir")
        if element.get("eulerseq"):
            compiler.euler_sequence = element.get("eulerseq")
    return compiler


def _read_meshes(root, compiler: Compiler, base: Path) -> dict[str, tuple[str, np.ndarray]]:
    """Asset name -> (absolute mesh path, scale)."""
    meshes: dict[str, tuple[str, np.ndarray]] = {}
    for asset in root.findall("asset"):
        for element in asset.findall("mesh"):
            filename = element.get("file")
            if not filename:
                continue
            name = element.get("name") or Path(filename).stem
            path = Path(compiler.mesh_dir) / filename if compiler.mesh_dir else Path(filename)
            if not path.is_absolute():
                path = base / path
            scale = _floats(element.get("scale"), 3, (1.0, 1.0, 1.0))
            meshes[name] = (str(path), scale)
    return meshes


def _frame_of(attributes: dict, compiler: Compiler) -> np.ndarray:
    """The 4x4 placement described by pos plus one of quat/euler/axisangle."""
    frame = np.eye(4)
    frame[:3, 3] = _floats(attributes.get("pos"), 3, (0.0, 0.0, 0.0))

    if attributes.get("quat"):
        frame[:3, :3] = _quat_to_matrix(_floats(attributes["quat"], 4, (1, 0, 0, 0)))
    elif attributes.get("euler"):
        angles = _floats(attributes["euler"], 3, (0, 0, 0))
        frame[:3, :3] = _euler_to_matrix(
            [compiler.to_radians(a) for a in angles], compiler.euler_sequence
        )
    elif attributes.get("axisangle"):
        values = _floats(attributes["axisangle"], 4, (0, 0, 1, 0))
        frame[:3, :3] = _axis_angle_to_matrix(values[:3], compiler.to_radians(values[3]))
    elif attributes.get("zaxis"):
        # Shorthand: rotate +Z onto the given direction.
        target = normalize(_floats(attributes["zaxis"], 3, (0, 0, 1)))
        axis = np.cross([0.0, 0.0, 1.0], target)
        norm = float(np.linalg.norm(axis))
        if norm > 1e-9:
            angle = float(np.arctan2(norm, float(np.dot([0, 0, 1], target))))
            frame[:3, :3] = _axis_angle_to_matrix(axis / norm, angle)
        elif target[2] < 0:
            frame[:3, :3] = _axis_angle_to_matrix((1, 0, 0), np.pi)
    return frame


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def _visual_for(attributes, compiler, meshes, offset: np.ndarray) -> VisualSpec | None:
    geom_type = attributes.get("type", "sphere")
    if attributes.get("mesh"):
        geom_type = "mesh"

    frame = offset @ _frame_of(attributes, compiler)
    rgba = attributes.get("rgba")
    color = tuple(_floats(rgba, 4, (0.7, 0.7, 0.7, 1.0))) if rgba else None
    size = _floats(attributes.get("size"), 3, (0.01, 0.01, 0.01))
    name = attributes.get("name")

    if geom_type == "mesh":
        entry = meshes.get(attributes.get("mesh", ""))
        if entry is None:
            return None
        path, scale = entry
        return VisualSpec(
            mesh_path=path, primitive=None, origin=frame,
            scale=scale, name=name, material_color=color,
        )

    if geom_type == "box":
        # MuJoCo sizes are half-extents; URDF box size is the full extent.
        primitive = ("box", {"size": [float(s) * 2.0 for s in size]})
    elif geom_type in ("cylinder", "capsule"):
        # A capsule's hemispherical caps are cosmetic at rig scale.
        primitive = ("cylinder", {"radius": float(size[0]), "length": float(size[1]) * 2.0})
    elif geom_type == "sphere":
        primitive = ("sphere", {"radius": float(size[0])})
    else:
        # plane, hfield, ellipsoid, sdf: not visual geometry Kinema draws.
        return None

    return VisualSpec(
        mesh_path=None, primitive=primitive, origin=frame,
        scale=np.ones(3), name=name, material_color=color,
    )


# --------------------------------------------------------------------------
# the tree walk
# --------------------------------------------------------------------------
class _Builder:
    def __init__(self, compiler: Compiler, meshes: dict, classes: dict, defaults: Defaults):
        self.compiler = compiler
        self.meshes = meshes
        self.classes = classes
        self.defaults = defaults
        self.links: dict[str, LinkSpec] = {}
        self.joints: list[JointSpec] = []
        self._anonymous = 0

    def unique(self, prefix: str) -> str:
        self._anonymous += 1
        return f"{prefix}_{self._anonymous}"

    def walk(self, body: ElementTree.Element, parent_link: str) -> None:
        attributes = _attributes(body, self.defaults, "body", self.classes)
        body_name = attributes.get("name") or self.unique("body")
        body_frame = _frame_of(attributes, self.compiler)

        joint_elements = [
            element
            for element in body.findall("joint") + body.findall("freejoint")
            if not _is_free(element)
        ]
        # A body with no joints is welded to its parent. So is a floating base:
        # a <freejoint> means the robot is not attached to the world, which
        # every legged MJCF model uses, and in Blender that is exactly what the
        # Root bone already is -- a freely movable control. Expanding it into
        # six IK joints instead would hand an animator a control they would only
        # ever want to move by hand.
        if not joint_elements:
            self.links[body_name] = LinkSpec(name=body_name)
            self.joints.append(
                JointSpec(
                    name=f"{body_name}_fixed", joint_type="fixed",
                    parent_link=parent_link, child_link=body_name,
                    origin=body_frame, axis=np.array([1.0, 0.0, 0.0]),
                )
            )
            self._add_geoms(body, body_name, np.eye(4))
            self._recurse(body, body_name)
            return

        # One URDF joint per MJCF joint. Several on one body become a chain of
        # massless intermediate links, which is how URDF spells a 2- or 3-DoF
        # joint anyway.
        #
        # MuJoCo applies a joint as T(p) · R(axis, q) · T(-p), pivoting about
        # ``p`` while leaving the body frame where it was. A URDF joint has no
        # such offset: its child frame *is* the pivot. So each joint's child
        # link is placed on the pivot, and a final fixed joint carries the
        # frame back by T(-p) to land exactly on MuJoCo's body frame. Without
        # that last step the body -- and every geom and child body under it --
        # sits one pivot offset away, which is invisible on the many robots
        # whose joints have p = 0 and badly wrong on those that do not.
        current_parent = parent_link
        accumulated = body_frame
        last = len(joint_elements) - 1
        last_pivot = np.eye(4)

        for index, element in enumerate(joint_elements):
            joint_attributes = _attributes(element, self.defaults, "joint", self.classes)
            joint_type = joint_attributes.get("type", "hinge")
            if joint_type not in _JOINT_TYPES:
                # "ball" is the realistic remainder: a 3-DoF spherical joint
                # with no single axis, so it has no honest one-bone equivalent.
                raise MjcfError(
                    f"joint type '{joint_type}' on body '{body_name}' is not supported "
                    f"(Kinema handles hinge, slide, and free bases)"
                )

            joint_name = joint_attributes.get("name") or self.unique("joint")
            # The joint pivots about its own pos inside the body, but a URDF
            # joint always pivots about the child origin -- so put the child
            # frame on the pivot and shift the body's contents back by it.
            pivot = np.eye(4)
            pivot[:3, 3] = _floats(joint_attributes.get("pos"), 3, (0.0, 0.0, 0.0))
            origin = accumulated @ pivot

            # Every joint gets its own link; the real body frame is added
            # afterwards by the closing fixed joint.
            child_name = f"{body_name}__{joint_name}"
            self.links[child_name] = LinkSpec(name=child_name)

            lower = upper = None
            if joint_attributes.get("range"):
                bounds = _floats(joint_attributes["range"], 2, (0.0, 0.0))
                limited = str(joint_attributes.get("limited", "auto")).lower()
                if limited not in ("false", "0"):
                    if joint_type == "hinge":
                        lower = self.compiler.to_radians(float(bounds[0]))
                        upper = self.compiler.to_radians(float(bounds[1]))
                    else:
                        lower, upper = float(bounds[0]), float(bounds[1])

            self.joints.append(
                JointSpec(
                    name=joint_name,
                    joint_type=_JOINT_TYPES[joint_type],
                    parent_link=current_parent,
                    child_link=child_name,
                    origin=origin,
                    axis=normalize(_floats(joint_attributes.get("axis"), 3, (0.0, 0.0, 1.0))),
                    lower=lower,
                    upper=upper,
                )
            )

            current_parent = child_name
            accumulated = np.linalg.inv(pivot)
            last_pivot = pivot
            if index == last:
                break

        # Close the chain: step back off the final pivot onto MuJoCo's body
        # frame, so geoms and child bodies attach exactly where MuJoCo puts them.
        self.links[body_name] = LinkSpec(name=body_name)
        self.joints.append(
            JointSpec(
                name=f"{body_name}_frame", joint_type="fixed",
                parent_link=current_parent, child_link=body_name,
                origin=np.linalg.inv(last_pivot), axis=np.array([1.0, 0.0, 0.0]),
            )
        )

        self._add_geoms(body, body_name, np.eye(4))
        self._recurse(body, body_name)

    def _add_geoms(self, body, link_name: str, offset: np.ndarray) -> None:
        link = self.links[link_name]
        for element in body.findall("geom"):
            attributes = _attributes(element, self.defaults, "geom", self.classes)
            # group 3+ is MuJoCo's convention for collision-only geometry.
            group = attributes.get("group")
            if group is not None and str(group).strip() in ("3", "4", "5"):
                continue
            visual = _visual_for(attributes, self.compiler, self.meshes, offset)
            if visual is not None:
                link.visuals.append(visual)

    def _recurse(self, body, link_name: str) -> None:
        # The child body's own frame is relative to this body's frame, which
        # after a jointed body means relative to the last pivot.
        for child in body.findall("body"):
            self.walk(child, link_name)


def model_from_mjcf(path: str | Path) -> RobotModel:
    """Parse an MJCF file into a :class:`RobotModel`."""
    path = Path(path)
    if not path.is_file():
        raise MjcfError(f"no such file: {path}")

    root = _load_tree(path)
    if root.tag != "mujoco":
        raise MjcfError(f"{path.name} is not an MJCF file (root tag <{root.tag}>)")

    compiler = _read_compiler(root)
    meshes = _read_meshes(root, compiler, path.parent)

    defaults = Defaults()
    classes: dict[str, Defaults] = {}
    for element in root.findall("default"):
        defaults = _merge_defaults(defaults, element)
        classes.update(defaults.children)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise MjcfError(f"{path.name} has no <worldbody>")

    builder = _Builder(compiler, meshes, classes, defaults)
    root_name = "world"
    builder.links[root_name] = LinkSpec(name=root_name)
    for body in worldbody.findall("body"):
        builder.walk(body, root_name)

    if not builder.joints:
        raise MjcfError(f"{path.name} contains no bodies under <worldbody>")

    return RobotModel(
        name=root.get("model") or path.stem,
        links=builder.links,
        joints=builder.joints,
        root_link=root_name,
    )
