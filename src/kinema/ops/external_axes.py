"""Add and remove external axes: a track, a rotary base, a positioner, a tool spindle.

The maths and the placeholder shapes are in ``rig/external_axes.py``. This is the
part that edits the armature, and most of it is keeping everything else on the rig
true while it does.

An axis is one more joint bone, set up by the same code the importer uses, so it
has a slider, a hold pin, a velocity check, keys and bake like any imported joint.
It is held by default: an external axis is positioned by hand and the arm reaches
from wherever it stands, which is what ``hold_external_axes`` does for a rail the
description already had.

What adding one has to keep right:

* **The stack, for an axis under the robot.** A new one goes in at the bottom, under
  any already there, and is placed from the footing of the lowest -- the robot and the
  axes it already rides are carried like one bigger robot. Its base and offset can put
  that stack somewhere other than where it stood. Everything on it then moves by the
  same rigid transform: its bones, the axes and their placeholders, the TCP, the IK
  target and swivel, the base link's meshes, the waypoints. The transform is stored on
  the axis, so removing it moves them all back.
* **The TCP, for an axis on the tool.** The TCP moves onto the axis at its offset,
  and the offset and parent it had are stored, so removing it puts them back.
* **Joint indices.** Blender orders pose bones by hierarchy, so an axis under the
  robot arrives at index 0 and every joint after it moves up one. Waypoints store
  joint vectors and the keyframable IK tip stores an index, so both are remapped by
  bone name, keys included.
* **The solver.** A rig with an external axis is no longer the robot in its
  description, so PyRoki is given a model written from the rig's own bones -- see
  ``solver/rig_model.py``.
"""

from __future__ import annotations

import math

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, StringProperty
from bpy.types import Operator
from mathutils import Matrix, Vector

from .. import handlers
from ..rig import builder
from ..rig import external_axes as ext
from ..solver import manager
from ..ui import axis_preview
from ..ui.panel import active_rig
from . import attach, deferral, ik, pose
from .waypoints import MAX_STORED_DOF

#: One material for every placeholder, so they read as stand-ins at a glance.
PLACEHOLDER_MATERIAL = "Kinema External Axis"
PLACEHOLDER_COLOR = (0.85, 0.45, 0.1, 1.0)

MOUNT_LABELS = {
    "BEFORE": "Under the Robot",
    "AFTER": "On the Tool",
    "EXTERNAL": "Standalone",
}
MOUNT_ICONS = {"BEFORE": "TRIA_DOWN_BAR", "AFTER": "TRIA_UP_BAR", "EXTERNAL": "OBJECT_ORIGIN"}


# --------------------------------------------------------------------------
# small conversions
# --------------------------------------------------------------------------
def _np4(matrix) -> np.ndarray:
    return np.array([[matrix[r][c] for c in range(4)] for r in range(4)])


def _matrix(array) -> Matrix:
    return Matrix([[float(v) for v in row] for row in np.asarray(array)])


def _flat(array) -> list[float]:
    return [float(v) for v in np.asarray(array).flatten()]


def _stored(owner, key) -> np.ndarray | None:
    values = owner.get(key) if owner is not None else None
    if values is None or len(values) != 16:
        return None
    return np.array([float(v) for v in values]).reshape(4, 4)


def _is_identity(array, tolerance: float = 1e-9) -> bool:
    return bool(np.allclose(array, np.eye(4), atol=tolerance))


# --------------------------------------------------------------------------
# reading the rig
# --------------------------------------------------------------------------
def axis_bones(rig) -> list:
    """The rig's external axes, as pose bones in rig order."""
    return [pb for pb in builder.joint_bones(rig) if builder.PROP_EXTERNAL in pb.bone]


def robot_base(rig) -> np.ndarray:
    """Where the robot's base stands at rest, in armature space."""
    stored = _stored(rig, builder.PROP_ROBOT_BASE)
    return stored if stored is not None else np.eye(4)


def _mount_of(bone) -> str | None:
    return bone.get(builder.PROP_EXTERNAL) if bone is not None else None


def _bottom_axis(rig):
    """The lowest axis under the robot -- the one standing on Root -- or None.

    Axes under the robot form one chain up from Root, and each new one goes in at the
    bottom, so the robot and the axes already there ride it like one bigger robot.
    """
    root = rig.data.bones.get(builder.ROOT_BONE)
    if root is None:
        return None
    return next((c for c in root.children if _mount_of(c) == "BEFORE"), None)


def stack_base(rig) -> np.ndarray:
    """Where the robot stands, with any axes already under it, in armature space.

    The robot's own base until an axis goes under it; then the footing of the lowest
    axis, just under its rail or base. It is what a new axis under the robot is placed
    from, and what "the current robot base" means in the dialog.
    """
    bottom = _bottom_axis(rig)
    if bottom is None:
        return robot_base(rig)
    frame = builder.link_frame_of(bottom)
    footing = _stored(bottom, builder.PROP_EXTERNAL_FOOTING)
    return _np4(frame) @ (footing if footing is not None else np.eye(4))


def _subtree(bone) -> list[str]:
    return [bone.name, *(child.name for child in bone.children_recursive)]


def _riding_the_robot(rig) -> list[str]:
    """Every bone that moves when a new axis goes in under the robot and moves it.

    The robot's own bones and the axes already under it, and the controls hung off
    Root to steer it -- the IK target and the swivel -- which move with it so the pose
    it holds is unchanged. Not standalone axes: those are the cell.
    """
    names: list[str] = []
    for child in rig.data.bones[builder.ROOT_BONE].children:
        if _mount_of(child) == "EXTERNAL":
            continue
        names += [name for name in _subtree(child) if name not in names]
    return names


def _on_the_stack(rig, obj) -> bool:
    """Whether an object on Root belongs to what a new axis under the robot carries.

    The robot's base-link meshes, and the fixed part of the lowest axis already under
    it. Not a standalone axis's placeholder, and nothing parented there by hand.
    """
    if builder.PROP_LINK_NAME in obj:
        return True
    owner = rig.data.bones.get(str(obj.get(builder.PROP_EXTERNAL_MESH, "")))
    return _mount_of(owner) == "BEFORE"


def _joint_names(rig) -> list[str]:
    return [pb.name for pb in builder.joint_bones(rig)]


def _bone_length(rig) -> float:
    """The display length the rig's joints were built with."""
    joints = builder.joint_bones(rig)
    return float(joints[0].bone.length) if joints else 0.05


def _unique_bone_name(rig, wanted: str) -> str:
    bones = rig.data.bones
    name, index = wanted, 1
    while name in bones:
        name = f"{wanted}.{index:03d}"
        index += 1
    return name


def _unique_link_name(rig, wanted: str) -> str:
    """A link name no joint on the rig already moves -- PyRoki keys links by name."""
    taken = {str(b.get(builder.PROP_CHILD_LINK)) for b in rig.data.bones}
    taken.add(str(rig.get(builder.PROP_TCP_LINK, "")))
    name, index = wanted, 1
    while name in taken:
        name = f"{wanted}.{index:03d}"
        index += 1
    return name


def robot_reach(rig) -> float:
    """The robot's extent at rest, in metres: what placeholder sizes scale with."""
    heads = [np.array(b.head_local, dtype=float) for b in rig.data.bones]
    tails = [np.array(b.tail_local, dtype=float) for b in rig.data.bones]
    points = np.array(heads + tails)
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))) if len(points) else 1.0


# --------------------------------------------------------------------------
# objects on bones
# --------------------------------------------------------------------------
def _rest_placement(rig, obj) -> Matrix:
    """Where a bone-parented object sits with the rig at rest, in armature space."""
    bone = rig.data.bones[obj.parent_bone]
    return (
        bone.matrix_local
        @ Matrix.Translation((0.0, bone.length, 0.0))
        @ obj.matrix_parent_inverse
        @ obj.matrix_basis
    )


def _place_on_bone(rig, obj, bone_name: str, rest: Matrix) -> None:
    """Parent ``obj`` to a bone so that, at rest, it sits at ``rest``.

    The link-mesh convention: parented to the tail, identity parent inverse, the
    whole placement in the basis. A link mesh's recorded rest follows it, so Reset
    Meshes puts it back here rather than where it was before.
    """
    bone = rig.data.bones[bone_name]
    obj.parent = rig
    obj.parent_type = "BONE"
    obj.parent_bone = bone_name
    obj.matrix_parent_inverse = Matrix.Identity(4)
    obj.matrix_basis = (
        bone.matrix_local @ Matrix.Translation((0.0, bone.length, 0.0))
    ).inverted_safe() @ rest
    if builder.PROP_LINK_REST in obj:
        obj[builder.PROP_LINK_REST] = _flat(_np4(obj.matrix_basis))


def _placeholder_material():
    material = bpy.data.materials.get(PLACEHOLDER_MATERIAL)
    if material is not None:
        return material
    material = bpy.data.materials.new(PLACEHOLDER_MATERIAL)
    material.diffuse_color = PLACEHOLDER_COLOR
    if material.node_tree is None and hasattr(material, "use_nodes"):
        material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF") if material.node_tree else None
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = PLACEHOLDER_COLOR
    return material


def _visual_collection(rig):
    """The collection the robot's link meshes are in, so placeholders sit with them."""
    for mesh in builder.link_meshes(rig):
        if mesh.get(builder.PROP_GEOMETRY_KIND, builder.KIND_VISUAL) == builder.KIND_VISUAL:
            if mesh.users_collection:
                return mesh.users_collection[0]
    if rig.users_collection:
        return rig.users_collection[0]
    return bpy.context.scene.collection


def _make_placeholder(rig, axis_name: str, part: str, mesh: ext.Mesh, bone_name: str,
                      rest: Matrix):
    vertices, faces = mesh
    data = bpy.data.meshes.new(f"{axis_name} {part}")
    data.from_pydata(vertices, [], faces)
    data.update()
    data.materials.append(_placeholder_material())
    obj = bpy.data.objects.new(f"{axis_name} {part}", data)
    _visual_collection(rig).objects.link(obj)
    _place_on_bone(rig, obj, bone_name, rest)
    obj[builder.PROP_EXTERNAL_MESH] = axis_name
    # Locked like a link mesh: it is part of the rig, placed by it.
    obj.lock_location = (True, True, True)
    obj.lock_rotation = (True, True, True)
    obj.lock_rotation_w = True
    obj.lock_scale = (True, True, True)
    return obj


def placeholders(rig, axis_name: str) -> list:
    return [o for o in rig.children if o.get(builder.PROP_EXTERNAL_MESH) == axis_name]


def _rides_the_robot(obj) -> bool:
    """Whether a bone-parented object is Kinema's to move with the robot.

    A link mesh is, and so is an axis placeholder -- an axis's own are removed with
    it, so any other found on it belongs to an axis further up. Anything else was
    parented by hand.
    """
    return builder.PROP_LINK_NAME in obj or builder.PROP_EXTERNAL_MESH in obj


def _remove_objects(objects) -> None:
    for obj in objects:
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if isinstance(data, bpy.types.Mesh) and data.users == 0:
            bpy.data.meshes.remove(data)


# --------------------------------------------------------------------------
# keeping indices, waypoints and the TCP right
# --------------------------------------------------------------------------
def _remap_joint_indices(rig, before: list[str], after: list[str]) -> None:
    """Carry joint vectors and the IK tip across a change in the rig's joints.

    By name: an axis under the robot arrives at index 0, and a stored index that
    kept its number would now name a different joint.
    """
    for waypoint in rig.kinema_waypoints:
        if waypoint.dof <= 0:
            continue
        values = dict(zip(before, list(waypoint.q)[: waypoint.dof], strict=False))
        q = [float(values.get(name, 0.0)) for name in after]
        waypoint.dof = len(q)
        waypoint.q = q + [0.0] * (MAX_STORED_DOF - len(q))

    def remap(index: int) -> int:
        if 0 <= index < len(before) and before[index] in after:
            return after.index(before[index])
        return -1

    rig.kinema_ik_tip = remap(int(rig.kinema_ik_tip))
    for container in ik.own_fcurve_containers(rig):
        for curve in container:
            if curve.data_path != ik.TIP_PATH:
                continue
            for point in curve.keyframe_points:
                value = float(remap(int(round(point.co[1]))))
                point.co[1] = value
                point.handle_left[1] = value
                point.handle_right[1] = value
            curve.update()

    # Found for the chain as it was; its width may not even match any more.
    for key in (ik.PROP_SOLUTIONS, ik.PROP_SOLUTIONS_DOF, ik.PROP_SOLUTIONS_GOAL):
        if key in rig:
            del rig[key]


def _move_waypoints(rig, shift: np.ndarray) -> None:
    """Move every taught pose with the robot, marker and stored copy alike."""
    moved = _matrix(shift)
    for waypoint in rig.kinema_waypoints:
        # Row-major, the way ops/waypoints.py stores it.
        stored = np.array([float(v) for v in waypoint.pose]).reshape(4, 4)
        waypoint.pose = _flat(shift @ stored)
        marker = waypoint.marker
        if marker is not None and marker.parent == rig:
            marker.matrix_basis = moved @ marker.matrix_basis


def _tcp_offset(rig, tcp) -> list[float]:
    """The TCP's offset from its parent's zero -- its mounting flange -- as it stands.

    Read off the bones rather than the panel's fields, which can have been edited
    without Update TCP being pressed.
    """
    flange = builder.tool_zero_frame(tcp.parent)
    if flange is None:
        return [*rig.kinema_tcp_offset, *rig.kinema_tcp_rpy]
    offset = flange.inverted_safe() @ tcp.matrix_local @ builder.BONE_TO_TOOL
    return [*offset.translation, *offset.to_euler("XYZ")]


def _place_tcp(rig, bone_name: str, offset, rpy) -> None:
    """Put the TCP on ``bone_name`` at a tool offset, through Set TCP itself."""
    rig.kinema_tcp_offset = tuple(float(v) for v in offset)
    rig.kinema_tcp_rpy = tuple(float(v) for v in rpy)
    bpy.ops.kinema.set_tcp(bone=bone_name)


def _assign_to_fk(rig, bone_name: str) -> None:
    bone = rig.data.bones[bone_name]
    for collection in list(bone.collections):
        collection.unassign(bone)
    for collection in rig.data.collections_all:
        if collection.name == builder.COLLECTION_FK:
            collection.assign(bone)


def _move_edit_bone(edit_bone, moved: Matrix) -> None:
    """Move a bone rigidly, roll included.

    Not ``EditBone.transform``: it corrects the roll with ``matrix @ z_axis``, and a
    4x4 times a 3-vector in mathutils is a *point*, so the translation lands in a
    direction. Any move that both turns and shifts left every bone rolled -- 1.5
    degrees on a 0.3 rad turn with a 0.2 m lift, measured. The matrix setter keeps
    the length and takes head, direction and roll together.
    """
    edit_bone.matrix = moved @ edit_bone.matrix


class _EditMode:
    """Open the rig's armature for editing, and always close it again.

    Closed back into Pose mode if that is where the rig was: the button is in the
    sidebar an animator uses while posing, and being dropped into Object mode by it
    would be a surprise.
    """

    def __init__(self, rig):
        self.rig = rig

    def __enter__(self):
        bpy.context.view_layer.objects.active = self.rig
        self.previous_mode = self.rig.mode
        if self.rig.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.mode_set(mode="EDIT")
        return self.rig.data.edit_bones

    def __exit__(self, *exc):
        bpy.ops.object.mode_set(mode="OBJECT")
        if self.previous_mode == "POSE":
            bpy.ops.object.mode_set(mode="POSE")
        return False


# --------------------------------------------------------------------------
# add
# --------------------------------------------------------------------------
def placement(rig, spec: ext.AxisSpec) -> tuple[str, np.ndarray, np.ndarray]:
    """``(parent bone, joint frame, robot shift)`` for ``spec`` on ``rig``, all at rest.

    An axis under the robot goes in at the bottom of the stack, on Root, placed from
    :func:`stack_base`; one on the tool from the tool frame; a standalone one from the
    robot's base. The shift is what moves the robot, identity unless under it. Raises
    ValueError when the rig cannot take the axis.
    """
    bones = rig.data.bones
    if builder.ROOT_BONE not in bones:
        raise ValueError("This rig has no Root bone")
    shift = np.eye(4)
    if spec.mount == "AFTER":
        tcp = bones.get(rig.get(builder.PROP_TCP_BONE) or builder.TCP_BONE)
        if tcp is None or tcp.parent is None:
            raise ValueError("An axis on the tool needs a TCP; create one first")
        parent_name = tcp.parent.name
        reference = _np4(tcp.matrix_local @ builder.BONE_TO_TOOL)
    elif spec.mount == "BEFORE":
        parent_name = builder.ROOT_BONE
        reference = stack_base(rig)
        shift = ext.robot_shift(spec, reference)
    else:
        parent_name = builder.ROOT_BONE
        reference = robot_base(rig)
    return parent_name, ext.base_frame(spec, reference), shift


def add_axis(rig, spec: ext.AxisSpec) -> str:
    """Add ``spec`` to ``rig``. Returns the new bone's name; raises ValueError."""
    problems = spec.problems()
    if problems:
        raise ValueError("; ".join(problems).capitalize())
    parent_name, frame, shift = placement(rig, spec)
    bones = rig.data.bones

    before = _joint_names(rig)
    if len(rig.kinema_waypoints) and len(before) + 1 > MAX_STORED_DOF:
        raise ValueError(f"Waypoints can store at most {MAX_STORED_DOF} joints")

    name = _unique_bone_name(rig, spec.name.strip())
    link = _unique_link_name(rig, f"{name}_link")
    length = _bone_length(rig)
    base = robot_base(rig)
    tcp_name = rig.get(builder.PROP_TCP_BONE) or builder.TCP_BONE
    head, y_direction, z_reference = ext.bone_placement(frame, spec.axis)

    # Everything that reads Bone data is read here: none of it is live in Edit mode.
    moves_robot = spec.mount == "BEFORE" and not _is_identity(shift)
    riding = _riding_the_robot(rig) if moves_robot else []
    to_reparent = []
    carried = []
    stacked = []
    if spec.mount == "BEFORE":
        # What stood on Root now stands on the new axis: the lowest axis already
        # under the robot, or the robot's first joints when there is none.
        to_reparent = [
            child.name for child in bones[parent_name].children
            if builder.PROP_JOINT_NAME in child and _mount_of(child) != "EXTERNAL"
        ]
        # So do the objects on Root that belong to it: the robot's base-link meshes,
        # or the lowest axis's rail or base.
        carried = [
            (obj, _matrix(shift) @ _rest_placement(rig, obj))
            for obj in rig.children
            if obj.parent_type == "BONE" and obj.parent_bone == parent_name
            and _on_the_stack(rig, obj)
        ]
        stacked = [b.name for b in bones if _mount_of(b) == "BEFORE"]
    previous_tcp = None
    if spec.mount == "AFTER":
        previous_tcp = (parent_name, _tcp_offset(rig, bones[tcp_name]))

    previous_active = bpy.context.view_layer.objects.active
    with handlers.suspended():
        with _EditMode(rig) as edit_bones:
            if moves_robot:
                moved = _matrix(shift)
                for bone_name in riding:
                    _move_edit_bone(edit_bones[bone_name], moved)
            bone = edit_bones.new(name)
            bone.head = Vector(head)
            bone.tail = Vector(head + y_direction * length)
            bone.align_roll(Vector(z_reference))  # align_roll takes the desired Z
            bone.parent = edit_bones[parent_name]
            bone.use_connect = False
            for child in to_reparent:
                edit_bones[child].parent = bone

        pose_bone = rig.pose.bones[name]
        builder.configure_joint_bone(
            pose_bone, revolute=spec.kind == "ROTARY", limits=spec.limits
        )
        builder.write_joint_props(
            pose_bone.bone, joint_name=name, joint_type=spec.joint_type, axis=spec.axis,
            limits=spec.limits, velocity=spec.speed, child_link=link, link_frame=frame,
        )
        pose_bone.bone[builder.PROP_EXTERNAL] = spec.mount
        pose_bone.kinema_ik_hold = spec.hold
        _assign_to_fk(rig, name)

        if spec.mount == "BEFORE":
            pose_bone.bone[builder.PROP_EXTERNAL_SHIFT] = _flat(shift)
            pose_bone.bone[builder.PROP_EXTERNAL_FOOTING] = _flat(ext.footing(spec))
            rig[builder.PROP_ROBOT_BASE] = _flat(shift @ base)
            for obj, rest in carried:
                _place_on_bone(rig, obj, name, rest)
            if moves_robot:
                # The axes above were stored in a frame this one has just moved.
                for other_name in stacked:
                    other = rig.data.bones[other_name]
                    inner = _stored(other, builder.PROP_EXTERNAL_SHIFT)
                    if inner is not None:
                        other[builder.PROP_EXTERNAL_SHIFT] = _flat(
                            shift @ inner @ np.linalg.inv(shift)
                        )
                _move_waypoints(rig, shift)

        if spec.placeholder:
            rest = rig.data.bones[name].matrix_local.copy()
            static, moving = ext.placeholder(spec)
            _make_placeholder(rig, name, "base", static, parent_name, rest)
            _make_placeholder(rig, name, "moving", moving, name, rest)

        _remap_joint_indices(rig, before, _joint_names(rig))

        if spec.mount == "AFTER":
            pose_bone.bone[builder.PROP_EXTERNAL_TCP_PARENT] = previous_tcp[0]
            pose_bone.bone[builder.PROP_EXTERNAL_TCP_OFFSET] = previous_tcp[1]
            _place_tcp(rig, name, spec.offset_location, spec.offset_rotation)
            pose.snap_goal_to_tcp(rig)

    if previous_active is not None:
        bpy.context.view_layer.objects.active = previous_active
    manager.invalidate(rig.name)
    handlers.reset(rig.name)
    bpy.context.view_layer.update()
    return name


# --------------------------------------------------------------------------
# remove
# --------------------------------------------------------------------------
def remove_axis(rig, name: str) -> None:
    """Take an external axis off ``rig``, putting back what adding it changed.

    Everything is read off the bones by name before Edit mode: leaving it rebuilds
    ``rig.data.bones``, and a Bone held across that points at freed memory.
    """
    bones = rig.data.bones
    bone = bones.get(name)
    if bone is None or builder.PROP_EXTERNAL not in bone:
        raise ValueError(f"'{name}' is not an external axis on this rig")
    mount = bone[builder.PROP_EXTERNAL]
    parent_name = bone.parent.name if bone.parent is not None else builder.ROOT_BONE
    before = _joint_names(rig)

    shift = _stored(bone, builder.PROP_EXTERNAL_SHIFT) if mount == "BEFORE" else None
    shift = shift if shift is not None else np.eye(4)
    undo = np.linalg.inv(shift)
    moves_robot = not _is_identity(shift)
    riding = []
    if moves_robot:
        riding = [child.name for child in bone.children_recursive]
        for child in bones[builder.ROOT_BONE].children:
            if child.name != name and _mount_of(child) not in ("BEFORE", "EXTERNAL"):
                riding += [n for n in _subtree(child) if n not in riding]
    children = [child.name for child in bone.children]

    own = placeholders(rig, name)
    on_bone = [
        obj for obj in rig.children
        if obj.parent_type == "BONE" and obj.parent_bone == name and obj not in own
    ]
    attached = [obj for obj in on_bone if builder.PROP_ATTACHMENT in obj]
    # Only what rode the robot onto the axis goes back with it: the robot's own link
    # meshes, and other axes' placeholders. Anything parented here by hand never
    # moved with the robot, so it stays where it stands, as an attachment does.
    reseat = [
        (obj, (_matrix(undo) if _rides_the_robot(obj) else Matrix.Identity(4))
         @ _rest_placement(rig, obj))
        for obj in on_bone if obj not in attached
    ]

    tcp_name = rig.get(builder.PROP_TCP_BONE) or builder.TCP_BONE
    tcp = bones.get(tcp_name)
    tcp_rides = mount == "AFTER" and tcp is not None and tcp.parent is not None \
        and tcp.parent.name == name
    stored_parent = str(bone.get(builder.PROP_EXTERNAL_TCP_PARENT, parent_name))
    stored_offset = [float(v) for v in bone.get(builder.PROP_EXTERNAL_TCP_OFFSET, [0.0] * 6)]
    # Axes stacked on this one on the tool remember it as where the TCP was; they
    # inherit what it remembered instead, so removing them later still lands.
    stacked = [
        other.name for other in bones
        if other.get(builder.PROP_EXTERNAL_TCP_PARENT) == name
    ]
    # Axes under the robot and above this one stored their shift in a frame this
    # one had already moved; taking it away changes that frame, so theirs follow.
    conjugate = [
        other.name for other in bone.children_recursive if _mount_of(other) == "BEFORE"
    ]

    for obj in attached:
        attach.detach(obj)
    _remove_objects(own)
    # Before the bone goes, not after: an object left naming a deleted bone is an
    # error in the depsgraph. Safe this early because the bone it moves to is below
    # the axis, and nothing below it moves.
    for obj, rest in reseat:
        _place_on_bone(rig, obj, parent_name, rest)

    previous_active = bpy.context.view_layer.objects.active
    with handlers.suspended():
        with _EditMode(rig) as edit_bones:
            parent = edit_bones.get(parent_name)
            for child in children:
                edit_bones[child].parent = parent
            if moves_robot:
                moved = _matrix(undo)
                for bone_name in riding:
                    _move_edit_bone(edit_bones[bone_name], moved)
            edit_bones.remove(edit_bones[name])

        for other_name in stacked:
            other = rig.data.bones[other_name]
            other[builder.PROP_EXTERNAL_TCP_PARENT] = stored_parent
            other[builder.PROP_EXTERNAL_TCP_OFFSET] = stored_offset
        if moves_robot:
            rig[builder.PROP_ROBOT_BASE] = _flat(undo @ robot_base(rig))
            for other_name in conjugate:
                other = rig.data.bones[other_name]
                inner = _stored(other, builder.PROP_EXTERNAL_SHIFT)
                if inner is not None:
                    other[builder.PROP_EXTERNAL_SHIFT] = _flat(undo @ inner @ shift)
            _move_waypoints(rig, undo)

        path = f'pose.bones["{bpy.utils.escape_identifier(name)}"]'
        for container in ik.own_fcurve_containers(rig):
            for curve in [c for c in container if c.data_path.startswith(path)]:
                container.remove(curve)

        _remap_joint_indices(rig, before, _joint_names(rig))

        if tcp_rides:
            target = stored_parent if stored_parent in rig.pose.bones else parent_name
            _place_tcp(rig, target, stored_offset[:3], stored_offset[3:])
            pose.snap_goal_to_tcp(rig)

    if previous_active is not None and previous_active.name in bpy.data.objects:
        bpy.context.view_layer.objects.active = previous_active
    manager.invalidate(rig.name)
    handlers.reset(rig.name)
    bpy.context.view_layer.update()


# --------------------------------------------------------------------------
# operators
# --------------------------------------------------------------------------
PRESET_ITEMS = [
    *((key, preset.label, preset.description) for key, preset in ext.PRESETS.items()),
    ("CUSTOM", "Custom", "Keep the fields as they are and set everything yourself"),
]
MOUNT_ITEMS = [
    ("BEFORE", MOUNT_LABELS["BEFORE"],
     "The robot rides it: a track or a rotary base. It goes in under any axis already "
     "there, placed from the bottom of the robot as it stands now"),
    ("AFTER", MOUNT_LABELS["AFTER"],
     "It rides the tool: a spindle or a slide. Placed from the tool frame, and the "
     "TCP moves onto it"),
    ("EXTERNAL", MOUNT_LABELS["EXTERNAL"],
     "It stands on its own beside the robot, like a positioner. Placed from the "
     "robot's base; attach the workpiece to it in the Bones panel"),
]
KIND_ITEMS = [
    ("LINEAR", "Linear", "Slides along its direction"),
    ("ROTARY", "Rotary", "Turns about its direction"),
]
DIRECTION_ITEMS = [
    (key, key, f"Along the {key.lstrip('-')} axis of the base placement"
     + (", reversed" if key.startswith("-") else ""))
    for key in ext.DIRECTIONS
]


def _apply_preset(operator, context) -> None:
    """Fill the dialog from the chosen preset, sized to the robot."""
    if operator.preset == "CUSTOM":
        return
    spec = ext.from_preset(operator.preset)
    operator.axis_name = spec.name
    operator.mount = spec.mount
    operator.kind = spec.kind
    operator.direction = spec.direction
    operator.continuous = spec.continuous
    if spec.kind == "LINEAR":
        operator.lower_distance, operator.upper_distance = spec.lower, spec.upper
        operator.speed_distance = spec.velocity or 0.0
    else:
        operator.lower_angle, operator.upper_angle = spec.lower, spec.upper
        operator.speed_angle = spec.velocity or 0.0
    operator.base_location = spec.base_location
    operator.base_rotation = spec.base_rotation
    operator.offset_location = spec.offset_location
    operator.offset_rotation = spec.offset_rotation
    operator.hold = spec.hold
    _apply_size(operator, context)


# Their redo panels are an edit in progress: see ops/deferral.py.
deferral.adjustable("KINEMA_OT_add_external_axis", "KINEMA_OT_remove_external_axis")


def forget() -> None:
    """Drop every draft: a new file shares no rig with the old one."""
    _drafts.clear()


#: rig -> the dialog's fields as it last drew them. A click in the viewport to
#: look at the preview closes the dialog, and Blender keeps an operator's fields only
#: when it finishes -- so without this, reopening it started again from the preset.
#: Keyed by the rig's identity, not its name: a rig deleted and another imported
#: under its name, or another file's rig of the same name, is a different robot.
_drafts: dict[int | str, dict] = {}


def _remember(operator, rig) -> None:
    _drafts[manager.rig_identity(rig)] = {
        name: (tuple(value) if hasattr(value, "__len__") and not isinstance(value, str)
               else value)
        for name, value in ((name, getattr(operator, name)) for name in _fields())
    }


def _restore(operator, draft: dict) -> None:
    """Put a draft back. Preset and mount first: setting them refills other fields."""
    first, last = ("preset", "mount"), ("size",)
    middle = tuple(name for name in _fields() if name not in first + last)
    for name in (*first, *middle, *last):
        if name in draft:
            setattr(operator, name, draft[name])


def _fields() -> tuple[str, ...]:
    return tuple(KINEMA_OT_add_external_axis.__annotations__)


def _ghost_sources(rig) -> list:
    """What the preview's ghost is made of: everything an axis under the robot carries.

    The robot's visual meshes and the placeholders of the axes already under it.
    """
    visual = [
        mesh for mesh in builder.link_meshes(rig)
        if mesh.get(builder.PROP_GEOMETRY_KIND, builder.KIND_VISUAL) == builder.KIND_VISUAL
    ]
    stacked = [
        obj for obj in rig.children
        if _mount_of(rig.data.bones.get(str(obj.get(builder.PROP_EXTERNAL_MESH, ""))))
        == "BEFORE"
    ]
    return visual + stacked


def _apply_size(operator, context) -> None:
    rig = active_rig(context)
    operator.size = ext.default_size(operator.mount, robot_reach(rig) if rig else 1.0)
    _refresh_preview(operator, context)


def _refresh_preview(operator, context) -> None:
    """Redraw the preview from the fields as they change, a slider drag included.

    The dialog's draw() only runs when Blender rebuilds the popup, which it does not
    do mid-drag; a property's update callback does run on every step. Reads the
    fields and nothing else, and hands the preview the dialog it is tracking, in
    case ``operator`` here is not the same Python object as the dialog's.
    """
    dialog = axis_preview._state.get("operator")
    if dialog is None:
        return
    rig = active_rig(context)
    # Kept here too, not only in draw(): a change the popup is not rebuilt after
    # would otherwise be missing from the draft if the dialog closed next.
    if rig is not None:
        _remember(operator, rig)
    spec = KINEMA_OT_add_external_axis.spec(operator)
    try:
        placed = placement(rig, spec) if rig is not None and not spec.problems() else None
    except ValueError:
        placed = None
    axis_preview.show(dialog, spec, placed)


class KINEMA_OT_add_external_axis(Operator):
    """Add a rail, turntable, positioner or tool spindle the description does not have"""

    bl_idname = "kinema.add_external_axis"
    bl_label = "Add External Axis"
    bl_options = {"REGISTER", "UNDO"}

    preset: EnumProperty(
        name="Preset", items=PRESET_ITEMS, default="LINEAR_TRACK", update=_apply_preset,
        description="Start from a common axis; every field can still be changed",
    )
    axis_name: StringProperty(
        name="Name", default="track",
        description="The joint's name, which is also its bone's and its slider's",
    )
    mount: EnumProperty(
        name="Mount", items=MOUNT_ITEMS, default="BEFORE", update=_apply_size,
        description="What the axis carries, and what it is placed from",
    )
    kind: EnumProperty(
        name="Motion", items=KIND_ITEMS, default="LINEAR", update=_refresh_preview,
    )
    direction: EnumProperty(
        name="Direction", items=DIRECTION_ITEMS, default="X", update=_refresh_preview,
        description="The axis it moves along or turns about, in its base placement",
    )
    continuous: BoolProperty(
        name="Continuous", default=False, update=_refresh_preview,
        description="No end stops: it can turn any number of times",
    )
    lower_distance: FloatProperty(
        name="Lower", default=-1.5, unit="LENGTH", update=_refresh_preview,
    )
    upper_distance: FloatProperty(
        name="Upper", default=1.5, unit="LENGTH", update=_refresh_preview,
    )
    lower_angle: FloatProperty(
        name="Lower", default=-math.pi, subtype="ANGLE", update=_refresh_preview,
    )
    upper_angle: FloatProperty(
        name="Upper", default=math.pi, subtype="ANGLE", update=_refresh_preview,
    )
    speed_distance: FloatProperty(
        name="Top Speed", default=1.0, min=0.0, unit="VELOCITY",
        description="Fastest it may move; the Velocity Limits check warns past it. Zero for none",
    )
    speed_angle: FloatProperty(
        name="Top Speed", default=math.radians(90.0), min=0.0, subtype="ANGLE",
        description=(
            "Fastest it may turn, per second; the Velocity Limits check warns past it. "
            "Zero for none"
        ),
    )
    base_location: FloatVectorProperty(
        name="Location", size=3, subtype="TRANSLATION", unit="LENGTH", update=_refresh_preview,
        description=(
            "Where the axis sits, from the current robot base -- under any axis already "
            "beneath the robot -- or from the tool frame, on the tool"
        ),
    )
    base_rotation: FloatVectorProperty(
        name="Rotation", size=3, subtype="EULER", update=_refresh_preview,
        description="Its orientation there, as roll, pitch and yaw about fixed X, Y and Z",
    )
    offset_location: FloatVectorProperty(
        name="Location", size=3, subtype="TRANSLATION", unit="LENGTH", update=_refresh_preview,
        description=(
            "Where what it carries sits on its moving part: the current robot base, "
            "or the TCP. Anything but zero moves it there"
        ),
    )
    offset_rotation: FloatVectorProperty(
        name="Rotation", size=3, subtype="EULER", update=_refresh_preview,
        description="That orientation, as roll, pitch and yaw about fixed X, Y and Z",
    )
    hold: BoolProperty(
        name="Hold for IK", default=True,
        description="Position it by hand and let IK solve the arm from where it stands",
    )
    placeholder: BoolProperty(
        name="Placeholder", default=True, update=_refresh_preview,
        description="Generate simple geometry for it, for when there is no model of the real one",
    )
    size: FloatProperty(
        name="Size", default=0.3, min=0.001, unit="LENGTH", update=_refresh_preview,
        description="The placeholder's width: a rail's, or a turntable's diameter",
    )

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return active_rig(context) is not None

    def invoke(self, context, event):
        rig = active_rig(context)
        draft = _drafts.get(manager.rig_identity(rig)) if rig is not None else None
        if draft:
            _restore(self, draft)
        else:
            _apply_preset(self, context)
        axis_preview.start(self, context, rig, _ghost_sources(rig) if rig else [])
        return context.window_manager.invoke_props_dialog(self, width=460)

    def cancel(self, context):
        axis_preview.stop()

    def spec(self) -> ext.AxisSpec:
        linear = self.kind == "LINEAR"
        return ext.AxisSpec(
            name=self.axis_name,
            mount=self.mount,
            kind=self.kind,
            direction=self.direction,
            lower=self.lower_distance if linear else self.lower_angle,
            upper=self.upper_distance if linear else self.upper_angle,
            continuous=self.continuous and not linear,
            velocity=self.speed_distance if linear else self.speed_angle,
            base_location=tuple(self.base_location),
            base_rotation=tuple(self.base_rotation),
            offset_location=tuple(self.offset_location),
            offset_rotation=tuple(self.offset_rotation),
            hold=self.hold,
            placeholder=self.placeholder,
            size=self.size,
        )

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.prop(self, "preset")
        layout.prop(self, "axis_name")
        layout.prop(self, "mount")
        layout.separator()

        linear = self.kind == "LINEAR"
        layout.row().prop(self, "kind", expand=True)
        layout.prop(self, "direction")
        if not linear:
            layout.prop(self, "continuous")
        if linear or not self.continuous:
            column = layout.column(align=True)
            column.prop(self, "lower_distance" if linear else "lower_angle")
            column.prop(self, "upper_distance" if linear else "upper_angle")
        layout.prop(self, "speed_distance" if linear else "speed_angle")

        on_tool = self.mount == "AFTER"
        box = layout.box()
        box.label(
            text="External axis base: calculated from the current "
            + ("tool frame" if on_tool else "robot base"),
            icon="EMPTY_AXIS",
        )
        column = box.column(align=True)
        column.prop(self, "base_location")
        column.prop(self, "base_rotation")
        if self.mount != "EXTERNAL":
            box = layout.box()
            box.label(
                text="External axis offset: calculated from the current "
                + ("TCP" if on_tool else "robot base"),
                icon="ORIENTATION_PARENT",
            )
            column = box.column(align=True)
            column.prop(self, "offset_location")
            column.prop(self, "offset_rotation")
        spec = self.spec()
        rig = active_rig(context)
        if rig is not None:
            _remember(self, rig)
        try:
            placed = placement(rig, spec) if rig is not None and not spec.problems() else None
        except ValueError:
            placed = None
        axis_preview.show(self, spec, placed)
        if self.mount == "BEFORE" and placed is not None:
            moved = placed[2]
            if not _is_identity(moved, 1e-6):
                x, y, z = moved[:3, 3]
                layout.label(
                    text=f"The robot moves onto it by {x:+.3f}, {y:+.3f}, {z:+.3f} m",
                    icon="INFO",
                )

        layout.separator()
        layout.prop(self, "hold")
        layout.prop(self, "placeholder")
        if self.placeholder:
            layout.prop(self, "size")
        for problem in spec.problems():
            layout.label(text=problem.capitalize(), icon="ERROR")

    def execute(self, context):
        axis_preview.stop()
        rig = active_rig(context)
        spec = self.spec()
        # Before the edit: add_axis ends on a depsgraph update, and live IK would
        # compile on it.
        started = deferral.defer(rig)
        try:
            name = add_axis(rig, spec)
        except ValueError as exc:
            # Refused before anything changed, so nothing to wait for.
            if started:
                deferral.withdraw(rig)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        message = f"Added '{name}' {MOUNT_LABELS[spec.mount].lower()}"
        if spec.hold:
            message += ", held for IK"
        self.report({"INFO"}, message)
        return {"FINISHED"}


class KINEMA_OT_remove_external_axis(Operator):
    """Remove an external axis, putting back the robot, the TCP and the waypoints"""

    bl_idname = "kinema.remove_external_axis"
    bl_label = "Remove External Axis"
    bl_options = {"REGISTER", "UNDO"}

    bone: StringProperty(name="Axis", default="", options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return active_rig(context) is not None

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(
            self, event, message=f"Remove '{self.bone}', its keys and its placeholder?"
        )

    def execute(self, context):
        rig = active_rig(context)
        started = deferral.defer(rig)
        try:
            remove_axis(rig, self.bone)
        except ValueError as exc:
            # Refused before anything changed, so nothing to wait for.
            if started:
                deferral.withdraw(rig)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Removed '{self.bone}'")
        return {"FINISHED"}


classes = (
    KINEMA_OT_add_external_axis,
    KINEMA_OT_remove_external_axis,
)
