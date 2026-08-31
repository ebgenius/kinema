"""Build a Blender armature from a :class:`~.kinematics.RobotModel`.

The rig this produces is the whole point of Kinema, so the rules it follows are
worth stating plainly.

**One armature.** Not armatures parented to armatures. A robot is one rig.

**One bone per actuated joint, with local Y on the joint axis.** A Blender bone
rotates about its own Y, so aligning Y to the URDF axis makes a single rotation
channel *be* the joint value. Every other channel is locked. The bone therefore
cannot also point at its child link -- that is what a custom shape is for.
Because a bone's rest matrix composes with its parent exactly the way URDF
joint origins do, the resulting rest pose is URDF forward kinematics at q = 0,
to machine precision.

**Fixed joints get no bone.** They have no degree of freedom, so a bone for one
is inert clutter in an animator's channel box. Their transforms are folded into
whatever the fixed chain carries -- meshes and tool frames still land exactly
where the URDF says.

**Link meshes are parented rigidly to bones**, not skinned. Robot links are
rigid bodies: vertex weights and an armature modifier would be slower and would
invite deformation that no real robot does.

**Bones are sorted into collections** so an animator can hide the mechanism and
see only controls.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import bpy
import numpy as np
from mathutils import Matrix, Vector

from ..io import meshes as mesh_io
from . import widgets
from .kinematics import JointSpec, RobotModel

#: Bone collection names, nested under a single "Kinema" parent.
COLLECTION_ROOT = "Kinema"
COLLECTION_FK = "FK"
COLLECTION_IK = "IK"
COLLECTION_TCP = "TCP"
COLLECTION_MECHANISM = "Mechanism"

ROOT_BONE = "Root"
TCP_BONE = "TCP"

#: Bone axes -> tool axes, as a change of basis.
#:
#: A bone's +Y is always head-to-tail, so a tool frame cannot sit in a bone's
#: own axes: it rides one permuted, with the tool's Z on the bone's Y. The
#: importer picks this permutation when it builds the TCP from a link frame
#: (tail along link Z, roll to link X); naming it here is what keeps every
#: other route to the same bone agreeing with that one.
#:
#: Columns are the tool axes expressed in bone coordinates, so
#: ``tool = bone_matrix @ BONE_TO_TOOL``.
BONE_TO_TOOL = Matrix(((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))).to_4x4()

#: Name of the constraint enforcing a joint's URDF limits.
LIMIT_CONSTRAINT = "Kinema Joint Limit"

#: Custom properties written onto each joint bone, so the rig is
#: self-describing: the solver and the FK panel read the URDF facts back off
#: the armature instead of needing the original file.
PROP_JOINT_NAME = "kinema_joint"
PROP_JOINT_TYPE = "kinema_joint_type"
PROP_LOWER = "kinema_lower"
PROP_UPPER = "kinema_upper"
PROP_AXIS = "kinema_axis"
#: The URDF link this joint moves, and the constant transform from the bone's
#: rest frame to that link's rest frame. PyRoki targets *links*, but Kinema's
#: bones are oriented on joint axes, so the two frames differ by a fixed
#: rotation. Storing it means IK still works after the .blend is reopened
#: somewhere the original URDF is not available.
PROP_CHILD_LINK = "kinema_child_link"
PROP_LINK_CORRECTION = "kinema_link_correction"

#: Marks the armature object itself as a Kinema rig.
PROP_IS_RIG = "kinema_rig"
PROP_ROBOT_NAME = "kinema_robot"
PROP_TCP_BONE = "kinema_tcp_bone"
PROP_TCP_LINK = "kinema_tcp_link"
#: Where this rig came from, so the solver can reload the description it needs.
#: ("catalog", <robot_descriptions key>) or ("file", <path to the URDF>).
PROP_SOURCE_KIND = "kinema_source_kind"
PROP_SOURCE = "kinema_source"
#: IK state, all stored on the rig so it survives save/reload.
PROP_IK_BONE = "kinema_ik_bone"
PROP_IK_ENABLED = "kinema_ik_enabled"
PROP_SOLVER_MODE = "kinema_solver_mode"

#: Written onto an attached object: the bone it rides, and what it was copied
#: from. Link visuals are bone-parented too, so without a marker there is no
#: way to tell a user's gripper from the robot's own casing.
PROP_ATTACHMENT = "kinema_attachment"
PROP_ATTACH_SOURCE = "kinema_attach_source"


@dataclass
class RigBuildOptions:
    #: Bone display length in metres. None picks one from the robot's size.
    bone_length: float | None = None
    enforce_limits: bool = True
    import_visuals: bool = True
    create_tcp: bool = True
    #: Link whose frame becomes the TCP. None picks the deepest link.
    tcp_link: str | None = None
    collection_name: str | None = None


@dataclass
class RigBuildResult:
    armature_object: bpy.types.Object | None = None
    collection: bpy.types.Collection | None = None
    #: joint name -> bone name (identical today, but keep the indirection).
    joint_bones: dict[str, str] = field(default_factory=dict)
    mesh_objects: list[bpy.types.Object] = field(default_factory=list)
    tcp_link: str | None = None
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _to_matrix(array: np.ndarray) -> Matrix:
    return Matrix([[float(v) for v in row] for row in array])


def _np4(matrix) -> np.ndarray:
    """mathutils.Matrix -> a plain 4x4 NumPy array."""
    return np.array([[matrix[r][c] for c in range(4)] for r in range(4)])


def _auto_bone_length(model: RobotModel) -> float:
    """Pick a display length proportional to the robot, clamped to sane bounds.

    Purely cosmetic -- bone length does not enter the kinematics, because every
    bone's head is placed absolutely and no bones are connected.
    """
    origins = np.array([frame[:3, 3] for frame in model.link_frames().values()])
    if len(origins) < 2:
        return 0.05
    extent = float(np.linalg.norm(origins.max(axis=0) - origins.min(axis=0)))
    return float(np.clip(extent * 0.09, 0.01, 0.20))


def _deepest_link(model: RobotModel) -> str:
    """The link furthest from the root by joint count -- a good default TCP."""
    depth = {model.root_link: 0}
    by_parent = model.joints_by_parent()
    stack = [model.root_link]
    while stack:
        parent = stack.pop()
        for joint in by_parent.get(parent, ()):
            depth[joint.child_link] = depth[parent] + 1
            stack.append(joint.child_link)
    return max(depth, key=lambda name: depth[name])


def _ensure_object_mode() -> None:
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")


def _activate(obj: bpy.types.Object) -> None:
    view_layer = bpy.context.view_layer
    for other in view_layer.objects:
        other.select_set(False)
    obj.select_set(True)
    view_layer.objects.active = obj


# --------------------------------------------------------------------------
# bone construction
# --------------------------------------------------------------------------
def _create_bones(
    armature_object: bpy.types.Object,
    model: RobotModel,
    options: RigBuildOptions,
    result: RigBuildResult,
) -> None:
    joint_frames = model.joint_frames()
    link_frames = model.link_frames()
    owner_of_link = model.nearest_actuated_ancestor()
    length = options.bone_length or _auto_bone_length(model)

    _activate(armature_object)
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = armature_object.data.edit_bones

    # Root control at the robot base.
    root = edit_bones.new(ROOT_BONE)
    root.head = Vector((0.0, 0.0, 0.0))
    root.tail = Vector((0.0, 0.0, length * 1.5))
    root.roll = 0.0

    actuated = model.actuated_joints
    for joint in actuated:
        frame = joint_frames[joint.name]
        head = Vector(frame[:3, 3])
        # The joint axis, taken into world space. This becomes the bone's Y.
        axis_world = Vector(frame[:3, :3] @ joint.axis).normalized()

        bone = edit_bones.new(joint.name)
        bone.head = head
        bone.tail = head + axis_world * length
        # Roll only affects the bone's X/Z; point Z along the link's own Z so
        # the rig reads consistently with the URDF frames.
        bone.align_roll(Vector(frame[:3, 2]))
        result.joint_bones[joint.name] = bone.name

    # Parent each bone to the bone of its nearest actuated ancestor. Because
    # rest matrices are absolute, the relative offset Blender derives already
    # contains every intervening fixed joint.
    for joint in actuated:
        bone = edit_bones[result.joint_bones[joint.name]]
        owner = owner_of_link.get(joint.parent_link)
        parent_name = result.joint_bones.get(owner) if owner else None
        bone.parent = edit_bones[parent_name] if parent_name else root
        # Never connect: a connected bone's head is pinned to its parent's
        # tail, which would drag every joint origin to the wrong place.
        bone.use_connect = False

    # TCP marker at the tool frame.
    tcp_link = options.tcp_link or _deepest_link(model)
    if options.create_tcp and tcp_link in link_frames:
        frame = link_frames[tcp_link]
        tcp = edit_bones.new(TCP_BONE)
        tcp.head = Vector(frame[:3, 3])
        # Tool frames point along their own +Z by ROS convention.
        tcp.tail = Vector(frame[:3, 3]) + Vector(frame[:3, 2]).normalized() * length * 0.8
        tcp.align_roll(Vector(frame[:3, 0]))
        owner = owner_of_link.get(tcp_link)
        parent_name = result.joint_bones.get(owner) if owner else None
        tcp.parent = edit_bones[parent_name] if parent_name else root
        tcp.use_connect = False
        result.tcp_link = tcp_link

    bpy.ops.object.mode_set(mode="OBJECT")


def _setup_collections(armature: bpy.types.Armature, result: RigBuildResult) -> None:
    collections = armature.collections
    parent = collections.new(COLLECTION_ROOT)
    groups = {
        name: collections.new(name, parent=parent)
        for name in (COLLECTION_FK, COLLECTION_IK, COLLECTION_TCP, COLLECTION_MECHANISM)
    }

    for bone_name in result.joint_bones.values():
        groups[COLLECTION_FK].assign(armature.bones[bone_name])
    if ROOT_BONE in armature.bones:
        groups[COLLECTION_FK].assign(armature.bones[ROOT_BONE])
    if TCP_BONE in armature.bones:
        groups[COLLECTION_TCP].assign(armature.bones[TCP_BONE])

    # IK starts empty and hidden; M5 populates it.
    groups[COLLECTION_IK].is_visible = False
    groups[COLLECTION_MECHANISM].is_visible = False


def _setup_pose_bones(
    armature_object: bpy.types.Object,
    model: RobotModel,
    options: RigBuildOptions,
    result: RigBuildResult,
) -> None:
    pose = armature_object.pose
    joint_by_name = {j.name: j for j in model.joints}
    link_frames = model.link_frames()
    shapes = widgets.ensure_widgets()

    for joint_name, bone_name in result.joint_bones.items():
        joint: JointSpec = joint_by_name[joint_name]
        pose_bone = pose.bones[bone_name]

        # Y first, so the meaningful channel is the one an animator reaches for.
        pose_bone.rotation_mode = "YXZ"

        if joint.is_revolute:
            pose_bone.lock_location = (True, True, True)
            pose_bone.lock_rotation = (True, False, True)
            pose_bone.lock_scale = (True, True, True)
            pose_bone.custom_shape = shapes["revolute"]
        else:  # prismatic
            pose_bone.lock_location = (True, False, True)
            pose_bone.lock_rotation = (True, True, True)
            pose_bone.lock_scale = (True, True, True)
            pose_bone.custom_shape = shapes["prismatic"]

        pose_bone.use_custom_shape_bone_size = True

        # Self-describing rig: the FK panel and solver read these back rather
        # than requiring the original URDF alongside the .blend.
        bone = pose_bone.bone
        bone[PROP_JOINT_NAME] = joint.name
        bone[PROP_JOINT_TYPE] = joint.joint_type
        bone[PROP_AXIS] = [float(v) for v in joint.axis]
        if joint.has_limits:
            bone[PROP_LOWER] = float(joint.lower)
            bone[PROP_UPPER] = float(joint.upper)

        # bone rest frame -> URDF link rest frame, so the solver can turn a
        # bone-space goal into the link-space goal PyRoki expects.
        bone[PROP_CHILD_LINK] = joint.child_link
        correction = np.linalg.inv(_np4(bone.matrix_local)) @ link_frames[joint.child_link]
        bone[PROP_LINK_CORRECTION] = [float(v) for v in correction.flatten()]

        if options.enforce_limits and joint.has_limits:
            if joint.is_revolute:
                constraint = pose_bone.constraints.new("LIMIT_ROTATION")
                constraint.use_limit_y = True
                constraint.min_y = float(joint.lower)
                constraint.max_y = float(joint.upper)
            else:
                constraint = pose_bone.constraints.new("LIMIT_LOCATION")
                constraint.use_min_y = True
                constraint.use_max_y = True
                constraint.min_y = float(joint.lower)
                constraint.max_y = float(joint.upper)
            constraint.name = LIMIT_CONSTRAINT
            constraint.owner_space = "LOCAL"

    if ROOT_BONE in pose.bones:
        root = pose.bones[ROOT_BONE]
        root.rotation_mode = "QUATERNION"
        root.custom_shape = shapes["root"]
        root.use_custom_shape_bone_size = True

    if TCP_BONE in pose.bones:
        tcp = pose.bones[TCP_BONE]
        tcp.rotation_mode = "QUATERNION"
        tcp.custom_shape = shapes["tcp"]
        tcp.use_custom_shape_bone_size = True
        # The TCP marker reports where the tool *is*; it is not a control.
        # M5 adds a separate, movable IK target.
        tcp.lock_location = (True, True, True)
        tcp.lock_rotation = (True, True, True)
        tcp.lock_rotation_w = True
        tcp.lock_scale = (True, True, True)


# --------------------------------------------------------------------------
# visual meshes
# --------------------------------------------------------------------------
def _attach_visuals_iter(
    armature_object: bpy.types.Object,
    model: RobotModel,
    collection: bpy.types.Collection,
    result: RigBuildResult,
):
    """Load every link's visual geometry, yielding ``(done, total)`` per visual.

    This is the seam that lets an operator spread mesh loading over several
    modal ticks. Mesh import cannot leave the main thread -- ``bpy`` is not
    thread-safe -- so keeping Blender's event loop alive means returning to it
    periodically, and that means the work has to be resumable.
    """
    link_frames = model.link_frames()
    owner_of_link = model.nearest_actuated_ancestor()

    pending: list[tuple[bpy.types.Object, Matrix, str]] = []
    material_cache: dict[tuple, bpy.types.Material] = {}
    total = sum(len(link.visuals) for link in model.links.values())
    done = 0

    for link_name, link in model.links.items():
        if not link.visuals:
            continue
        owner_joint = owner_of_link.get(link_name)
        bone_name = result.joint_bones.get(owner_joint) if owner_joint else ROOT_BONE

        for index, visual in enumerate(link.visuals):
            label = visual.name or (
                link_name if index == 0 else f"{link_name}.{index:03d}"
            )
            done += 1
            try:
                if visual.mesh_path:
                    objects = mesh_io.load_mesh(
                        visual.mesh_path, collection=collection, name=label
                    )
                else:
                    kind, params = visual.primitive
                    objects = mesh_io.make_primitive(
                        kind, params, collection=collection, name=label
                    )
            except mesh_io.MeshLoadError as exc:
                result.warnings.append(str(exc))
                # Yield before continuing: a robot whose meshes are all missing
                # would otherwise run the whole loop inside one tick.
                yield done, total
                continue

            # world = link frame * <visual origin> * <mesh scale>
            world = (
                _to_matrix(link_frames[link_name])
                @ _to_matrix(visual.origin)
                @ Matrix.Diagonal(Vector(visual.scale)).to_4x4()
            )
            for obj in objects:
                if visual.material_color and not obj.data.materials:
                    obj.data.materials.append(
                        _link_material(visual.material_color, material_cache)
                    )
                pending.append((obj, world, bone_name))
                result.mesh_objects.append(obj)
            yield done, total

    # Two passes: assign every parent, refresh once, then place. Setting
    # matrix_world solves for the local transform using the parent's current
    # matrix, so the depsgraph has to be up to date -- but only once, not per
    # object, which matters on a 200-mesh humanoid.
    #
    # Deliberately not yielded: splitting this would mean one depsgraph
    # evaluation per chunk instead of one for the whole rig, which is the cost
    # the two-pass structure exists to avoid. It may overrun a caller's tick
    # budget, and that is the right trade.
    for obj, _, bone_name in pending:
        obj.parent = armature_object
        obj.parent_type = "BONE"
        obj.parent_bone = bone_name
    bpy.context.view_layer.update()
    for obj, world, _ in pending:
        obj.matrix_world = world


def _link_material(rgba, cache) -> bpy.types.Material:
    key = tuple(round(float(c), 4) for c in rgba)
    if key in cache:
        return cache[key]
    material = bpy.data.materials.new(name="kinema_link")
    if material.node_tree is None and hasattr(material, "use_nodes"):
        material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF") if material.node_tree else None
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = key
        if key[3] < 1.0:
            bsdf.inputs["Alpha"].default_value = key[3]
    cache[key] = material
    return material


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def build_rig(
    model: RobotModel,
    options: RigBuildOptions | None = None,
) -> RigBuildResult:
    """Create the armature and geometry for ``model`` in the current scene."""
    steps = build_rig_iter(model, options)
    while True:
        try:
            next(steps)
        except StopIteration as stop:
            return stop.value


def build_rig_iter(
    model: RobotModel,
    options: RigBuildOptions | None = None,
    result: RigBuildResult | None = None,
):
    """Generator form of :func:`build_rig`, for building across modal ticks.

    Yields ``(done, total)`` visuals and returns the :class:`RigBuildResult`
    (via ``StopIteration.value``). Pass ``result`` to keep a handle on the
    partial rig while it is being built, so a cancelled build can delete what
    already exists.

    Only mesh loading is chunked. Bone construction is not: it toggles Edit
    Mode, and leaving the file in Edit Mode across a tick -- while the operator
    is passing events through and the user still has control -- is a real
    corruption path. It touches no files and is fast.
    """
    options = options or RigBuildOptions()
    result = result if result is not None else RigBuildResult()

    _ensure_object_mode()

    name = options.collection_name or model.name
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    result.collection = collection

    armature = bpy.data.armatures.new(f"{name}_armature")
    armature_object = bpy.data.objects.new(name, armature)
    collection.objects.link(armature_object)
    result.armature_object = armature_object

    armature_object[PROP_IS_RIG] = True
    armature_object[PROP_ROBOT_NAME] = model.name

    _create_bones(armature_object, model, options, result)
    _setup_collections(armature, result)
    _setup_pose_bones(armature_object, model, options, result)

    if result.tcp_link:
        armature_object[PROP_TCP_BONE] = TCP_BONE
        armature_object[PROP_TCP_LINK] = result.tcp_link
        # Seed the panel's parent field and tool offset from what was actually
        # built, so a freshly imported rig arrives with them filled in.
        #
        # The offset is rarely zero, and that is the point: the tool frame is
        # the *deepest* link, which usually sits behind one or more fixed joints
        # from the last actuated one. Fixed joints get no bone, so that distance
        # is invisible on the rig -- recording it here is what puts it in front
        # of the user, and what lets "Update TCP" reproduce the import exactly
        # rather than snapping the marker back to the flange.
        tcp_bone = armature.bones.get(TCP_BONE)
        if tcp_bone is not None and tcp_bone.parent is not None:
            armature_object.kinema_tcp_parent = tcp_bone.parent.name
            flange = link_frame_of(tcp_bone.parent)
            if flange is not None:
                offset = flange.inverted_safe() @ tcp_bone.matrix_local @ BONE_TO_TOOL
                armature_object.kinema_tcp_offset = offset.translation
                armature_object.kinema_tcp_rpy = offset.to_euler("XYZ")

    if options.import_visuals:
        yield from _attach_visuals_iter(armature_object, model, collection, result)

    # Bones in front of geometry, which is what you want when the controls are
    # small dials buried inside a robot's own casing.
    armature.display_type = "OCTAHEDRAL"
    armature_object.show_in_front = True

    _activate(armature_object)
    return result


def discard_rig(result: RigBuildResult) -> None:
    """Delete a partially built rig, for a cancelled or failed build.

    Meshes first, then the armature, then the collection: removing a collection
    that still owns objects orphans them in ``bpy.data`` rather than freeing
    them.
    """
    for obj in list(result.mesh_objects):
        if obj is not None and obj.name in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
    armature_object = result.armature_object
    if armature_object is not None and armature_object.name in bpy.data.objects:
        armature = armature_object.data
        bpy.data.objects.remove(armature_object, do_unlink=True)
        if armature is not None and armature.users == 0:
            bpy.data.armatures.remove(armature)
    if result.collection is not None and result.collection.name in bpy.data.collections:
        bpy.data.collections.remove(result.collection)


def is_kinema_rig(obj: bpy.types.Object | None) -> bool:
    return bool(
        obj is not None and obj.type == "ARMATURE" and obj.get(PROP_IS_RIG, False)
    )


def joint_bones(armature_object: bpy.types.Object) -> list[bpy.types.PoseBone]:
    """Pose bones that correspond to actuated URDF joints, in rig order."""
    if not is_kinema_rig(armature_object):
        return []
    return [
        pose_bone
        for pose_bone in armature_object.pose.bones
        if PROP_JOINT_NAME in pose_bone.bone
    ]


def link_frame_of(bone) -> Matrix | None:
    """The rest frame of the URDF link a joint bone drives, in armature space.

    This is the frame a tool offset is meaningful in -- the flange, with its Z
    out of the face -- and it is not the bone's own frame, because the bone's
    axes are aligned to the joint axis instead. The constant transform between
    the two was recorded on the bone at build time, so this works on a .blend
    whose original description is long gone.

    Returns None for a bone that drives no link: Root, the TCP marker, the IK
    control.
    """
    stored = bone.get(PROP_LINK_CORRECTION) if bone is not None else None
    if stored is None or len(stored) != 16:
        return None
    correction = Matrix([[float(stored[row * 4 + col]) for col in range(4)] for row in range(4)])
    return bone.matrix_local @ correction


def bone_attachment(
    armature_object: bpy.types.Object, bone_name: str
) -> bpy.types.Object | None:
    """The object attached to ``bone_name``, or None.

    Derived by scanning the rig's children rather than stored in a list, so it
    survives save/reload and cannot go stale: if the user deletes an attachment
    or unparents it by hand, it simply stops being found. Same reasoning as the
    rest of the rig being self-describing.
    """
    if armature_object is None:
        return None
    for child in armature_object.children:
        if (
            child.parent_type == "BONE"
            and child.parent_bone == bone_name
            and PROP_ATTACHMENT in child
        ):
            return child
    return None


def attachments(armature_object: bpy.types.Object) -> list[bpy.types.Object]:
    """Every object attached to this rig's bones, in child order."""
    if armature_object is None:
        return []
    return [
        child
        for child in armature_object.children
        if child.parent_type == "BONE" and PROP_ATTACHMENT in child
    ]
