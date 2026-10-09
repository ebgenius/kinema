"""Collision in the scene: the robot's capsules, and the obstacles in its cell.

**Fit Capsules** reads the meshes on each joint bone, the link's imported
collision meshes where it has them and what the link shows otherwise, an
attached tool included, and fits capsules around them (``rig/collision.py``).
They are kept on the rig, in each bone's frame, with a fingerprint of the
meshes they came from, so the panel can say when the meshes have changed
since.

**Obstacles** are the objects in the *Kinema Obstacles* collection. Each is a
box by default, its own bounds placed where it stands; or a sphere, the size
of its largest half-extent; or a floor, everything below the plane through its
origin square to its Z axis. Read on every frame, so one that moves is
measured where it is.

The Motion Check measures every capsule against every obstacle on every frame
of the job, and says where a link comes closest, or hits.
"""

from __future__ import annotations

import hashlib
import time

import bpy
import numpy as np
from bpy.props import (
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    StringProperty,
)
from bpy.types import Operator, PropertyGroup

from ..rig import builder
from ..rig import collision as geometry
from ..ui.panel import active_rig

#: The collection whose objects are obstacles.
OBSTACLES = "Kinema Obstacles"

SHAPES = (
    (geometry.KIND_BOX, "Box", "Its own bounds, placed where it stands"),
    (geometry.KIND_SPHERE, "Sphere", "A ball the size of its largest half-extent"),
    (
        geometry.KIND_FLOOR,
        "Floor",
        "Everything below the plane through its origin, square to its Z",
    ),
)


class KinemaCapsule(PropertyGroup):
    """One capsule of the robot, in the frame of the bone it rides on."""

    bone: StringProperty(name="Bone")
    a: FloatVectorProperty(name="End A", size=3, subtype="TRANSLATION", unit="LENGTH")
    b: FloatVectorProperty(name="End B", size=3, subtype="TRANSLATION", unit="LENGTH")
    radius: FloatProperty(name="Radius", unit="LENGTH", min=0.0)


# --------------------------------------------------------------------------
# the robot
# --------------------------------------------------------------------------
def link_meshes(rig) -> dict[str, list]:
    """Bone name -> the objects riding on it that stand for its link.

    The joint bones, and the TCP, which carries a tool attached to it. Not Root:
    the base stands where it stands, on the floor it would always be touching.

    The imported collision meshes, where the bone has them: they are the
    description's own envelope. Otherwise everything else on it, what the link
    shows and any tool attached, a collection attached as its instance.
    """
    bones = {pose_bone.name for pose_bone in builder.joint_bones(rig)}
    tcp = rig.get(builder.PROP_TCP_BONE) or builder.TCP_BONE
    if tcp in rig.pose.bones:
        bones.add(tcp)
    found: dict[str, dict[str, list]] = {}
    for obj in rig.children:
        if obj.parent_type != "BONE" or obj.parent_bone not in bones:
            continue
        instances = obj.type == "EMPTY" and obj.instance_type == "COLLECTION"
        if obj.type != "MESH" and not (instances and obj.instance_collection is not None):
            continue
        kind = obj.get(builder.PROP_GEOMETRY_KIND, "")
        found.setdefault(obj.parent_bone, {}).setdefault(kind, []).append(obj)
    chosen = {}
    for bone, kinds in found.items():
        others = [
            obj for kind, objects in kinds.items() if kind != builder.KIND_COLLISION
            for obj in objects
        ]
        chosen[bone] = kinds.get(builder.KIND_COLLISION) or others
    return chosen


def bone_points(context, rig, bone_name: str, objects) -> np.ndarray:
    """``objects`` as points in the bone's own frame: their vertices, as evaluated,
    and points spread over their faces (``rig/collision.py``)."""
    depsgraph = context.evaluated_depsgraph_get()
    to_bone = np.linalg.inv(np.array(rig.matrix_world @ rig.pose.bones[bone_name].matrix))
    wanted = {obj.name: obj for obj in objects}
    seen = set()
    parts = []
    # Instances as well as objects: an attached collection is an empty that
    # instances it, and what it shows exists only as the depsgraph's instances.
    for instance in depsgraph.object_instances:
        owner = instance.parent if instance.is_instance else instance.object
        if owner is None or owner.original.name not in wanted:
            continue
        seen.add(owner.original.name)
        if instance.object.type != "MESH":
            continue
        mesh = instance.object.to_mesh()
        try:
            parts.append(_surface(mesh, to_bone @ np.array(instance.matrix_world)))
        finally:
            instance.object.to_mesh_clear()
    # A hidden object isn't evaluated, and the collision meshes are hidden: its
    # own mesh, unmodified, where it stands.
    for name, obj in wanted.items():
        if name not in seen and obj.type == "MESH":
            parts.append(_surface(obj.data, to_bone @ np.array(obj.matrix_world)))
    return np.concatenate(parts) if parts else np.zeros((0, 3))


def _surface(mesh, matrix: np.ndarray) -> np.ndarray:
    """A mesh's vertices and points over its faces, through ``matrix``."""
    flat = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", flat)
    local = np.c_[flat.reshape(-1, 3), np.ones(len(flat) // 3)]
    vertices = (matrix @ local.T).T[:, :3]
    mesh.calc_loop_triangles()
    triangles = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", triangles)
    return geometry.surface_points(vertices, triangles.reshape(-1, 3))


def source_signature(rig) -> str:
    """A fingerprint of the meshes the capsules are fitted to: their shapes, and where
    they sit on their bones.

    The vertices themselves, not only how many: a mesh reshaped in Edit Mode
    keeps its count and its transform, and the capsules would no longer bound it.
    """
    digest = hashlib.sha1()
    for bone, objects in sorted(link_meshes(rig).items()):
        for obj in sorted(objects, key=lambda o: o.name):
            digest.update(f"{bone}|{obj.name}|".encode())
            # Where it sits on the bone, from what puts it there and nothing
            # posed: read off the posed bone, float32 differs in the sixth
            # decimal from one pose to the next, and playing the job would read
            # as a change.
            placed = obj.matrix_parent_inverse @ obj.matrix_basis
            digest.update(np.round(np.array(placed), 5).tobytes())
            if obj.type == "MESH":
                _hash_mesh(digest, obj)
            else:
                for part in sorted(obj.instance_collection.all_objects, key=lambda o: o.name):
                    digest.update(f"{part.name}|".encode())
                    digest.update(np.round(np.array(part.matrix_world), 6).tobytes())
                    if part.type == "MESH":
                        _hash_mesh(digest, part)
    return digest.hexdigest()


def _hash_mesh(digest, obj) -> None:
    digest.update(_mesh_digest(obj.data))


#: Bumped whenever Blender reports a mesh's geometry has changed, by
#: :func:`note_updates`. Hashing every vertex of a robot takes about 10 ms, and
#: the panel asks on every redraw, through playback too; a mesh's hash is only
#: worked out again once some mesh has changed.
_generation = 0
#: Mesh name -> (the generation it was hashed in, its hash).
_mesh_hashes: dict[str, tuple[int, bytes]] = {}


def _mesh_digest(mesh) -> bytes:
    cached = _mesh_hashes.get(mesh.name)
    if cached is not None and cached[0] == _generation:
        return cached[1]
    flat = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", flat)
    # To a tenth of a millimetre: float noise in a mesh untouched isn't a change.
    found = hashlib.sha1(np.round(flat, 4).tobytes()).digest()
    _mesh_hashes[mesh.name] = (_generation, found)
    return found


def note_updates(depsgraph) -> None:
    """A depsgraph update: whether any mesh's geometry changed in it.

    Moving, posing and playing don't change a mesh's geometry, only where it is.
    """
    global _generation
    for update in depsgraph.updates:
        if update.is_updated_geometry and isinstance(update.id, bpy.types.Mesh):
            _generation += 1
            return


def forget() -> None:
    """A new file: its meshes share names, not shapes, with the last one's."""
    global _generation
    _generation += 1
    _mesh_hashes.clear()


def capsules_of(rig) -> list[geometry.Capsule]:
    """The capsules last fitted for ``rig``, on bones it still has."""
    bones = rig.pose.bones
    return [
        geometry.Capsule(item.bone, np.array(item.a), np.array(item.b), float(item.radius))
        for item in getattr(rig, "kinema_capsules", ())
        if item.bone in bones
    ]


def capsule_ends(rig, capsules) -> tuple[np.ndarray, np.ndarray]:
    """Each capsule's ends in world space, as the rig stands now: two (C, 3) arrays."""
    if not capsules:
        return np.zeros((0, 3)), np.zeros((0, 3))
    placed = {}
    for capsule in capsules:
        if capsule.bone not in placed:
            placed[capsule.bone] = np.array(rig.matrix_world @ rig.pose.bones[capsule.bone].matrix)
    a = np.array([placed[c.bone][:3, :3] @ c.a + placed[c.bone][:3, 3] for c in capsules])
    b = np.array([placed[c.bone][:3, :3] @ c.b + placed[c.bone][:3, 3] for c in capsules])
    return a, b


def closest_approach(rig):
    """The stored check's closest approach to an obstacle over the job, if it is current.

    None when no move was measured, or when the waypoints have changed since the
    check: then it describes another job.
    """
    from .waypoints import job_signature

    if rig.kinema_motion_check_job != job_signature(rig):
        return None
    return min(
        (check for check in rig.kinema_motion_check if check.clearance_obstacle),
        key=lambda check: check.clearance,
        default=None,
    )


def capsules_stale(rig) -> bool:
    """Whether the meshes have changed since the capsules were fitted."""
    return bool(len(rig.kinema_capsules)) and rig.kinema_capsules_source != source_signature(rig)


class KINEMA_OT_fit_capsules(Operator):
    bl_idname = "kinema.fit_capsules"
    bl_label = "Fit Capsules"
    bl_description = (
        "Fit capsules around each link's meshes: what the Motion Check measures the "
        "robot by against the obstacles. Again after changing a link's meshes or tool"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return active_rig(context) is not None

    def execute(self, context: bpy.types.Context) -> set[str]:
        rig = active_rig(context)
        meshes = link_meshes(rig)
        if not meshes:
            self.report({"ERROR"}, "No meshes ride on this robot's joints to fit capsules to")
            return {"CANCELLED"}
        started = time.perf_counter()
        fitted = []
        window = context.window_manager
        window.progress_begin(0, len(meshes))
        try:
            for done, (bone, objects) in enumerate(sorted(meshes.items())):
                points = bone_points(context, rig, bone, objects)
                fitted += [(bone, a, b, radius) for a, b, radius in geometry.fit(points)]
                window.progress_update(done + 1)
        finally:
            window.progress_end()
        rig.kinema_capsules.clear()
        for bone, a, b, radius in fitted:
            item = rig.kinema_capsules.add()
            item.bone, item.a, item.b, item.radius = bone, tuple(a), tuple(b), radius
        rig.kinema_capsules_source = source_signature(rig)
        _redraw(context)
        self.report(
            {"INFO"},
            f"Fitted {len(fitted)} capsules on {len(meshes)} links in "
            f"{time.perf_counter() - started:.1f} s",
        )
        return {"FINISHED"}


# --------------------------------------------------------------------------
# obstacles
# --------------------------------------------------------------------------
def obstacle_collection(scene, create: bool = False):
    """The scene's obstacles collection, made and linked under the scene if asked to."""
    collection = bpy.data.collections.get(OBSTACLES)
    if collection is None:
        if not create:
            return None
        collection = bpy.data.collections.new(OBSTACLES)
    if create and collection not in scene.collection.children_recursive:
        scene.collection.children.link(collection)
    return collection


def obstacle_objects(scene) -> list:
    collection = obstacle_collection(scene)
    if collection is None:
        return []
    in_scene = set(scene.objects)
    return [
        obj for obj in collection.all_objects
        if obj in in_scene and not builder.is_kinema_rig(obj)
    ]


def obstacles(context) -> list[geometry.Obstacle]:
    """The scene's obstacles, as they stand on the current frame."""
    depsgraph = context.evaluated_depsgraph_get()
    return [obstacle_of(obj.evaluated_get(depsgraph)) for obj in obstacle_objects(context.scene)]


def obstacle_of(obj) -> geometry.Obstacle:
    """``obj`` as the shape its Obstacle setting makes it."""
    kind = getattr(obj, "kinema_obstacle_shape", geometry.KIND_BOX)
    world = obj.matrix_world
    location, rotation, _ = world.decompose()
    linear = np.array(world.to_3x3())
    turned = np.eye(4)
    turned[:3, :3] = np.array(rotation.to_matrix())
    if kind == geometry.KIND_FLOOR:
        # The plane its own X and Y span, however a parent has sheared them.
        normal = np.cross(linear[:, 0], linear[:, 1])
        if np.linalg.norm(normal) > 0.0:
            turned[:3, :3] = _frame_with_z(normal / np.linalg.norm(normal))
        turned[:3, 3] = location
        return geometry.Obstacle(obj.name, kind, turned, np.zeros(3))
    low, high = _local_bounds(obj)
    local_half = (high - low) / 2.0
    centre = np.array(world @ _vector((low + high) / 2.0))
    turned[:3, 3] = centre
    if kind == geometry.KIND_SPHERE:
        # As far as its own surface reaches, however it is stretched.
        radius = float(np.linalg.norm(linear, 2) * local_half.max())
        return geometry.Obstacle(obj.name, kind, turned, np.array([radius]))
    # Square to its rotation, wide enough for every corner of what it is: a
    # parent scaled unevenly and turned shears the box, and the shear has to be
    # inside the box measured, not cut off.
    in_frame = np.array(rotation.to_matrix()).T @ linear
    half = np.abs(in_frame) @ local_half
    return geometry.Obstacle(obj.name, geometry.KIND_BOX, turned, half)


def _frame_with_z(z: np.ndarray) -> np.ndarray:
    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def _local_bounds(obj) -> tuple[np.ndarray, np.ndarray]:
    if obj.type == "EMPTY":
        size = obj.empty_display_size
        return np.full(3, -size), np.full(3, size)
    corners = np.array([tuple(corner) for corner in obj.bound_box])
    return corners.min(axis=0), corners.max(axis=0)


def _vector(values):
    from mathutils import Vector

    return Vector(tuple(float(v) for v in values))


def _candidates(context) -> list:
    """Selected objects that could stand in the robot's way: not a rig, nor anything on one."""
    return [
        obj
        for obj in context.selected_objects
        if not builder.is_kinema_rig(obj) and not builder.is_kinema_rig(obj.parent)
    ]


class KINEMA_OT_add_obstacles(Operator):
    bl_idname = "kinema.add_obstacles"
    bl_label = "Add Selected"
    bl_description = (
        "Make the selected objects obstacles the Motion Check measures the robot against"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return bool(_candidates(context))

    def execute(self, context: bpy.types.Context) -> set[str]:
        collection = obstacle_collection(context.scene, create=True)
        added = [obj for obj in _candidates(context) if obj.name not in collection.objects]
        for obj in added:
            collection.objects.link(obj)
        _redraw(context)
        self.report({"INFO"}, f"{len(added)} obstacle{'s' if len(added) != 1 else ''} added")
        return {"FINISHED"}


class KINEMA_OT_remove_obstacles(Operator):
    bl_idname = "kinema.remove_obstacles"
    bl_label = "Remove Selected"
    bl_description = "Stop measuring the robot against the selected objects"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        collection = obstacle_collection(context.scene)
        return collection is not None and any(
            obj.name in collection.objects for obj in context.selected_objects
        )

    def execute(self, context: bpy.types.Context) -> set[str]:
        collection = obstacle_collection(context.scene)
        removed = [obj for obj in context.selected_objects if obj.name in collection.objects]
        for obj in removed:
            # Kept in the scene: an object only in the obstacles would vanish with it.
            if len(obj.users_collection) == 1:
                context.scene.collection.objects.link(obj)
            collection.objects.unlink(obj)
        _redraw(context)
        self.report({"INFO"}, f"{len(removed)} obstacle{'s' if len(removed) != 1 else ''} removed")
        return {"FINISHED"}


def _redraw(context) -> None:
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type in {"VIEW_3D", "PROPERTIES"}:
                area.tag_redraw()


def register_props() -> None:
    bpy.types.Object.kinema_capsules = CollectionProperty(type=KinemaCapsule)
    bpy.types.Object.kinema_capsules_source = StringProperty(
        name="Capsules Source",
        description="Fingerprint of the meshes the capsules were fitted to",
        default="",
    )
    bpy.types.Object.kinema_obstacle_shape = EnumProperty(
        name="Obstacle Shape",
        description="What the Motion Check measures this obstacle as",
        items=SHAPES,
        default=geometry.KIND_BOX,
    )


def unregister_props() -> None:
    del bpy.types.Object.kinema_capsules
    del bpy.types.Object.kinema_capsules_source
    del bpy.types.Object.kinema_obstacle_shape


classes = (
    KinemaCapsule,
    KINEMA_OT_fit_capsules,
    KINEMA_OT_add_obstacles,
    KINEMA_OT_remove_obstacles,
)
