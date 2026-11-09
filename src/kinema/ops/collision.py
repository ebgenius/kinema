"""Collision in the scene: the robot's capsules, and the obstacles in its cell.

**Fit Capsules** reads the meshes on each joint bone, the link's imported
collision meshes where it has them and what the link shows otherwise, an
attached tool included, and fits capsules around them (``rig/collision.py``).
They are kept on the rig, in each bone's frame, with a fingerprint of the
meshes they came from, so the panel can say when the meshes have changed
since.

**Obstacles** are the objects in the scene's *Kinema Obstacles* collection,
one per scene, directly: not those in a collection inside it. Each is a box
by default, its own bounds placed where it stands; or a sphere, the smallest
ball about its centre that holds it; or a floor, everything below the plane
through its origin square to its Z axis. Read on every frame, so one that
moves is measured where it is.

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
    (geometry.KIND_SPHERE, "Sphere", "The smallest ball about its centre that holds it"),
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

    For each link on the bone, its imported collision meshes where it has
    them: they are the description's own envelope. Otherwise what it shows.
    Per link, not per bone: links behind fixed joints ride on the bone of the
    joint before them, and one with no collision meshes is still there. And
    whatever else rides on the bone, not imported with a link: a tool
    attached, a collection attached as its instance.
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
        if kind not in (builder.KIND_VISUAL, builder.KIND_COLLISION):
            kind, link = "", ""
        else:
            link = obj.get(builder.PROP_LINK_NAME, "")
        found.setdefault(obj.parent_bone, {}).setdefault(link, {}).setdefault(kind, []).append(obj)
    chosen = {}
    for bone, links in found.items():
        envelope, added = [], []
        for _link, kinds in sorted(links.items(), key=lambda item: item[0]):
            added += kinds.get("", [])
            envelope += kinds.get(builder.KIND_COLLISION) or kinds.get(builder.KIND_VISUAL, [])
        chosen[bone] = envelope + added
    return chosen


def bone_points(context, rig, bone_name: str, objects) -> np.ndarray:
    """``objects`` as points in the bone's own frame: their vertices, as evaluated,
    and points spread over their faces (``rig/collision.py``).

    Placed on the bone from what puts them there, not through the posed bone:
    Blender keeps posed matrices in single precision, and going out to the
    world and back moves every point by micrometres that differ from one pose
    to the next. Enough to change how many capsules a link gets, so the same
    meshes fitted in two poses came out differently.
    """
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
        placed = _on_bone(rig, owner.original)
        if placed is None:
            matrix = to_bone @ np.array(instance.matrix_world)
        elif instance.is_instance:
            # The collection's objects where they stand in it, less its offset.
            local = np.array(instance.object.original.matrix_world, dtype=float)
            local[:3, 3] -= np.array(owner.original.instance_collection.instance_offset)
            matrix = placed @ local
        else:
            matrix = placed
        mesh = instance.object.to_mesh()
        try:
            parts.append(_surface(mesh, matrix))
        finally:
            instance.object.to_mesh_clear()
    # A hidden object isn't evaluated, and the collision meshes are hidden: its
    # own mesh, unmodified, where it stands.
    for name, obj in wanted.items():
        if name not in seen and obj.type == "MESH":
            placed = _on_bone(rig, obj)
            matrix = placed if placed is not None else to_bone @ np.array(obj.matrix_world)
            parts.append(_surface(obj.data, matrix))
    return np.concatenate(parts) if parts else np.zeros((0, 3))


def _on_bone(rig, obj) -> np.ndarray | None:
    """Where ``obj`` sits in its bone's frame, from what puts it there and nothing posed.

    Blender hangs a child of a bone from the bone's tail. None where a
    constraint moves it as well: then only where it stands says. A muted one
    moves nothing.
    """
    if obj.parent != rig or obj.parent_type != "BONE" or _constrained(obj):
        return None
    bone = rig.data.bones.get(obj.parent_bone)
    if bone is None:
        return None
    tail = np.eye(4)
    tail[1, 3] = bone.length
    return (
        tail
        @ np.array(obj.matrix_parent_inverse, dtype=float)
        @ np.array(obj.matrix_basis, dtype=float)
    )


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


#: A link's choices of how many capsules it gets: Auto, or a count.
CAPSULE_COUNTS = [
    (
        "AUTO",
        "Auto",
        "The fewest, up to four, within a quarter of the tightest fit by volume. Volume "
        "can keep one capsule round a squat link whose ends stand far out past it",
    ),
    *[
        (str(count), str(count), f"{count} capsule{'s' if count > 1 else ''} round this link")
        for count in range(1, geometry.MAX_PARTS + 1)
    ],
]


def capsule_count(rig, bone_name: str) -> int | None:
    """How many capsules a link was asked for, or None for Auto."""
    pose_bone = rig.pose.bones.get(bone_name)
    value = getattr(pose_bone, "kinema_capsule_count", "AUTO") if pose_bone else "AUTO"
    return None if value == "AUTO" else int(value)


def _on_capsule_count_changed(_pose_bone, context) -> None:
    # The capsules no longer follow it: the panels have to ask again.
    note_change()
    _redraw(context)


def source_signature(rig) -> str:
    """A fingerprint of the meshes the capsules are fitted to: their shapes, and where
    they sit on their bones.

    The vertices themselves, not only how many: a mesh reshaped in Edit Mode
    keeps its count and its transform, and the capsules would no longer bound it.
    """
    digest = hashlib.sha1()
    for bone, objects in sorted(link_meshes(rig).items()):
        # A child of a bone hangs from its tail: a bone made longer moves it.
        rest = rig.data.bones.get(bone)
        digest.update(f"{bone}|{round(rest.length, 6) if rest else ''}|".encode())
        # How many capsules the link was asked for, where it was: the fit
        # follows it. Auto adds nothing, so capsules fitted before there was a
        # count to set still read as fitted.
        parts = capsule_count(rig, bone)
        if parts is not None:
            digest.update(f"parts={parts}|".encode())
        for obj in sorted(objects, key=lambda o: o.name):
            digest.update(f"{obj.name}|{_what_moves(obj)}|".encode())
            # Where it sits on the bone, from what puts it there and nothing
            # posed: read off the posed bone, float32 differs in the sixth
            # decimal from one pose to the next, and playing the job would read
            # as a change.
            placed = obj.matrix_parent_inverse @ obj.matrix_basis
            digest.update(np.round(np.array(placed), 5).tobytes())
            # What would move it on the bone: a fit refuses it, so one added
            # since asks for Fit Again, which says why.
            digest.update(_keys(obj))
            digest.update(_drivers_and_constraints(obj))
            if obj.type == "MESH":
                _hash_mesh(digest, obj)
            else:
                # Where the collection is instanced from moves all of it.
                offset = obj.instance_collection.instance_offset
                digest.update(np.round(np.array(offset), 6).tobytes())
                for part in sorted(_parts(obj.instance_collection), key=lambda o: o.name):
                    digest.update(f"{part.name}|".encode())
                    digest.update(np.round(np.array(part.matrix_world), 6).tobytes())
                    digest.update(_keys(part))
                    digest.update(_drivers_and_constraints(part))
                    if part.type == "MESH":
                        # Shown only through the instance, so never visible
                        # itself: read as the instance shows it.
                        if _deformed(part):
                            digest.update(_evaluated_digest(part))
                        else:
                            digest.update(_mesh_digest(part.data))
    return digest.hexdigest()


def _parts(collection, seen=None) -> list:
    """Every object ``collection`` instances, through collections it instances in turn."""
    seen = set() if seen is None else seen
    if collection.name in seen:
        return []
    seen.add(collection.name)
    found = list(collection.all_objects)
    for obj in list(found):
        if obj.instance_type == "COLLECTION" and obj.instance_collection is not None:
            found += _parts(obj.instance_collection, seen)
    return found


def moving_on_bone(rig) -> list[tuple[str, str]]:
    """The robot's link objects that move on their bones: (name, what moves it).

    Capsules ride rigidly on their bone, so they can't hold one of these: a
    constraint, a driver or keys on its transform puts it somewhere else from
    one pose or frame to the next, and animated shape keys or modifiers
    reshape it. An attached collection moves if anything in it does.
    """
    found = []
    for objects in link_meshes(rig).values():
        for obj in objects:
            why = _what_moves(obj)
            if why:
                found.append((obj.name, why))
    return sorted(found)


def _what_moves(obj) -> str:
    why = _moved_by(obj) or _reshaped_by(obj)
    if why or obj.instance_type != "COLLECTION" or obj.instance_collection is None:
        return why
    for part in sorted(_parts(obj.instance_collection), key=lambda o: o.name):
        why = _moved_by(part) or _reshaped_by(part)
        if why:
            return f"{why}, on {part.name}"
    return ""


def _constrained(obj) -> bool:
    return any(not constraint.mute for constraint in obj.constraints)


def _moved_by(obj) -> str:
    """What moves ``obj`` from one pose or frame to the next, or ""."""
    if _constrained(obj):
        return "a constraint"
    data = obj.animation_data
    if data is not None and any(
        not driver.mute and driver.data_path in _TRANSFORM_PATHS for driver in data.drivers
    ):
        return "a driver"
    if any(not curve.mute for curve in _transform_curves(obj)) or any(
        curves for _track, _strip, curves in _strip_curves(obj)
    ):
        return "keys"
    return ""


def _reshaped_by(obj) -> str:
    """What reshapes ``obj`` from one frame to the next, or "": the fit reads one frame."""
    keys = getattr(getattr(obj, "data", None), "shape_keys", None)
    if keys is not None and _animated(keys):
        return "animated shape keys"
    if _animated(obj, "modifiers["):
        return "animated modifiers"
    if obj.type == "MESH" and _animated(obj.data):
        return "an animated mesh"
    return ""


def _animated(block, prefix: str = "") -> bool:
    """Whether keys, a strip or a driver play on ``block``'s paths starting ``prefix``."""
    from .ik import _iter_fcurve_containers

    data = getattr(block, "animation_data", None)
    if data is None:
        return False
    if any(not d.mute and d.data_path.startswith(prefix) for d in data.drivers):
        return True
    actions = [data.action] if data.action is not None else []
    if data.use_nla:
        actions += [
            strip.action
            for track in data.nla_tracks if not track.mute
            for strip in track.strips if not strip.mute and strip.action is not None
        ]
    return any(
        not curve.mute and curve.data_path.startswith(prefix)
        for action in actions
        for container in _iter_fcurve_containers(action)
        for curve in container
    )


def _hash_mesh(digest, obj) -> None:
    """A mesh object's shape as the fit reads it: with its modifiers and shape keys.

    The fit reads what Blender evaluates for an object it evaluates; a hidden
    one, it reads unmodified. So does this.
    """
    if obj.visible_get() and _deformed(obj):
        digest.update(_evaluated_digest(obj))
    else:
        digest.update(_mesh_digest(obj.data))


def _deformed(obj) -> bool:
    """Whether what Blender evaluates ``obj`` to can differ from its mesh."""
    keys = getattr(obj.data, "shape_keys", None)
    return any(modifier.show_viewport for modifier in obj.modifiers) or (
        keys is not None and len(keys.key_blocks) > 0
    )


#: Bumped whenever Blender reports a mesh's geometry has changed, by
#: :func:`note_updates`. Hashing every vertex of a robot takes about 10 ms, and
#: the panel asks on every redraw, through playback too; a mesh's hash is only
#: worked out again once some mesh has changed.
_generation = 0
#: Mesh name -> (the generation it was hashed in, its hash).
_mesh_hashes: dict[str, tuple[int, bytes]] = {}
#: Bumped on every change Blender reports to the scene, and on frame changes,
#: undo and redo: what's worked out for a redraw holds until it moves on.
_epoch = 0
#: Rig name -> ((epoch, generation, the capsules' source), whether stale).
_stale: dict[str, tuple[tuple, bool]] = {}


def _mesh_digest(mesh) -> bytes:
    cached = _mesh_hashes.get(mesh.name)
    if cached is not None and cached[0] == _generation:
        return cached[1]
    found = _shape_digest(mesh)
    _mesh_hashes[mesh.name] = (_generation, found)
    return found


def _evaluated_digest(obj) -> bytes:
    key = f"evaluated|{obj.name}"
    cached = _mesh_hashes.get(key)
    if cached is not None and cached[0] == _generation:
        return cached[1]
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        found = _shape_digest(mesh)
    finally:
        evaluated.to_mesh_clear()
    _mesh_hashes[key] = (_generation, found)
    return found


def _shape_digest(mesh) -> bytes:
    """A mesh's shape as the fit reads it: its vertices, and the faces it spreads points over.

    The faces too: one added or removed between the same vertices changes the
    surface the fit samples, and not a vertex.
    """
    flat = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", flat)
    corners = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", corners)
    sides = np.empty(len(mesh.polygons), dtype=np.int32)
    mesh.polygons.foreach_get("loop_total", sides)
    digest = hashlib.sha1()
    # To a tenth of a millimetre: float noise in a mesh untouched isn't a change.
    digest.update(np.round(flat, 4).tobytes())
    digest.update(corners.tobytes())
    digest.update(sides.tobytes())
    return digest.digest()


def note_updates(depsgraph) -> None:
    """A depsgraph update: whether any mesh's geometry changed in it.

    A mesh reshaped, or what its object's modifiers make of it: Blender reports
    a modifier's own change as the mesh updated, and one following another
    object as the object updated. Moving, posing and playing don't change a
    mesh's geometry, only where it is.
    """
    global _generation, _epoch
    _epoch += 1
    for update in depsgraph.updates:
        if not update.is_updated_geometry:
            continue
        # The object too: a modifier that follows another object changes what
        # the object evaluates to when that object moves, and the mesh itself
        # isn't touched.
        # A shape key's value changes what the object evaluates to, not its
        # mesh: Blender reports the object.
        changed = update.id
        if isinstance(changed, bpy.types.Mesh) or (
            isinstance(changed, bpy.types.Object)
            and changed.type == "MESH"
            and _deformed(changed)
        ):
            _generation += 1
            return


def note_change() -> None:
    """The scene may have changed without a depsgraph update: a frame set."""
    global _epoch
    _epoch += 1


def note_undo() -> None:
    """An undo or a redo: any mesh may be back to a shape it had, and nothing says which."""
    global _epoch, _generation
    _epoch += 1
    _generation += 1


def forget() -> None:
    """A new file: its meshes share names, not shapes, with the last one's."""
    global _generation, _epoch
    _generation += 1
    _epoch += 1
    _mesh_hashes.clear()
    _stale.clear()


def capsules_of(rig) -> list[geometry.Capsule]:
    """The capsules last fitted for ``rig``, on bones it still has."""
    bones = rig.pose.bones
    return [
        geometry.Capsule(item.bone, np.array(item.a), np.array(item.b), float(item.radius))
        for item in getattr(rig, "kinema_capsules", ())
        if item.bone in bones
    ]


def capsule_ends(rig, capsules) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each capsule in world space, as the rig stands now: its ends, two (C, 3)
    arrays, and its radius, (C,).

    The radius scaled as the ends are: by the rig's scale, and its parents'. A
    scale that stretches one way more than another makes an ellipse of the
    capsule's section, and the radius is its widest, so the capsule holds it.
    """
    if not capsules:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    placed, stretch = {}, {}
    for capsule in capsules:
        if capsule.bone not in placed:
            matrix = np.array(rig.matrix_world @ rig.pose.bones[capsule.bone].matrix)
            placed[capsule.bone] = matrix
            stretch[capsule.bone] = float(np.linalg.norm(matrix[:3, :3], 2))
    a = np.array([placed[c.bone][:3, :3] @ c.a + placed[c.bone][:3, 3] for c in capsules])
    b = np.array([placed[c.bone][:3, :3] @ c.b + placed[c.bone][:3, 3] for c in capsules])
    radii = np.array([c.radius * stretch[c.bone] for c in capsules])
    return a, b, radii


def collision_state(rig, scene) -> str:
    """A fingerprint of what a check measures the robot's clearance by.

    The capsules, each one and the meshes they were fitted to, and every
    obstacle: its shape and where it stands.
    Stored with a check, so a clearance measured against obstacles since moved,
    or capsules since fitted again, isn't shown as current. An object something
    animates is fingerprinted by its keys, not by where the current frame has
    put it: scrubbing changes nothing the check measured.

    Empty with neither capsules nor obstacles: there is nothing to measure,
    and a check stored before this fingerprint existed matches it.
    """
    objects = sorted(obstacle_objects(scene), key=lambda o: o.name)
    if not len(rig.kinema_capsules) and not objects:
        return ""
    digest = hashlib.sha1(rig.kinema_capsules_source.encode())
    # The capsules themselves, not only the meshes they were fitted to: the
    # same meshes fitted another way, or a capsule edited, measure differently.
    for capsule in rig.kinema_capsules:
        digest.update(f"|{capsule.bone}|".encode())
        digest.update(np.round([*capsule.a, *capsule.b, capsule.radius], 6).tobytes())
    # The robot's own placement: moved, every capsule is somewhere else. Its
    # pose is the job's, in the waypoints' fingerprint.
    _placement(digest, rig)
    # Where its bones rest: Set TCP moves the TCP's, and the tool's capsules
    # with it. Not their pose, which is the job's.
    for bone in rig.data.bones:
        parent = bone.parent.name if bone.parent else ""
        digest.update(f"{bone.name}|{parent}|{round(bone.length, 6)}|".encode())
        digest.update(np.round(np.array(bone.matrix_local), 6).tobytes())
    for obj in objects:
        digest.update(f"|{obj.name}|{obj.type}|{obj.kinema_obstacle_shape}|".encode())
        # The bounds it's measured by: an empty's are its display size.
        digest.update(np.round(np.concatenate(_local_bounds(obj)), 6).tobytes())
        if obj.type == "MESH":
            _hash_mesh(digest, obj)
        _placement(digest, obj)
    return digest.hexdigest()


def _placement(digest, obj) -> None:
    """Where ``obj`` stands, and everything that puts it there, up its parents."""
    link = obj
    while link is not None:
        digest.update(f"{link.name}|{link.parent_type}|{link.parent_bone}|".encode())
        digest.update(np.round(np.array(link.matrix_parent_inverse), 6).tobytes())
        digest.update(_keys(link))
        digest.update(_unkeyed_transform(link))
        digest.update(_drivers_and_constraints(link))
        link = link.parent


#: The transform channels an object's matrix_basis is made from, and how many
#: values each has.
_TRANSFORM = (
    ("location", 3), ("rotation_euler", 3), ("rotation_quaternion", 4),
    ("rotation_axis_angle", 4), ("scale", 3), ("delta_location", 3),
    ("delta_rotation_euler", 3), ("delta_rotation_quaternion", 4), ("delta_scale", 3),
)


_TRANSFORM_PATHS = frozenset(path for path, _ in _TRANSFORM)


def _transform_curves(obj) -> list:
    """The curves of ``obj``'s own animation that move it: its transform channels'."""
    from .ik import own_fcurve_containers

    return [
        curve
        for container in own_fcurve_containers(obj)
        for curve in container
        if curve.data_path in _TRANSFORM_PATHS
    ]


def _strip_curves(obj) -> list:
    """The NLA strips that play on ``obj``: (track, strip, its transform curves).

    Only those that play: tracks and strips not muted, the soloed track alone
    where one is, and none with the NLA turned off.
    """
    from .ik import _iter_fcurve_containers

    data = obj.animation_data
    if data is None or not data.use_nla:
        return []
    tracks = [track for track in data.nla_tracks if track.is_solo] or list(data.nla_tracks)
    found = []
    for track in tracks:
        if track.mute:
            continue
        for strip in track.strips:
            if strip.mute or strip.action is None:
                continue
            handle = None
            if getattr(strip.action, "layers", None):
                handle = getattr(getattr(strip, "action_slot", None), "handle", None)
                if handle is None:
                    continue
            curves = [
                curve
                for container in _iter_fcurve_containers(strip.action, handle)
                for curve in container
                if curve.data_path in _TRANSFORM_PATHS
            ]
            found.append((track, strip, curves))
    return found


def _unkeyed_transform(obj) -> bytes:
    """Every transform channel nothing keys or drives, by its value.

    Keys usually cover some channels only, a moving obstacle's location, say;
    its scale and turn are set by hand, and changing them changes what it
    covers. A keyed channel is in its keys instead, and a driven one in its
    driver's settings: where the current frame has put it is no change.
    """
    # A muted curve sets nothing: its channel is wherever it was put.
    keyed = {
        (curve.data_path, curve.array_index)
        for curve in _transform_curves(obj)
        if not curve.mute
    }
    for _track, _strip, curves in _strip_curves(obj):
        keyed |= {(curve.data_path, curve.array_index) for curve in curves if not curve.mute}
    data = obj.animation_data
    for driver in (data.drivers if data is not None else ()):
        if not driver.mute:
            keyed.add((driver.data_path, driver.array_index))
    values = [obj.rotation_mode]
    for path, size in _TRANSFORM:
        for index in range(size):
            if (path, index) not in keyed:
                values.append(f"{path}[{index}]={round(getattr(obj, path)[index], 6)}")
    return "|".join(values).encode()


def _drivers_and_constraints(obj) -> bytes:
    """What drives an object or constrains it, by every setting it has.

    Every setting, not a chosen few: a Copy Location's offset, a driver's
    variable type, any of them moves the object. Not what they read: a
    constraint follows its target and a driver its inputs, and those are
    fingerprinted only where they are obstacles themselves. Moving the target
    of an obstacle's constraint is not seen.
    """
    parts = []
    data = obj.animation_data
    for curve in (data.drivers if data is not None else ()):
        variables = [
            f"{_settings(variable)}<{';'.join(_settings(target) for target in variable.targets)}>"
            for variable in curve.driver.variables
        ]
        modifiers = [f"{modifier.type}:{_settings(modifier)}" for modifier in curve.modifiers]
        keys = [tuple(round(value, 6) for value in point.co) for point in curve.keyframe_points]
        parts.append(
            f"{curve.data_path}[{curve.array_index}]:{curve.mute}:{_settings(curve.driver)}:"
            f"{variables}:{modifiers}:{keys}"
        )
    for constraint in obj.constraints:
        parts.append(f"{constraint.type}:{_settings(constraint)}")
    return "|".join(parts).encode()


#: Settings that change how something is shown in the interface, not what it does.
_INTERFACE_ONLY = frozenset(
    {"rna_type", "show_expanded", "active", "is_override_data_local", "select"}
)


def _settings(struct) -> str:
    """Every writable setting of ``struct`` but its lists, as the same text in every session.

    What it points to by name, not followed. Numbers rounded, so a value read
    back from a file is the value stored. Flag sets sorted: a set's order
    changes from one Python process to the next.
    """
    values = []
    for prop in struct.bl_rna.properties:
        if prop.is_readonly or prop.type == "COLLECTION" or prop.identifier in _INTERFACE_ONLY:
            continue
        value = getattr(struct, prop.identifier)
        if prop.type == "POINTER":
            value = getattr(value, "name", "") if value is not None else ""
        elif prop.type == "FLOAT":
            value = np.round(np.array(value, dtype=float), 6).tolist()
        elif prop.type == "ENUM" and prop.is_enum_flag:
            value = sorted(value)
        elif getattr(prop, "array_length", 0):
            value = tuple(value)
        values.append(f"{prop.identifier}={value}")
    return ",".join(values)


def _keys(obj) -> bytes:
    """How an object's own animation moves it: empty when nothing does.

    Each transform curve's keys, and everything that shapes the curve between
    and past them: each key's interpolation and easing, the curve's
    extrapolation, whether it is muted, its modifiers. Only the transform's:
    a robot's action keys its joints too, and those are the job's.

    The active action's, and each NLA strip's that plays, with the strip's own
    timing and blending and its track's.
    """
    data = obj.animation_data
    if data is None:
        return b""
    strips = _strip_curves(obj)
    if data.action is None and not strips:
        return b""
    parts = [
        f"use_nla={data.use_nla}|{data.action_blend_type}|{data.action_extrapolation}|"
        f"{round(data.action_influence, 6)}".encode(),
        _curves_digest(_transform_curves(obj) if data.action is not None else []),
    ]
    for track, strip, curves in strips:
        parts.append(f"{track.name}:{track.is_solo}:{_settings(strip)}".encode())
        parts.append(_curves_digest(curves))
    return b"|".join(parts)


def _curves_digest(curves) -> bytes:
    """``curves``' keys, and everything that shapes each curve between and past them."""
    parts = []
    for curve in sorted(curves, key=lambda c: (c.data_path, c.array_index)):
        points = curve.keyframe_points
        shape = []
        for field in ("co", "handle_left", "handle_right"):
            flat = np.empty(len(points) * 2)
            points.foreach_get(field, flat)
            shape.append(flat)
        shape = np.concatenate(shape)
        timing = [
            (point.interpolation, point.easing, round(point.back, 6), round(point.amplitude, 6),
             round(point.period, 6))
            for point in points
        ]
        modifiers = [f"{modifier.type}:{_settings(modifier)}" for modifier in curve.modifiers]
        parts.append(
            f"{curve.data_path}[{curve.array_index}]:{curve.mute}:{curve.extrapolation}:"
            f"{timing}:{modifiers}:".encode()
        )
        parts.append(np.round(shape, 6).tobytes())
    return b"|".join(parts)


def check_measured_this(rig, scene) -> bool:
    """Whether the stored check measured the capsules and obstacles there now."""
    return rig.kinema_motion_check_collision == collision_state(rig, scene)


def closest_approach(rig, scene):
    """The stored check's closest approach to an obstacle over the job, if it is current.

    None when no move was measured, or when the waypoints, the obstacles or the
    capsules have changed since the check: then it describes something else.
    None too while the capsules are stale: measured with capsules that no
    longer fit the robot, it isn't the robot's clearance.
    """
    from .waypoints import job_signature

    if rig.kinema_motion_check_job != job_signature(rig):
        return None
    if not check_measured_this(rig, scene) or capsules_stale(rig):
        return None
    return min(
        (check for check in rig.kinema_motion_check if check.clearance_obstacle),
        key=lambda check: check.clearance,
        default=None,
    )


def capsules_stale(rig) -> bool:
    """Whether the meshes, or a link's capsule count, have changed since the capsules were fitted.

    Worked out once per change to the scene: the panels ask on every redraw,
    several times, and fingerprinting where every mesh of a robot sits takes
    about a millisecond.
    """
    if not len(rig.kinema_capsules):
        return False
    key = (_epoch, _generation, rig.kinema_capsules_source)
    cached = _stale.get(rig.name)
    if cached is not None and cached[0] == key:
        return cached[1]
    stale = rig.kinema_capsules_source != source_signature(rig)
    _stale[rig.name] = (key, stale)
    return stale


class KINEMA_OT_clear_capsules(Operator):
    bl_idname = "kinema.clear_capsules"
    bl_label = "Clear Capsules"
    bl_description = (
        "Remove the robot's capsules. The Motion Check stops measuring it against the "
        "obstacles, and nothing is drawn or fingerprinted for it as it moves"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        rig = active_rig(context)
        return rig is not None and bool(len(rig.kinema_capsules))

    def execute(self, context: bpy.types.Context) -> set[str]:
        rig = active_rig(context)
        count = len(rig.kinema_capsules)
        rig.kinema_capsules.clear()
        rig.kinema_capsules_source = ""
        _stale.pop(rig.name, None)
        _redraw(context)
        self.report({"INFO"}, f"{count} capsule{'s' if count != 1 else ''} cleared")
        return {"FINISHED"}


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
        moving = moving_on_bone(rig)
        if moving:
            named = ", ".join(f"{name} ({why})" for name, why in moving[:4])
            more = f" and {len(moving) - 4} more" if len(moving) > 4 else ""
            self.report(
                {"ERROR"},
                f"Capsules ride rigidly on their bones, and these move on theirs: {named}{more}. "
                "Remove what moves them, or apply it",
            )
            return {"CANCELLED"}
        started = time.perf_counter()
        fitted = []
        window = context.window_manager
        window.progress_begin(0, len(meshes))
        try:
            for done, (bone, objects) in enumerate(sorted(meshes.items())):
                points = bone_points(context, rig, bone, objects)
                found = geometry.fit(points, parts=capsule_count(rig, bone))
                fitted += [(bone, a, b, radius) for a, b, radius in found]
                window.progress_update(done + 1)
        except ImportError as exc:
            self.report({"ERROR"}, f"Fit Capsules needs trimesh, which didn't load: {exc}")
            return {"CANCELLED"}
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
    """The scene's obstacles collection, made and linked under the scene if asked to.

    Kept on the scene, not looked up by name: collections belong to the file,
    so a name would find another scene's obstacles, or a collection of the
    user's that happens to be called the same.
    """
    collection = scene.kinema.obstacles
    if collection is None:
        if not create:
            return None
        collection = bpy.data.collections.new(OBSTACLES)
        scene.kinema.obstacles = collection
    if create and collection not in scene.collection.children_recursive:
        scene.collection.children.link(collection)
    return collection


def obstacle_objects(scene) -> list:
    """The objects in the scene's obstacles collection itself, in this scene.

    Not those in collections inside it: Add Selected and Remove Selected put
    objects in and take them out of this one collection, and an obstacle only
    there through another collection could be neither removed nor told apart.
    Never the robot, nor anything riding on it, however it came to.
    """
    collection = obstacle_collection(scene)
    if collection is None:
        return []
    return [
        obj for obj in collection.objects
        if obj.name in scene.objects and not on_a_robot(obj)
    ]


def nested_objects(scene) -> int:
    """How many objects are in collections inside the obstacles collection: not measured."""
    collection = obstacle_collection(scene)
    if collection is None:
        return 0
    return len({obj for child in collection.children for obj in child.all_objects})


def on_a_robot(obj) -> bool:
    """Whether ``obj`` is a robot, or rides on one: a robot is up its parents somewhere."""
    while obj is not None:
        if builder.is_kinema_rig(obj):
            return True
        obj = obj.parent
    return False


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
        # Mirrored in X or Y, the cross product turns round; its Z doesn't.
        if float(normal @ linear[:, 2]) < 0.0:
            normal = -normal
        if np.linalg.norm(normal) > 0.0:
            turned[:3, :3] = _frame_with_z(normal / np.linalg.norm(normal))
        turned[:3, 3] = location
        return geometry.Obstacle(obj.name, kind, turned, np.zeros(3))
    low, high = _local_bounds(obj)
    local_half = (high - low) / 2.0
    centre = np.array(world @ _vector((low + high) / 2.0))
    turned[:3, 3] = centre
    if kind == geometry.KIND_SPHERE:
        radius = _reach(obj, centre, low, high)
        return geometry.Obstacle(obj.name, kind, turned, np.array([radius]))
    # Square to its rotation, wide enough for every corner of what it is: a
    # parent scaled unevenly and turned shears the box, and the shear has to be
    # inside the box measured, not cut off.
    in_frame = np.array(rotation.to_matrix()).T @ linear
    half = np.abs(in_frame) @ local_half
    return geometry.Obstacle(obj.name, geometry.KIND_BOX, turned, half)


def _reach(obj, centre: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    """How far from ``centre`` any of ``obj`` reaches, as it stands: a sphere's radius.

    Its vertices, so a ball, stretched or not, is measured as the ball it is,
    and anything else is held whole. Without vertices, its bounds' corners.
    """
    world = np.array(obj.matrix_world)
    if obj.type == "MESH":
        mesh = obj.to_mesh()
        try:
            flat = np.empty(len(mesh.vertices) * 3)
            mesh.vertices.foreach_get("co", flat)
        finally:
            obj.to_mesh_clear()
        points = flat.reshape(-1, 3)
    else:
        xs, ys, zs = zip(low, high, strict=True)
        points = np.array([(x, y, z) for x in xs for y in ys for z in zs])
    if not len(points):
        return 0.0
    placed = points @ world[:3, :3].T + world[:3, 3]
    return float(np.linalg.norm(placed - centre, axis=1).max())


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
    return [obj for obj in context.selected_objects if not on_a_robot(obj)]


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
        scene = context.scene
        removed = [obj for obj in context.selected_objects if obj.name in collection.objects]
        for obj in removed:
            collection.objects.unlink(obj)
            # Kept in this scene: an object only in its obstacles would leave it.
            # Its other collections may be another scene's.
            if obj.name not in scene.objects:
                scene.collection.objects.link(obj)
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
    # What a stored check measured the robot's clearance by: see collision_state().
    bpy.types.Object.kinema_motion_check_collision = StringProperty(
        name="Checked Collision",
        description="Fingerprint of the capsules and obstacles the last check measured",
        default="",
    )
    bpy.types.Object.kinema_obstacle_shape = EnumProperty(
        name="Obstacle Shape",
        description="What the Motion Check measures this obstacle as",
        items=SHAPES,
        default=geometry.KIND_BOX,
    )
    bpy.types.PoseBone.kinema_capsule_count = EnumProperty(
        name="Capsules",
        description=(
            "How many capsules Fit Capsules puts round this link. Set more where one fat "
            "capsule stands far out past the link, as Show Collision Shapes draws it; Fit "
            "Again after"
        ),
        items=CAPSULE_COUNTS,
        default="AUTO",
        update=_on_capsule_count_changed,
    )


def unregister_props() -> None:
    del bpy.types.PoseBone.kinema_capsule_count
    del bpy.types.Object.kinema_capsules
    del bpy.types.Object.kinema_capsules_source
    del bpy.types.Object.kinema_motion_check_collision
    del bpy.types.Object.kinema_obstacle_shape


classes = (
    KinemaCapsule,
    KINEMA_OT_fit_capsules,
    KINEMA_OT_clear_capsules,
    KINEMA_OT_add_obstacles,
    KINEMA_OT_remove_obstacles,
)
