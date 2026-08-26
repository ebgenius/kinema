"""Load a link's visual geometry, whatever format the URDF points at.

Robot descriptions reference a small set of mesh formats. Blender 5.2 still
ships importers for STL, OBJ and PLY; COLLADA is handled by Kinema's own
importer because Blender 5.0 removed the native one.

Every importer here is asked to leave the mesh in *file* coordinates. The URDF
already says where the geometry belongs via the link frame, the ``<visual>``
origin and the ``<mesh scale>`` attribute, so any extra "helpful" axis
conversion by an importer has to be switched off or it fights the URDF.
"""

from __future__ import annotations

from pathlib import Path

import bpy

from . import dae

#: Extensions we can load, lowercase, including the dot.
SUPPORTED_SUFFIXES = frozenset({".dae", ".stl", ".obj", ".ply"})


class MeshLoadError(RuntimeError):
    """A visual mesh could not be loaded."""


def _new_objects_from(before: set, collection) -> list[bpy.types.Object]:
    """Objects that appeared since ``before``, moved into ``collection``."""
    created = [obj for obj in bpy.data.objects if obj not in before]
    for obj in created:
        for existing in list(obj.users_collection):
            existing.objects.unlink(obj)
        collection.objects.link(obj)
    return created


def _import_via_operator(operator, filepath: Path, collection, **kwargs):
    before = set(bpy.data.objects)
    try:
        result = operator(filepath=str(filepath), **kwargs)
    except (RuntimeError, TypeError) as exc:
        raise MeshLoadError(f"{filepath.name}: {exc}") from exc
    if "CANCELLED" in result:
        raise MeshLoadError(f"{filepath.name}: importer cancelled")
    return _new_objects_from(before, collection)


def load_mesh(
    filepath: str | Path,
    *,
    collection: bpy.types.Collection,
    name: str = "",
) -> list[bpy.types.Object]:
    """Import one mesh file and return the objects created.

    Objects are left at the origin in file coordinates; the caller positions
    them from the URDF.
    """
    path = Path(filepath)
    if not path.is_file():
        raise MeshLoadError(f"mesh not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise MeshLoadError(f"unsupported mesh format '{suffix}': {path.name}")

    if suffix == ".dae":
        # A URDF's <mesh scale> and the link frame fully determine placement,
        # so the file's own unit/up-axis metadata must still be honoured (it
        # describes the file's own coordinates) but nothing else applied.
        result = dae.import_dae(path, collection=collection, name_prefix=name or path.stem)
        if not result.objects:
            raise MeshLoadError(f"{path.name}: no geometry ({'; '.join(result.warnings)})")
        return result.objects

    if suffix == ".stl":
        # global_scale=1 and forward/up set to the identity mapping: the STL
        # importer otherwise rotates Y-up files, which URDF never wants.
        objects = _import_via_operator(
            bpy.ops.wm.stl_import, path, collection,
            global_scale=1.0, forward_axis="Y", up_axis="Z",
        )
    elif suffix == ".obj":
        objects = _import_via_operator(
            bpy.ops.wm.obj_import, path, collection,
            global_scale=1.0, forward_axis="Y", up_axis="Z",
        )
    else:  # .ply
        objects = _import_via_operator(
            bpy.ops.wm.ply_import, path, collection, global_scale=1.0,
        )

    if not objects:
        raise MeshLoadError(f"{path.name}: importer produced no objects")

    if name:
        for index, obj in enumerate(objects):
            obj.name = name if index == 0 else f"{name}.{index:03d}"
            if obj.data is not None:
                obj.data.name = obj.name
    return objects


def make_primitive(
    kind: str,
    params: dict,
    *,
    collection: bpy.types.Collection,
    name: str = "",
) -> list[bpy.types.Object]:
    """Build a URDF geometric primitive (box / cylinder / sphere).

    Created through ``bpy.data`` and mesh ops rather than ``bpy.ops.mesh.*``,
    which depend on an active object and a usable context -- neither of which
    exists when importing from a background process or a file browser.
    """
    import bmesh

    mesh = bpy.data.meshes.new(name or kind)
    bm = bmesh.new()
    try:
        if kind == "box":
            size = params.get("size", (1.0, 1.0, 1.0))
            bmesh.ops.create_cube(bm, size=1.0)
            bmesh.ops.scale(bm, vec=tuple(float(s) for s in size), verts=bm.verts)
        elif kind == "cylinder":
            radius = float(params.get("radius", 0.5))
            length = float(params.get("length", 1.0))
            bmesh.ops.create_cone(
                bm, cap_ends=True, cap_tris=False, segments=32,
                radius1=radius, radius2=radius, depth=length,
            )
        elif kind == "sphere":
            radius = float(params.get("radius", 0.5))
            bmesh.ops.create_uvsphere(
                bm, u_segments=32, v_segments=16, radius=radius,
            )
        else:
            raise MeshLoadError(f"unknown primitive '{kind}'")
        bm.to_mesh(mesh)
    finally:
        bm.free()

    mesh.update()
    obj = bpy.data.objects.new(name or kind, mesh)
    collection.objects.link(obj)
    return [obj]
