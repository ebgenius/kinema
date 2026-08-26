"""COLLADA (.dae) mesh import, built on pycollada.

Blender 5.0 removed its C++ COLLADA module outright (Blender 4.5 LTS was the
last release to ship it). That is a problem for robotics specifically, because
a large share of ROS and Gazebo robot descriptions still reference ``.dae``
meshes -- so on Blender 5 an otherwise-valid URDF imports as a robot with no
visible geometry.

This module reads the subset of COLLADA that robot meshes actually use:
triangles and polylists, per-vertex normals, UVs, and material diffuse colour.
It deliberately does not attempt animation, skinning, or controllers.

Two details cause most of the "imported robot is the wrong size / lying on its
side" bugs seen in other importers, and both are handled here:

* ``<unit meter="...">`` -- CAD exporters routinely emit millimetres. A file
  with ``meter="0.001"`` whose scale is ignored produces a robot 1000x too big.
* ``<up_axis>`` -- COLLADA defaults to Y_UP, Blender is Z_UP. Ignoring this
  lays the robot on its side. RViz and Gazebo both honour it, so Kinema does
  too, and URDF authors have already accounted for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import bpy
from mathutils import Matrix

# Rotations that bring each COLLADA up-axis into Blender's Z-up world.
_UP_AXIS_TO_ZUP: dict[str, Matrix] = {
    "Y_UP": Matrix.Rotation(1.5707963267948966, 4, "X"),
    "Z_UP": Matrix.Identity(4),
    "X_UP": Matrix.Rotation(1.5707963267948966, 4, "Z"),
}


class DaeImportError(RuntimeError):
    """A COLLADA file could not be read."""


@dataclass
class DaeImportResult:
    objects: list[bpy.types.Object] = field(default_factory=list)
    unit_meter: float = 1.0
    up_axis: str = "Y_UP"
    warnings: list[str] = field(default_factory=list)

    @property
    def meshes(self) -> list[bpy.types.Mesh]:
        return [obj.data for obj in self.objects]


def _load_collada(filepath: Path):
    try:
        import collada
    except ImportError as exc:  # pragma: no cover - dependency is bundled
        raise DaeImportError(
            "pycollada is not available; Kinema cannot read .dae files"
        ) from exc

    try:
        # ignore=... keeps a single malformed effect or unsupported controller
        # from aborting the whole file. Robot meshes are frequently exported by
        # CAD tools that emit slightly non-conformant COLLADA.
        return collada.Collada(
            str(filepath),
            ignore=[
                collada.common.DaeUnsupportedError,
                collada.common.DaeBrokenRefError,
            ],
        )
    except Exception as exc:  # noqa: BLE001
        raise DaeImportError(f"could not parse {filepath.name}: {exc}") from exc


def _correction_matrix(document, *, apply_unit: bool, apply_up_axis: bool) -> Matrix:
    """Build the file-level transform: unit scale, then up-axis rotation."""
    matrix = Matrix.Identity(4)

    if apply_up_axis:
        up_axis = str(getattr(document.assetInfo, "upaxis", "Y_UP") or "Y_UP")
        # pycollada may report either "Y_UP" or the enum's repr.
        up_axis = up_axis.rsplit(".", 1)[-1].upper()
        matrix = _UP_AXIS_TO_ZUP.get(up_axis, Matrix.Identity(4)) @ matrix

    if apply_unit:
        unit = getattr(document.assetInfo, "unitmeter", None)
        if unit and unit > 0 and abs(unit - 1.0) > 1e-9:
            matrix = matrix @ Matrix.Scale(unit, 4)

    return matrix


def _material_for(primitive, cache: dict[str, bpy.types.Material]):
    """Create (or reuse) a Blender material from a COLLADA effect."""
    material = getattr(primitive, "material", None)
    if isinstance(material, str):
        # Unbound fallback path: `material` is the raw symbol from
        # <triangles material="..."> rather than a resolved Material object.
        effect, name = None, material
    else:
        effect = getattr(material, "effect", None)
        name = (
            getattr(material, "id", None)
            or getattr(material, "name", None)
            or "dae_material"
        )

    if name in cache:
        return cache[name]

    blender_material = bpy.data.materials.new(name=name)
    # Blender 5.x creates a node tree for new materials automatically, and
    # `use_nodes` is deprecated for removal in 6.0 -- only touch it on older
    # builds that still need the opt-in.
    if blender_material.node_tree is None and hasattr(blender_material, "use_nodes"):
        blender_material.use_nodes = True
    node_tree = blender_material.node_tree
    bsdf = node_tree.nodes.get("Principled BSDF") if node_tree else None

    if bsdf is not None and effect is not None:
        diffuse = getattr(effect, "diffuse", None)
        # diffuse is either an RGBA tuple or a Map (texture); only the flat
        # colour is meaningful without also wiring up image nodes.
        if isinstance(diffuse, (tuple, list)) and len(diffuse) >= 3:
            rgba = tuple(diffuse[:3]) + (diffuse[3] if len(diffuse) > 3 else 1.0,)
            bsdf.inputs["Base Color"].default_value = rgba
            if len(diffuse) > 3 and diffuse[3] < 1.0:
                bsdf.inputs["Alpha"].default_value = diffuse[3]
        shininess = getattr(effect, "shininess", None)
        if isinstance(shininess, (int, float)) and shininess > 0:
            # COLLADA shininess is a Blinn exponent; map it into 0..1 roughness.
            bsdf.inputs["Roughness"].default_value = max(
                0.0, min(1.0, 1.0 - min(shininess, 128.0) / 128.0)
            )

    cache[name] = blender_material
    return blender_material


def _iter_triangle_sets(bound_geometry):
    """Yield bound triangle sets, converting polylists as needed."""
    for primitive in bound_geometry.primitives():
        # BoundPolylist and BoundPolygons both expose triangleset().
        converter = getattr(primitive, "triangleset", None)
        if callable(converter):
            try:
                yield converter()
                continue
            except Exception:  # noqa: BLE001 - fall through to raw primitive
                pass
        yield primitive


def _node_transforms(document) -> dict[int, Matrix]:
    """Map geometry ``id()`` to the world matrix of the node instancing it.

    Only needed by the unbound fallback below; the normal path gets baked
    transforms from ``scene.objects()``.
    """
    transforms: dict[int, Matrix] = {}

    def walk(nodes, parent: Matrix) -> None:
        for node in nodes:
            matrix = parent
            raw = getattr(node, "matrix", None)
            if raw is not None:
                try:
                    matrix = parent @ Matrix([list(row) for row in raw])
                except Exception:  # noqa: BLE001
                    matrix = parent
            geometry = getattr(node, "geometry", None)
            if geometry is not None:
                transforms.setdefault(id(geometry), matrix)
            walk(getattr(node, "children", ()) or (), matrix)

    try:
        walk(getattr(document.scene, "nodes", ()) or (), Matrix.Identity(4))
    except Exception:  # noqa: BLE001
        pass
    return transforms


def _iter_unbound_geometries(document):
    """Fallback: read geometry straight from the library, ignoring materials.

    ``scene.objects('geometry')`` resolves each ``<instance_geometry>`` through
    its ``<bind_material>``. If an effect fails to load -- most often a
    ``<surface>`` pointing at a texture image the file never declares, which is
    common in exported robot meshes -- pycollada drops the material, the bind
    cannot resolve, and the traversal yields *nothing at all* even though the
    triangles are perfectly readable.

    Reading ``document.geometries`` directly bypasses material resolution
    entirely, so a missing texture costs the material rather than the mesh.
    """
    transforms = _node_transforms(document)
    for geometry in getattr(document, "geometries", ()) or ():
        matrix = transforms.get(id(geometry), Matrix.Identity(4))
        for primitive in getattr(geometry, "primitives", ()) or ():
            converter = getattr(primitive, "triangleset", None)
            if callable(converter):
                try:
                    yield converter(), matrix
                    continue
                except Exception:  # noqa: BLE001
                    pass
            yield primitive, matrix


def _build_mesh(name: str, triangle_set, material_cache) -> bpy.types.Mesh | None:
    """Convert one bound triangle set into a Blender mesh."""
    vertices = getattr(triangle_set, "vertex", None)
    indices = getattr(triangle_set, "vertex_index", None)
    if vertices is None or indices is None or len(indices) == 0:
        return None

    mesh = bpy.data.meshes.new(name)
    faces = indices.reshape(-1, 3)

    mesh.from_pydata(
        [tuple(v) for v in vertices],
        [],
        [tuple(int(i) for i in face) for face in faces],
    )
    mesh.update()
    if mesh.validate(verbose=False):
        # validate() returns True when it had to remove invalid geometry --
        # common with CAD exports containing degenerate triangles.
        mesh.update()

    # UVs, if the file carries any.
    texcoords = getattr(triangle_set, "texcoordset", None)
    texcoord_indices = getattr(triangle_set, "texcoord_indexset", None)
    if texcoords is not None and len(texcoords) and texcoord_indices is not None:
        try:
            uv_layer = mesh.uv_layers.new(name="UVMap")
            uvs = texcoords[0]
            uv_index = texcoord_indices[0].reshape(-1)
            flat = [0.0] * (len(mesh.loops) * 2)
            for loop_index in range(min(len(mesh.loops), len(uv_index))):
                u, v = uvs[uv_index[loop_index]][:2]
                flat[loop_index * 2] = float(u)
                flat[loop_index * 2 + 1] = float(v)
            uv_layer.data.foreach_set("uv", flat)
        except Exception:  # noqa: BLE001 - UVs are cosmetic; never fail import
            pass

    # Normals: prefer the file's, otherwise smooth-shade nothing and let
    # Blender compute face normals.
    normals = getattr(triangle_set, "normal", None)
    normal_indices = getattr(triangle_set, "normal_index", None)
    if normals is not None and normal_indices is not None and len(normals):
        try:
            flat_indices = normal_indices.reshape(-1)
            loop_normals = [
                tuple(normals[flat_indices[i]]) for i in range(len(mesh.loops))
            ]
            mesh.normals_split_custom_set(loop_normals)
            for polygon in mesh.polygons:
                polygon.use_smooth = True
        except Exception:  # noqa: BLE001 - bad normals should not fail import
            pass

    material = _material_for(triangle_set, material_cache)
    if material is not None:
        mesh.materials.append(material)

    return mesh


def import_dae(
    filepath: str | Path,
    *,
    collection: bpy.types.Collection | None = None,
    apply_unit_scale: bool = True,
    apply_up_axis: bool = True,
    name_prefix: str = "",
) -> DaeImportResult:
    """Import a COLLADA file and return the objects created.

    Args:
        filepath: the .dae file to read.
        collection: collection to link objects into; defaults to the scene's.
        apply_unit_scale: honour ``<unit meter="...">``. Disable only if the
            caller has already accounted for the file's units.
        apply_up_axis: honour ``<up_axis>``. Disable for files known to be
            authored Z-up despite declaring otherwise.
        name_prefix: prepended to created object names, to keep link meshes
            from different robots distinguishable in the outliner.
    """
    path = Path(filepath)
    if not path.is_file():
        raise DaeImportError(f"no such file: {path}")

    document = _load_collada(path)
    result = DaeImportResult(
        unit_meter=float(getattr(document.assetInfo, "unitmeter", 1.0) or 1.0),
        up_axis=str(getattr(document.assetInfo, "upaxis", "Y_UP") or "Y_UP"),
    )

    if document.scene is None:
        raise DaeImportError(f"{path.name} contains no visual scene")

    correction = _correction_matrix(
        document, apply_unit=apply_unit_scale, apply_up_axis=apply_up_axis
    )
    target = collection or bpy.context.scene.collection
    material_cache: dict[str, bpy.types.Material] = {}
    stem = name_prefix or path.stem

    def emit(triangle_set, extra: Matrix) -> None:
        name = f"{stem}" if not result.objects else f"{stem}.{len(result.objects):03d}"
        try:
            mesh = _build_mesh(name, triangle_set, material_cache)
        except Exception as exc:  # noqa: BLE001
            result.warnings.append(f"skipped a primitive in {path.name}: {exc}")
            return
        if mesh is None:
            return
        obj = bpy.data.objects.new(name, mesh)
        obj.matrix_world = correction @ extra
        target.objects.link(obj)
        result.objects.append(obj)

    # Preferred path: scene traversal bakes each node's transform into the
    # vertices and resolves materials, so only the file-level correction remains.
    for bound_geometry in document.scene.objects("geometry"):
        for triangle_set in _iter_triangle_sets(bound_geometry):
            emit(triangle_set, Matrix.Identity(4))

    if not result.objects and getattr(document, "geometries", None):
        result.warnings.append(
            f"{path.name}: material binding failed (often a missing texture); "
            f"imported geometry without materials"
        )
        for triangle_set, matrix in _iter_unbound_geometries(document):
            emit(triangle_set, matrix)

    if not result.objects:
        result.warnings.append(f"{path.name} produced no geometry")

    return result
