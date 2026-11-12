"""COLLADA importer tests. Require a real ``bpy``; run via ``dev.py test``."""

from __future__ import annotations

import pytest

from ..conftest import requires_bpy

pytestmark = requires_bpy


@pytest.fixture
def dae(addon):
    import importlib

    return importlib.import_module(f"{addon.__name__}.io.dae")


def test_imports_basic_triangle(dae, fixture_dir, clean_scene):
    result = dae.import_dae(fixture_dir / "triangle_zup_m.dae")

    assert len(result.objects) == 1
    mesh = result.objects[0].data
    assert len(mesh.vertices) == 3
    assert len(mesh.polygons) == 1
    assert result.unit_meter == pytest.approx(1.0)


def test_millimetre_yup_file_lands_in_metres_and_z_up(dae, fixture_dir, clean_scene):
    """The two corrections that cause most 'wrong size / on its side' bugs.

    The fixture declares millimetres and Y-up, with a vertex at
    (2000, 0, 0) and another at (0, 0, 3000) in file space. Correctly handled,
    those become 2 m along X and 3 m along *Y* in Blender's Z-up world.
    """
    from mathutils import Vector

    result = dae.import_dae(fixture_dir / "triangle_yup_mm.dae")
    assert result.unit_meter == pytest.approx(0.001)

    obj = result.objects[0]
    world = [obj.matrix_world @ Vector(v.co) for v in obj.data.vertices]
    extent = [max(p[i] for p in world) - min(p[i] for p in world) for i in range(3)]

    assert extent[0] == pytest.approx(2.0, abs=1e-6), "millimetres were not scaled"
    # Y_UP -> Z_UP sends file +Z to Blender -Y, so the 3 m edge lands on Y.
    assert extent[1] == pytest.approx(3.0, abs=1e-6), "up-axis was not corrected"
    assert extent[2] == pytest.approx(0.0, abs=1e-6)


def test_unit_scale_can_be_disabled(dae, fixture_dir, clean_scene):
    from mathutils import Vector

    result = dae.import_dae(fixture_dir / "triangle_yup_mm.dae", apply_unit_scale=False)
    obj = result.objects[0]
    world = [obj.matrix_world @ Vector(v.co) for v in obj.data.vertices]
    assert max(p[0] for p in world) == pytest.approx(2000.0, abs=1e-3)


def test_up_axis_can_be_disabled(dae, fixture_dir, clean_scene):
    result = dae.import_dae(fixture_dir / "triangle_yup_mm.dae", apply_up_axis=False)
    rotation = result.objects[0].matrix_world.to_euler()
    assert abs(rotation.x) < 1e-9


def test_broken_texture_still_yields_geometry(dae, fixture_dir, clean_scene):
    """Regression: a missing texture must not cost the whole mesh.

    pycollada drops an effect whose <surface> names an image the file never
    declares. That cascades -- the material fails to bind, and the scene
    traversal then yields *no geometry at all*, so a perfectly good robot link
    imports as nothing. Found on anymal_d's tilt.dae.
    """
    result = dae.import_dae(fixture_dir / "broken_texture.dae")

    assert result.objects, "broken texture reference swallowed the geometry"
    assert len(result.objects[0].data.polygons) == 1
    assert any("material binding failed" in w for w in result.warnings)


def test_missing_file_raises(dae, fixture_dir):
    with pytest.raises(dae.DaeImportError):
        dae.import_dae(fixture_dir / "does_not_exist.dae")


def test_objects_land_in_requested_collection(dae, fixture_dir, clean_scene):
    import bpy

    collection = bpy.data.collections.new("robot_meshes")
    clean_scene.collection.children.link(collection)

    result = dae.import_dae(fixture_dir / "triangle_zup_m.dae", collection=collection)
    assert result.objects[0].name in collection.objects


def test_name_prefix_applied(dae, fixture_dir, clean_scene):
    result = dae.import_dae(fixture_dir / "triangle_zup_m.dae", name_prefix="link_0")
    assert result.objects[0].name.startswith("link_0")


# -- the box fixtures ---------------------------------------------------------
#
# A 10 x 10 x 20 cm box, written in centimetres with one flat normal per face,
# placed 0.192 m up by its node: the shape of a CAD-exported robot link. The
# variants add what CAD exports carry. Each test below failed before its fix.


def _world_triangles(objects):
    """(n, 3, 3) world-space triangles and (n, 3, 3) world-space corner normals."""
    import numpy as np

    triangles, normals = [], []
    for obj in objects:
        mesh = obj.data
        mesh.calc_loop_triangles()
        matrix = np.array(obj.matrix_world)
        linear = matrix[:3, :3]
        co = np.array([v.co for v in mesh.vertices]) @ linear.T + matrix[:3, 3]
        corner = np.array([c.vector for c in mesh.corner_normals]) @ np.linalg.inv(linear)
        for tri in mesh.loop_triangles:
            triangles.append(co[list(tri.vertices)])
            normals.append(corner[list(tri.loops)])
    return np.array(triangles), np.array(normals)


def _worst_normal_angle(objects):
    """Degrees between each corner normal and its own triangle: 0 when flat-shaded."""
    import numpy as np

    triangles, normals = _world_triangles(objects)
    face = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    face /= np.linalg.norm(face, axis=1, keepdims=True)
    normals = normals / np.linalg.norm(normals, axis=2, keepdims=True)
    cosine = np.clip((normals * face[:, None, :]).sum(axis=2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)).max())


def _centroid(objects):
    triangles, _ = _world_triangles(objects)
    return [float(v) for v in triangles.reshape(-1, 3).mean(axis=0)]


def test_authored_normals_are_kept(dae, fixture_dir, clean_scene):
    """Regression: the file's flat normals came out smoothed across every edge.

    Custom normals were set before the faces were marked smooth, and marking
    them smooth merged the fans the normals were stored against, so Blender
    averaged them: a box's corners sat 54.7 degrees off every face. Every
    .dae with normals imported this way.
    """
    result = dae.import_dae(fixture_dir / "box_flat_cm.dae")

    assert sum(len(o.data.polygons) for o in result.objects) == 12
    assert _worst_normal_angle(result.objects) < 0.5


def test_broken_texture_keeps_every_instance_where_its_node_puts_it(dae, fixture_dir,
                                                                    clean_scene):
    """Regression: with every binding broken, the fallback read the library alone.

    The box is instanced twice, by nodes 1 m apart, and its material's texture
    names an image the file never declares. The geometry came in once, without
    either node's placement.
    """
    result = dae.import_dae(fixture_dir / "box_two_instances_broken_texture.dae")

    assert sum(len(o.data.polygons) for o in result.objects) == 24
    assert _centroid(result.objects) == pytest.approx([0.5, 0.0, 0.292], abs=1e-6)
    assert any("material binding failed" in w for w in result.warnings)


def test_one_broken_material_loses_no_instance(dae, fixture_dir, clean_scene):
    """Regression: one instance whose material failed vanished without a warning.

    Only the second node's material is broken. pycollada drops that instance,
    the scene still yields the first, so the old fallback never ran.
    """
    result = dae.import_dae(fixture_dir / "box_one_broken_material.dae")

    assert sum(len(o.data.polygons) for o in result.objects) == 24
    assert _centroid(result.objects) == pytest.approx([0.5, 0.0, 0.292], abs=1e-6)
    assert any("1 instance(s)" in w for w in result.warnings)
    assert any(s.material and s.material.name == "mat"
               for o in result.objects for s in o.material_slots)


def test_zipped_file_with_a_broken_material_loses_no_instance(dae, fixture_dir, tmp_path,
                                                              clean_scene):
    """pycollada reads a zipped COLLADA file; the binding-free re-read must too."""
    import zipfile

    zipped = tmp_path / "box_one_broken_material.zae"
    with zipfile.ZipFile(zipped, "w") as archive:
        archive.write(fixture_dir / "box_one_broken_material.dae", "box.dae")

    result = dae.import_dae(zipped)

    assert sum(len(o.data.polygons) for o in result.objects) == 24
    assert any("1 instance(s)" in w for w in result.warnings)


def test_failed_reread_keeps_what_the_first_read_gave(dae, fixture_dir, monkeypatch,
                                                      clean_scene):
    """The binding-free re-read only adds: if it fails, the import still succeeds."""

    def fail(document):
        raise ValueError("stand-in parse failure")

    monkeypatch.setattr(dae, "_without_bindings", fail)
    result = dae.import_dae(fixture_dir / "box_one_broken_material.dae")

    assert sum(len(o.data.polygons) for o in result.objects) == 12
    assert any("could not re-read" in w and "stand-in parse failure" in w
               for w in result.warnings)


def test_lines_do_not_become_triangles(dae, fixture_dir, clean_scene):
    """Regression: a <lines> primitive was read three indices at a time.

    Three of the box's edges made two triangles the file never had. Franka's
    link7 carries such a <lines>.
    """
    result = dae.import_dae(fixture_dir / "box_with_lines.dae")

    assert sum(len(o.data.polygons) for o in result.objects) == 12


def test_faces_validate_drops_do_not_shift_normals(dae, fixture_dir, clean_scene):
    """Regression: after mesh.validate() dropped a face, the rest took the next one's normals.

    The fixture adds a face that repeats a vertex, before the box's own, and a
    reversed copy of the first face after them. Blender keeps 12 faces, and
    normals and UVs were then read by position from the unfiltered list.

    Each face's three corners share one UV, u = face index / 16, so the faces
    Blender keeps must carry u = 1/16 .. 12/16: not 0 (the degenerate face) or
    13/16 (the duplicate).
    """
    result = dae.import_dae(fixture_dir / "box_degenerate_faces.dae")

    assert sum(len(o.data.polygons) for o in result.objects) == 12
    assert _worst_normal_angle(result.objects) < 0.5

    face_u = []
    for obj in result.objects:
        uv = obj.data.uv_layers["UVMap"].data
        for polygon in obj.data.polygons:
            corner_u = {round(uv[i].uv[0] * 16, 3) for i in polygon.loop_indices}
            assert len(corner_u) == 1, f"one face, several UVs: {corner_u}"
            face_u.append(corner_u.pop())
    assert sorted(face_u) == list(range(1, 13))
