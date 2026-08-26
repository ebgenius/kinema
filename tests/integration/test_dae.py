"""COLLADA importer tests. Require a real ``bpy``; run via ``dev.py test``."""

from __future__ import annotations

import math

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
