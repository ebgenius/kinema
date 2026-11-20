"""A description's path kept relative to the .blend (#92): rig/description_path.py.

Needs a real ``bpy``: the path is written as the file is saved, and read as it
is opened somewhere else.

The claims:

* a description in the .blend's folder, or under it, is kept relative to it as
  the file is saved, and as it is saved somewhere else;
* a project folder moved or copied as a whole finds its own copy of the robot,
  and a .blend moved on its own still finds the description where it was;
* a description anywhere else, or the preference for relative paths off, keeps
  the absolute path only;
* a Save Copy leaves the open file's path as it was;
* a rig linked from a library finds the description beside the library;
* Locate Description keeps the file it found relative, where it can be.
"""

from __future__ import annotations

import importlib
import os
import shutil

import pytest

from ..conftest import requires_bpy

pytestmark = requires_bpy


@pytest.fixture
def modules(addon):
    return lambda name: importlib.import_module(f"{addon.__name__}.{name}")


@pytest.fixture
def unsaved(clean_scene):
    """Leave Blender on an unsaved, empty file, as the tests after these expect."""
    import bpy

    yield
    bpy.ops.wm.read_homefile(use_empty=True)
    assert bpy.data.filepath == ""


@pytest.fixture
def project(tmp_path, fixture_dir, unsaved):
    """A project folder with the robot's description under it."""
    folder = tmp_path / "project"
    (folder / "robots").mkdir(parents=True)
    shutil.copy(fixture_dir / "arm6.urdf", folder / "robots" / "arm6.urdf")
    return folder


def _build(modules, source):
    import bpy

    builder = modules("rig.builder")
    assert "FINISHED" in bpy.ops.kinema.build_robot(filepath=str(source))
    rig = next(o for o in bpy.data.objects if builder.is_kinema_rig(o))
    bpy.context.view_layer.objects.active = rig
    return rig


def _save(path):
    import bpy

    path.parent.mkdir(parents=True, exist_ok=True)
    assert "FINISHED" in bpy.ops.wm.save_as_mainfile(filepath=str(path))


def _open(path, modules):
    """The rig in the .blend at ``path``, opened."""
    import bpy

    assert "FINISHED" in bpy.ops.wm.open_mainfile(filepath=str(path), load_ui=False)
    builder = modules("rig.builder")
    return next(o for o in bpy.data.objects if builder.is_kinema_rig(o))


def _relative(modules, rig):
    return rig.get(modules("rig.description_path").PROP_SOURCE_RELATIVE)


class TestSaving:
    def test_a_description_under_the_blends_folder_is_kept_relative(self, modules, project):
        rig = _build(modules, project / "robots" / "arm6.urdf")
        assert _relative(modules, rig) is None, "precondition: nothing to be relative to yet"
        _save(project / "scene.blend")
        kept = "//" + os.path.join("robots", "arm6.urdf")
        assert _relative(modules, rig) == kept
        assert _relative(modules, _open(project / "scene.blend", modules)) == kept, "in the file"

    def test_save_as_keeps_it_pointing_at_the_same_file(self, modules, project):
        rig = _build(modules, project / "robots" / "arm6.urdf")
        _save(project / "scene.blend")
        _save(project / "robots" / "beside.blend")
        assert _relative(modules, rig) == "//arm6.urdf"
        _save(project / "robots" / "deeper" / "below.blend")
        assert _relative(modules, rig) is None, "the description isn't under it"

    def test_a_description_elsewhere_stays_absolute(self, modules, project, tmp_path):
        source = tmp_path / "robots" / "arm6.urdf"
        source.parent.mkdir()
        shutil.copy(project / "robots" / "arm6.urdf", source)
        rig = _build(modules, source)
        _save(project / "scene.blend")
        assert _relative(modules, rig) is None
        assert rig[modules("rig.builder").PROP_SOURCE] == str(source)

    def test_the_preference_off_keeps_it_absolute(self, modules, project):
        import bpy

        filepaths = bpy.context.preferences.filepaths
        rig = _build(modules, project / "robots" / "arm6.urdf")
        _save(project / "scene.blend")
        assert _relative(modules, rig), "precondition: kept relative while it is on"
        filepaths.use_relative_paths = False
        try:
            _save(project / "scene.blend")
        finally:
            filepaths.use_relative_paths = True
        assert _relative(modules, rig) is None

    def test_a_save_copy_leaves_the_open_files_own(self, modules, project):
        import bpy

        rig = _build(modules, project / "robots" / "arm6.urdf")
        _save(project / "scene.blend")
        copy = project / "robots" / "copy.blend"
        assert "FINISHED" in bpy.ops.wm.save_as_mainfile(filepath=str(copy), copy=True)
        assert _relative(modules, rig) == "//" + os.path.join("robots", "arm6.urdf")
        with bpy.data.libraries.load(str(copy), link=True) as (_, linked):
            linked.objects = [rig.name]
        assert _relative(modules, linked.objects[0]) == "//arm6.urdf", "the copy's own"

    def test_a_save_copy_doesnt_find_another_file_by_the_copys_path(
        self, modules, project, fixture_dir
    ):
        """The copy's ``//arm6.urdf``, read from the open file's folder, names another
        robot's file there."""
        import bpy

        paths = modules("rig.description_path")
        rig = _build(modules, project / "robots" / "arm6.urdf")
        _save(project / "scene.blend")
        shutil.copy(fixture_dir / "arm3.urdf", project / "arm6.urdf")
        copy = project / "robots" / "copy.blend"
        assert "FINISHED" in bpy.ops.wm.save_as_mainfile(filepath=str(copy), copy=True)
        assert paths.source_path(rig) == str(project / "robots" / "arm6.urdf")
        assert _relative(modules, rig) == "//" + os.path.join("robots", "arm6.urdf")

    def test_a_save_that_fails_leaves_the_open_files_own(self, modules, project):
        import bpy

        rig = _build(modules, project / "robots" / "arm6.urdf")
        _save(project / "scene.blend")
        nowhere = project / "robots" / "no such folder" / "scene.blend"
        with pytest.raises(RuntimeError, match="Cannot open file"):
            bpy.ops.wm.save_as_mainfile(filepath=str(nowhere))
        assert _relative(modules, rig) == "//" + os.path.join("robots", "arm6.urdf")


class TestMoving:
    def test_a_project_copied_as_a_whole_finds_its_own_robot(self, modules, project, tmp_path):
        """Both copies are there: the one beside the .blend is the one it reads."""
        builder, paths = modules("rig.builder"), modules("rig.description_path")
        locate, manager = modules("ops.locate"), modules("solver.manager")
        _build(modules, project / "robots" / "arm6.urdf")
        _save(project / "scene.blend")
        moved = tmp_path / "elsewhere" / "moved"
        shutil.copytree(project, moved)
        rig = _open(moved / "scene.blend", modules)
        assert os.path.isfile(rig[builder.PROP_SOURCE]), "precondition: the old one is there too"

        found = str(moved / "robots" / "arm6.urdf")
        assert paths.source_path(rig) == found
        assert not locate.description_missing(rig)
        assert manager._load_source_urdf(rig) is not None
        original = rig[builder.PROP_SOURCE]
        _save(moved / "scene.blend")
        assert rig[builder.PROP_SOURCE] == original, "the file it names is there: kept"

    def test_a_project_moved_finds_its_robot_with_the_old_place_gone(
        self, modules, project, tmp_path
    ):
        builder = modules("rig.builder")
        locate, manager = modules("ops.locate"), modules("solver.manager")
        _build(modules, project / "robots" / "arm6.urdf")
        _save(project / "scene.blend")
        moved = tmp_path / "moved"
        shutil.move(str(project), str(moved))
        rig = _open(moved / "scene.blend", modules)
        assert not locate.description_missing(rig)
        assert manager._load_source_urdf(rig) is not None
        _save(moved / "scene.blend")
        found = str(moved / "robots" / "arm6.urdf")
        assert rig[builder.PROP_SOURCE] == found, "brought up to date, for Kinema 0.6"

    def test_a_blend_moved_on_its_own_finds_the_description_where_it_was(
        self, modules, project, tmp_path
    ):
        paths, locate = modules("rig.description_path"), modules("ops.locate")
        _build(modules, project / "robots" / "arm6.urdf")
        _save(project / "scene.blend")
        alone = tmp_path / "alone" / "scene.blend"
        alone.parent.mkdir()
        shutil.move(str(project / "scene.blend"), str(alone))
        rig = _open(alone, modules)
        assert paths.source_path(rig) == str(project / "robots" / "arm6.urdf")
        assert not locate.description_missing(rig)

    def test_a_rig_linked_from_a_library_finds_it_beside_the_library(
        self, modules, project, tmp_path
    ):
        import bpy

        paths, manager = modules("rig.description_path"), modules("solver.manager")
        rig = _build(modules, project / "robots" / "arm6.urdf")
        name = rig.name
        _save(project / "library.blend")
        moved = tmp_path / "moved"
        shutil.move(str(project), str(moved))
        bpy.ops.wm.read_homefile(use_empty=True)
        with bpy.data.libraries.load(str(moved / "library.blend"), link=True) as (_, linked):
            linked.objects = [name]
        rig = linked.objects[0]
        assert rig.library is not None, "precondition: linked"
        assert paths.source_path(rig) == str(moved / "robots" / "arm6.urdf")
        assert manager._load_source_urdf(rig) is not None


class TestLocating:
    def test_a_located_description_is_kept_relative_where_it_can_be(
        self, modules, project, fixture_dir
    ):
        import bpy

        builder = modules("rig.builder")
        rig = _build(modules, project / "robots" / "arm6.urdf")
        _save(project / "scene.blend")
        (project / "robots" / "arm6.urdf").unlink()
        found = project / "found" / "arm6.urdf"
        found.parent.mkdir()
        shutil.copy(fixture_dir / "arm6.urdf", found)
        assert bpy.ops.kinema.locate_description(filepath=str(found)) == {"FINISHED"}
        assert rig[builder.PROP_SOURCE] == str(found)
        assert _relative(modules, rig) == "//" + os.path.join("found", "arm6.urdf")

    def test_a_located_description_elsewhere_drops_the_relative_path(
        self, modules, project, fixture_dir
    ):
        import bpy

        rig = _build(modules, project / "robots" / "arm6.urdf")
        _save(project / "scene.blend")
        (project / "robots" / "arm6.urdf").unlink()
        assert _relative(modules, rig), "precondition: kept relative"
        target = fixture_dir / "arm6.urdf"
        assert bpy.ops.kinema.locate_description(filepath=str(target)) == {"FINISHED"}
        assert _relative(modules, rig) is None

    def test_locating_another_copy_points_at_it_while_the_first_is_there(
        self, modules, project, fixture_dir
    ):
        """Asked for by name with the file still there: the path kept relative names
        the first copy, and would put it back over the one chosen."""
        import bpy

        builder, paths = modules("rig.builder"), modules("rig.description_path")
        rig = _build(modules, project / "robots" / "arm6.urdf")
        _save(project / "scene.blend")
        assert paths.source_path(rig) == str(project / "robots" / "arm6.urdf")
        target = fixture_dir / "arm6.urdf"
        assert bpy.ops.kinema.locate_description(filepath=str(target)) == {"FINISHED"}
        assert paths.source_path(rig) == str(target)
        assert rig[builder.PROP_SOURCE] == str(target)
