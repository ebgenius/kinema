"""Importing an entry from a Catalog Folder (#105): what is read, where its
packages and meshes are found, and why an entry can't be imported."""

from __future__ import annotations

from pathlib import Path

import pytest

from ..conftest import FIXTURE_DIR, catalog_clone, load_addon_module

folder = load_addon_module("catalog.folder")
source = load_addon_module("catalog.source")
loader = load_addon_module("io.loader")
resolve = load_addon_module("io.resolve")


@pytest.fixture
def clone(tmp_path) -> Path:
    return catalog_clone(tmp_path)


def _entry(root: Path, key: str):
    entry = folder.load(root).get(key)
    assert entry is not None, key
    return entry


class TestImportable:
    def test_an_entry_is_its_description_file(self, clone):
        found = source.importable(_entry(clone, "fixture_arm"))
        assert not found.problem
        assert found.path == clone / "src/robot_repo/fixture_robot/urdf/arm.urdf.xacro"
        assert found.xacro_args == ""

    def test_with_the_arguments_its_entry_names(self, clone):
        found = source.importable(_entry(clone, "fixture_named"))
        assert found.xacro_args == "name:=fixture_named segment_length:=0.25"

    def test_a_submodule_not_cloned_says_so_and_how_to(self, clone):
        found = source.importable(_entry(clone, "fixture_absent"))
        assert found.problem == "Fixture Absent: src/absent_repo isn't cloned"
        assert found.command.startswith(f'git -C "{clone}" ')
        assert found.command.endswith("submodule update --init -- src/absent_repo")
        # Anonymous HTTPS, as kinema_catalog's launcher clones: .gitmodules names
        # SSH URLs.
        assert 'url."https://github.com/".insteadOf=git@github.com:' in found.command

    def test_a_submodule_with_no_directory_isnt_cloned_either(self, clone):
        (clone / "src" / "absent_repo").rmdir()
        assert source.importable(_entry(clone, "fixture_absent")).problem.endswith(
            "isn't cloned"
        )

    def test_an_unpacked_zip_has_nothing_to_clone_with(self, tmp_path):
        root = catalog_clone(tmp_path, git=False)
        found = source.importable(_entry(root, "fixture_absent"))
        assert found.problem == f"Fixture Absent: src/absent_repo isn't in {root}"
        assert found.command == ""

    def test_a_file_the_submodule_doesnt_have(self, clone):
        (clone / "src/tool_repo/urdf/tool.urdf").unlink()
        found = source.importable(_entry(clone, "fixture_tool"))
        assert found.problem == "Fixture Tool: no src/tool_repo/urdf/tool.urdf"


class TestContaining:
    def test_an_entrys_file_is_in_its_catalog(self, clone):
        path = clone / "src/robot_repo/fixture_robot/urdf/arm.urdf.xacro"
        root, entry = folder.containing(path)
        assert root == clone.resolve()
        assert entry.id == "fixture_arm"

    def test_a_file_no_entry_names_is_still_in_the_catalog(self, clone):
        path = clone / "src/common_repo/fixture_common/urdf/common_materials.xacro"
        assert folder.containing(path) == (clone.resolve(), None)

    def test_a_file_in_no_catalog(self):
        assert folder.containing(FIXTURE_DIR / "arm6.urdf") is None

    def test_a_catalog_json_that_isnt_an_index_is_passed_over(self, clone):
        (clone / "src" / "robot_repo" / "catalog.json").write_text("{}", encoding="utf-8")
        path = clone / "src/robot_repo/fixture_robot/urdf/arm.urdf.xacro"
        assert folder.containing(path)[1].id == "fixture_arm"


class TestPackages:
    def test_a_package_in_another_submodule_is_found(self, clone):
        """fixture_arm includes fixture_common, from common_repo: past the top of
        its own submodule."""
        result = loader.load_file(_entry(clone, "fixture_arm").path)
        assert result.error is None, result.error
        assert {"base_link", "link_1", "link_2"} <= set(result.model.links)

    def test_the_whole_src_is_searched(self, clone):
        path = clone / "src/robot_repo/fixture_robot/urdf/arm.urdf.xacro"
        assert resolve.catalog_sources(path) == [(clone / "src").resolve()]
        assert "fixture_common" in resolve.package_map(path)

    def test_not_twice_where_the_files_own_search_covers_it(self, clone):
        """A wrapper kept at the catalog's top, outside src/, is searched from the
        clone's top already."""
        wrapper = clone / "descriptions" / "wrapper.urdf"
        wrapper.parent.mkdir()
        wrapper.write_text("<robot name='w'/>", encoding="utf-8")
        assert resolve.package_search_root(wrapper) == clone.resolve()
        assert resolve.catalog_sources(wrapper, clone) == []

    def test_outside_a_catalog_nothing_is_added(self):
        assert resolve.catalog_sources(FIXTURE_DIR / "arm6.urdf") == []

    def test_the_entrys_arguments_reach_the_render(self, clone):
        entry = _entry(clone, "fixture_named")
        result = loader.load_file(entry.path, xacro_args=dict(entry.args))
        assert result.error is None, result.error
        assert result.model.name == "fixture_named"


class TestMeshBase:
    def test_a_relative_mesh_starts_at_the_entrys_mesh_base(self, clone):
        result = loader.load_file(_entry(clone, "fixture_tool").path)
        assert result.error is None, result.error
        (visual,) = result.model.links["tool_base"].visuals
        assert Path(visual.mesh_path) == (clone / "src/tool_repo/meshes/part.stl").resolve()
        assert Path(visual.mesh_path).is_file()

    def test_without_one_it_starts_at_the_file(self, clone):
        path = clone / "src/robot_repo/fixture_robot/urdf/arm.urdf.xacro"
        assert resolve.mesh_base(path) is None
        resolver = resolve.make_mesh_resolver(path)
        assert Path(resolver("meshes/x.stl")) == path.resolve().parent / "meshes" / "x.stl"

    def test_a_file_no_entry_names_has_none(self, clone):
        path = clone / "src/common_repo/fixture_common/urdf/common_materials.xacro"
        assert resolve.mesh_base(path) is None
