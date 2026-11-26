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


def _rewrite(root: Path, key: str, **fields) -> None:
    """Change fields of one entry in ``root``'s index."""
    import json

    index = root / "catalog.json"
    data = json.loads(index.read_text(encoding="utf-8"))
    (record,) = [record for record in data["entries"] if record["id"] == key]
    record.update(fields)
    # Longer by a field's worth at least, so the cached load is read again.
    data["generated_from"] = data.get("generated_from", "") + " rewritten"
    index.write_text(json.dumps(data), encoding="utf-8")
    assert all(getattr(_entry(root, key), name) == value for name, value in fields.items())


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

    @pytest.mark.parametrize(
        "submodule", ["absent; echo unexpected #", "absent && calc", "$(calc)", "a b", "a%PATH%"]
    )
    def test_a_submodule_a_shell_would_read_more_into_has_no_command(self, clone, submodule):
        """The command is pasted into a shell: one the catalog names is never
        anything but git's."""
        _rewrite(clone, "fixture_absent", submodule=submodule)
        found = source.importable(_entry(clone, "fixture_absent"))
        assert found.problem.endswith("isn't cloned")
        assert found.command == ""

    @pytest.mark.parametrize("name", ["cat$HOME", "cat`x`", 'cat"x', "cat%x%", "cat!x"])
    def test_nor_a_catalog_in_a_directory_a_shell_would_expand(self, tmp_path, name):
        directory = tmp_path / name
        try:
            directory.mkdir()
        except OSError:
            pytest.skip(f"{name} can't be a directory's name here")
        found = source.importable(_entry(catalog_clone(directory), "fixture_absent"))
        assert found.problem.endswith("isn't cloned")
        assert found.command == ""

    def test_a_directory_with_spaces_has_its_command(self, tmp_path):
        directory = tmp_path / "my catalogs (2)"
        directory.mkdir()
        root = catalog_clone(directory)
        found = source.importable(_entry(root, "fixture_absent"))
        assert found.command.startswith(f'git -C "{root}" ')

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("entry", "/etc/robot.urdf"),
            ("entry", "C:/robots/robot.urdf"),
            ("entry", "../elsewhere/robot.urdf"),
            ("entry", "src/tool_repo/../../../robot.urdf"),
            ("mesh_base", "/etc"),
            ("mesh_base", ".."),
            ("submodule", "../../elsewhere"),
        ],
    )
    def test_a_path_outside_the_catalog_isnt_followed(self, clone, field, value):
        _rewrite(clone, "fixture_tool", **{field: value})
        found = source.importable(_entry(clone, "fixture_tool"))
        assert found.problem == f"Fixture Tool: its {field} {value} is outside the catalog"

    def test_a_submodule_that_cant_be_looked_into_isnt_called_uncloned(
        self, clone, monkeypatch
    ):
        unreadable = clone / "src" / "tool_repo"
        real = Path.iterdir

        def iterdir(self):
            if self == unreadable:
                raise PermissionError(13, "Access is denied", str(self))
            return real(self)

        monkeypatch.setattr(Path, "iterdir", iterdir)
        found = source.importable(_entry(clone, "fixture_tool"))
        assert found.problem.startswith("Fixture Tool: src/tool_repo can't be read: ")
        assert "Access is denied" in found.problem
        assert found.command == ""


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

    def test_an_entry_that_is_a_link_is_found_through_it(self, clone):
        """Its file resolves elsewhere in the catalog: the entry is still found,
        and with it the entry's mesh_base."""
        link = clone / "descriptions" / "tool.urdf"
        link.parent.mkdir()
        try:
            link.symlink_to(clone / "src/tool_repo/urdf/tool.urdf")
        except OSError:
            pytest.skip("symbolic links can't be made here")
        _rewrite(clone, "fixture_tool", entry="descriptions/tool.urdf")
        assert folder.containing(link)[1].id == "fixture_tool"
        assert resolve.mesh_base(link) == clone.resolve() / "src/tool_repo"


class TestWithin:
    @pytest.mark.parametrize(
        "path", ["src/a/b.urdf", "", ".", "src/a/../b.urdf", "descriptions/x.xacro"]
    )
    def test_inside(self, path):
        assert folder.within(path)

    @pytest.mark.parametrize(
        "path",
        ["/etc/x.urdf", "C:/x.urdf", "C:x.urdf", "\\\\server\\share\\x.urdf", "..", "a/../../x"],
    )
    def test_outside(self, path):
        assert not folder.within(path)


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

    def test_an_empty_mesh_base_is_the_catalogs_top(self, clone):
        _rewrite(clone, "fixture_tool", mesh_base="")
        assert resolve.mesh_base(clone / "src/tool_repo/urdf/tool.urdf") == clone.resolve()

    def test_one_outside_the_catalog_isnt_followed(self, clone):
        _rewrite(clone, "fixture_tool", mesh_base="../..")
        assert resolve.mesh_base(clone / "src/tool_repo/urdf/tool.urdf") is None

    def test_a_file_no_entry_names_has_none(self, clone):
        path = clone / "src/common_repo/fixture_common/urdf/common_materials.xacro"
        assert resolve.mesh_base(path) is None
