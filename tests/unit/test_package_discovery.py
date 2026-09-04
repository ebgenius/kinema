"""Where Kinema looks for ROS packages, and how far.

Two bugs met here, and the second is why the first could not be fixed alone.

`$(find <sibling>)` in a xacro failed outright: xacrodoc resolves the file's own
package and nothing beside it, so a KUKA arm reaching into `kuka_resources` for
its materials never loaded.

The obvious fix -- hand the search a few ancestor directories -- turns package
indexing into a full-disk scan for any description saved somewhere shallow,
because the ancestor list reaches the filesystem root. That does not fail; it
just never returns. So the root has to be bounded, and these are the bounds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ..conftest import load_addon_module

resolve = load_addon_module("io.resolve")

FIXTURE_WS = Path(__file__).resolve().parents[1] / "fixtures" / "ros_ws"


class TestPackageSearchRoot:
    def test_a_package_resolves_to_its_parent(self):
        """Where siblings live. This is the whole fix in one assertion."""
        xacro = FIXTURE_WS / "fixture_robot" / "urdf" / "arm.urdf.xacro"
        assert resolve.package_search_root(xacro) == FIXTURE_WS

    def test_the_sibling_is_findable_from_there(self):
        xacro = FIXTURE_WS / "fixture_robot" / "urdf" / "arm.urdf.xacro"
        index = resolve._index_packages(resolve.package_search_root(xacro))
        assert "fixture_common" in index
        assert "fixture_robot" in index

    def test_it_stops_at_a_checkout_boundary(self, tmp_path):
        """Above a clone is somebody else's disk."""
        repo = tmp_path / "some_repo"
        (repo / ".git").mkdir(parents=True)
        (repo / "urdf").mkdir()
        robot = repo / "urdf" / "robot.urdf"
        robot.write_text("<robot name='r'/>", encoding="utf-8")
        assert resolve.package_search_root(robot) == repo

    def test_a_boundary_wins_over_a_containing_package(self, tmp_path):
        """A repo that is itself one package: its own directory is the root, not
        its parent, which would be wherever the user happened to clone it."""
        repo = tmp_path / "single_package_repo"
        (repo / ".git").mkdir(parents=True)
        (repo / "package.xml").write_text("<package/>", encoding="utf-8")
        (repo / "urdf").mkdir()
        robot = repo / "urdf" / "robot.urdf"
        robot.write_text("<robot name='r'/>", encoding="utf-8")
        assert resolve.package_search_root(robot) == repo

    def test_a_loose_file_uses_its_own_directory(self, tmp_path):
        """No package, no repo -- nothing to index, and certainly no reason to
        walk upwards looking."""
        robot = tmp_path / "robot.urdf"
        robot.write_text("<robot name='r'/>", encoding="utf-8")
        assert resolve.package_search_root(robot) == tmp_path

    def test_it_never_returns_a_filesystem_root(self, tmp_path):
        """The bug that made this bounded. A root here means rglob over an
        entire drive: no error, no result, just a hang."""
        candidates = [
            FIXTURE_WS / "fixture_robot" / "urdf" / "arm.urdf.xacro",
            tmp_path / "robot.urdf",
            Path(tmp_path.anchor) / "robot.urdf",
        ]
        for candidate in candidates:
            root = resolve.package_search_root(candidate)
            assert root != Path(root.anchor), f"{candidate} -> filesystem root"

    def test_a_directory_is_accepted_as_well_as_a_file(self):
        package = FIXTURE_WS / "fixture_robot"
        assert resolve.package_search_root(package) == FIXTURE_WS


class TestIndexing:
    def test_indexing_is_bounded_to_the_root(self, tmp_path):
        """Packages outside the root must not appear, however near they are."""
        inside = tmp_path / "ws" / "pkg_inside"
        outside = tmp_path / "pkg_outside"
        for directory in (inside, outside):
            directory.mkdir(parents=True)
            (directory / "package.xml").write_text("<package/>", encoding="utf-8")

        index = resolve._index_packages(tmp_path / "ws")
        assert "pkg_inside" in index
        assert "pkg_outside" not in index

    def test_a_missing_root_is_not_an_error(self, tmp_path):
        assert resolve._index_packages(tmp_path / "nope") == {}


class TestMeshResolverStillWorks:
    """The resolver is the other caller of the bounded root, and the one that
    was silently exposed: it is lazy, so the drive scan only started once a
    package:// mesh needed resolving."""

    def test_it_resolves_across_sibling_packages(self):
        xacro = FIXTURE_WS / "fixture_robot" / "urdf" / "arm.urdf.xacro"
        resolver = resolve.make_mesh_resolver(xacro)
        found = resolver("package://fixture_common/urdf/common_materials.xacro")
        assert Path(found).is_file(), found

    def test_extra_search_paths_are_still_honoured(self, tmp_path):
        """Explicit caller-supplied roots are scanned even though the automatic
        one is now a single directory."""
        extra_pkg = tmp_path / "elsewhere" / "extra_pkg"
        (extra_pkg / "meshes").mkdir(parents=True)
        (extra_pkg / "package.xml").write_text("<package/>", encoding="utf-8")
        mesh = extra_pkg / "meshes" / "part.dae"
        mesh.write_text("<COLLADA/>", encoding="utf-8")

        robot = tmp_path / "robot.urdf"
        robot.write_text("<robot name='r'/>", encoding="utf-8")

        resolver = resolve.make_mesh_resolver(
            robot, extra_search_paths=[tmp_path / "elsewhere"]
        )
        assert Path(resolver("package://extra_pkg/meshes/part.dae")) == mesh


@pytest.mark.parametrize("depth", [1, 3, 6, 9])
def test_search_root_is_cheap_at_any_depth(tmp_path, depth):
    """Not a timing test -- a shape test. The failure mode was a root returned
    from a deep walk, and it costs nothing to assert the walk terminates
    somewhere sane regardless of how deep the file sits."""
    directory = tmp_path
    for level in range(depth):
        directory = directory / f"level_{level}"
    directory.mkdir(parents=True)
    robot = directory / "robot.urdf"
    robot.write_text("<robot name='r'/>", encoding="utf-8")

    root = resolve.package_search_root(robot)
    assert root != Path(root.anchor)
    assert tmp_path in root.parents or root == tmp_path or root == directory
