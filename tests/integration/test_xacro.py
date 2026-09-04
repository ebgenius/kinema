"""Importing a xacro from disk.

38 of the catalogue's 186 robots ship only a xacro, and since Kinema stopped
downloading descriptions this is the path a user takes to reach any of them.
It had no coverage at all, and was broken on Windows the whole time.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ..conftest import load_addon_module

loader = load_addon_module("io.loader")


@pytest.fixture(scope="module")
def xacro_path(fixture_dir):
    return fixture_dir / "kinema_fixture" / "urdf" / "arm.urdf.xacro"


@pytest.fixture(scope="module")
def ws_robot(fixture_dir):
    """A robot in a two-package workspace, reaching across to its sibling."""
    return fixture_dir / "ros_ws" / "fixture_robot" / "urdf" / "arm.urdf.xacro"


def test_xacro_renders_and_parses(xacro_path):
    """Macros, properties and $(find) all have to survive the render.

    The fixture's three joints only exist as one xacro macro invoked three
    times, so a model with three joints proves the render really ran rather
    than the file being parsed as plain XML.
    """
    result = loader.load_file(xacro_path)
    assert result.error is None, result.error
    assert result.model is not None
    assert len(result.model.joints) == 3
    assert {j.name for j in result.model.joints} == {"joint_1", "joint_2", "joint_3"}


def test_the_render_is_not_left_behind(xacro_path):
    """The rendered URDF is a temp file and must be cleaned up.

    xacrodoc's own temp_urdf_file_path holds the file open while yielding its
    path, which Windows refuses to reopen -- so loader renders it by hand. That
    makes cleanup ours to get right too.
    """
    temp_dir = Path(tempfile.gettempdir())
    before = set(temp_dir.glob("kinema-*.urdf"))
    loader.load_file(xacro_path)
    assert set(temp_dir.glob("kinema-*.urdf")) == before


def test_source_is_recorded_as_the_xacro_not_the_render(xacro_path):
    """The rig has to remember the file the user picked: the render is gone by
    the time the solver wants to reload it."""
    result = loader.load_file(xacro_path)
    assert result.source == ("file", str(xacro_path))


class TestCrossPackageIncludes:
    """`$(find <sibling package>)`, which is how real vendor descriptions are
    laid out and which failed outright until 0.3.2.

    kroshu/kuka_robot_descriptions is the case that surfaced it: an LBR iiwa in
    `kuka_lbr_iiwa_support` includes materials from `kuka_resources` beside it,
    and the import died with `PackageNotFoundError: kuka_resources` even though
    the package was sitting in the same clone. The fixture is that shape,
    reduced.
    """

    def test_a_sibling_package_include_resolves(self, ws_robot):
        result = loader.load_file(ws_robot)
        assert result.error is None, result.error
        assert result.model is not None

    def test_the_macro_from_the_sibling_actually_ran(self, ws_robot):
        """A resolved include that produced nothing would still parse. The links
        only exist because a macro defined in the *other* package expanded."""
        model = loader.load_file(ws_robot).model
        assert {"base_link", "link_1", "link_2"} <= set(model.links)

    def test_the_joints_survive_the_render(self, ws_robot):
        model = loader.load_file(ws_robot).model
        assert [j.name for j in model.actuated_joints] == ["joint_1", "joint_2"]

    def test_packages_do_not_leak_between_imports(self, ws_robot, fixture_dir):
        """xacrodoc's package finder is module-global, so one import's packages
        stay resolvable in the next unless it is reset.

        `orphan_ws` exists to make that visible. Its robot needs
        `fixture_common`, which lives only in `ros_ws` -- so importing ros_ws
        first and orphan_ws second must still fail. Without the reset it would
        succeed, silently rendering one workspace's robot against another's
        materials.

        Asserting the failure is the only way to catch this: two workspaces with
        different package names both resolve either way, which is why the first
        version of this test passed with the reset removed.
        """
        assert loader.load_file(ws_robot).error is None, "ros_ws should import"

        orphan = fixture_dir / "orphan_ws" / "orphan_robot" / "urdf" / "arm.urdf.xacro"
        result = loader.load_file(orphan)

        assert result.error is not None, (
            "orphan_ws resolved fixture_common, which it does not contain -- "
            "packages leaked from the previous import"
        )
        assert "fixture_common" in result.error

    def test_the_orphan_fails_for_the_right_reason(self, fixture_dir):
        """The other half. `orphan_ws` must fail *because its package is
        missing*, not because the file is malformed -- otherwise the leak test
        above passes on a broken fixture and detects nothing."""
        orphan = fixture_dir / "orphan_ws" / "orphan_robot" / "urdf" / "arm.urdf.xacro"
        error = loader.load_file(orphan).error
        assert error is not None
        assert "fixture_common" in error, error

    def test_importing_twice_gives_the_same_answer(self, ws_robot, fixture_dir):
        first = loader.load_file(ws_robot)
        loader.load_file(fixture_dir / "kinema_fixture" / "urdf" / "arm.urdf.xacro")
        again = loader.load_file(ws_robot)

        assert first.error is None and again.error is None
        assert [j.name for j in first.model.actuated_joints] == [
            j.name for j in again.model.actuated_joints
        ]

    def test_the_solver_can_reload_a_xacro_rig(self, ws_robot):
        """The solver reloads the description later to build PyRoki's model, and
        used to hand the xacro to yourdfpy raw -- so `$(arg …)` hit a float
        parser and the rig fell back to NumPy with "could not convert string to
        float: '$(arg'". Silently, because falling back is what the manager does
        with any reload failure.

        Checked at the reload seam rather than through a solve, so it holds even
        where the JAX stack is unavailable -- but the seam lives behind
        ``rig.builder``, which needs bpy, so this one is Blender-tier only.
        """
        pytest.importorskip("bpy")

        from ..conftest import load_addon_module

        manager = load_addon_module("solver.manager")

        class FakeRig(dict):
            def get(self, key, default=None):
                return dict.get(self, key, default)

        builder_mod = load_addon_module("rig.builder")
        rig = FakeRig({
            builder_mod.PROP_SOURCE_KIND: "file",
            builder_mod.PROP_SOURCE: str(ws_robot),
        })

        urdf = manager._load_source_urdf(rig)
        assert urdf is not None
        names = {j.name for j in urdf.robot.joints}
        assert {"joint_1", "joint_2"} <= names

    def test_the_reload_uses_the_arguments_stored_on_the_rig(self, fixture_dir):
        """A description that needs an argument needs it on reload too.

        Without the stored arguments this raises exactly as the import would
        have -- and the manager answers any reload failure by falling back to
        NumPy without saying so, which is the whole reason PROP_XACRO_ARGS
        exists. Asserting the rendered name proves the value arrived rather
        than merely that something parsed.
        """
        pytest.importorskip("bpy")

        from ..conftest import load_addon_module

        manager = load_addon_module("solver.manager")
        builder_mod = load_addon_module("rig.builder")

        needs_args = (
            fixture_dir / "ros_ws" / "Renamed_Package_Dir"
            / "urdf" / "needs_args.urdf.xacro"
        )

        class FakeRig(dict):
            def get(self, key, default=None):
                return dict.get(self, key, default)

        rig = FakeRig({
            builder_mod.PROP_SOURCE_KIND: "file",
            builder_mod.PROP_SOURCE: str(needs_args),
            builder_mod.PROP_XACRO_ARGS: "name:=reloaded_arm",
        })
        urdf = manager._load_source_urdf(rig)
        assert urdf is not None
        assert urdf.robot.name == "reloaded_arm"

    def test_the_reload_fails_loudly_without_them(self, fixture_dir):
        """The other half: drop the property and the reload must raise rather
        than quietly produce something. If this ever passes, the test above is
        no longer proving anything."""
        pytest.importorskip("bpy")

        from ..conftest import load_addon_module

        manager = load_addon_module("solver.manager")
        builder_mod = load_addon_module("rig.builder")

        needs_args = (
            fixture_dir / "ros_ws" / "Renamed_Package_Dir"
            / "urdf" / "needs_args.urdf.xacro"
        )

        class FakeRig(dict):
            def get(self, key, default=None):
                return dict.get(self, key, default)

        rig = FakeRig({
            builder_mod.PROP_SOURCE_KIND: "file",
            builder_mod.PROP_SOURCE: str(needs_args),
        })
        with pytest.raises(manager.SolverError):
            manager._load_source_urdf(rig)

    def test_a_package_outside_the_checkout_is_not_found(self, ws_robot):
        """The bound is real, not decorative: resolution stops at the workspace
        rather than searching upwards until something matches."""
        from ..conftest import load_addon_module

        resolve = load_addon_module("io.resolve")
        index = resolve._index_packages(resolve.package_search_root(ws_robot))
        assert "kinema_fixture" not in index, (
            "the single-package fixture lives outside ros_ws and must not be "
            "reachable from inside it"
        )
