"""Importing from a Catalog Folder, in Blender (#105): a pick from the search
is a rig, and the folder can be added to the Asset Browser."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

pytest.importorskip("bpy")

import bpy  # noqa: E402

from ..conftest import catalog_clone, load_addon_module  # noqa: E402

builder = load_addon_module("rig.builder")
import_robot = load_addon_module("ops.import_robot")


@pytest.fixture
def props(addon, clean_scene):
    kinema = bpy.context.scene.kinema
    before = {name: getattr(kinema, name) for name in import_robot.SETTING_NAMES}
    kinema.catalog_pick = ""
    kinema.catalog_pick_folder = ""
    yield kinema
    for name, value in before.items():
        setattr(kinema, name, value)
    kinema.catalog_pick = ""
    kinema.catalog_pick_folder = ""


@pytest.fixture
def preferences(addon, props):
    found = importlib.import_module(f"{addon.__name__}.prefs").get_prefs()
    before = found.catalog_folder
    yield found
    found.catalog_folder = before


@pytest.fixture
def clone(preferences, tmp_path) -> Path:
    root = catalog_clone(tmp_path)
    preferences.catalog_folder = str(root)
    return root


def _rigs() -> list:
    return [
        obj for obj in bpy.context.scene.objects
        if obj.type == "ARMATURE" and builder.PROP_SOURCE in obj
    ]


class TestPick:
    def test_a_pick_is_a_rig(self, clone):
        """The search's two clicks: open it, and pick."""
        assert bpy.ops.kinema.browse_catalog(robot_key="fixture_arm") == {"FINISHED"}
        (rig,) = _rigs()
        assert Path(rig[builder.PROP_SOURCE]) == (
            clone / "src/robot_repo/fixture_robot/urdf/arm.urdf.xacro"
        )

    def test_the_solver_reloads_it_with_no_folder_set(self, clone, preferences):
        """Its packages are found from the file's place, not the preferences:
        fixture_arm needs one from another submodule."""
        manager = load_addon_module("solver.manager")
        bpy.ops.kinema.browse_catalog(robot_key="fixture_arm")
        (rig,) = _rigs()
        preferences.catalog_folder = ""
        urdf = manager._load_source_urdf(rig)
        assert {"joint_1", "joint_2"} <= {joint.name for joint in urdf.robot.joints}

    def test_with_the_arguments_its_entry_names(self, clone, props):
        """Not the sidebar's, which are for a file imported by hand; and kept on
        the rig, for the reload."""
        props.xacro_args = "name:=not_this"
        bpy.ops.kinema.browse_catalog(robot_key="fixture_named")
        (rig,) = _rigs()
        assert rig[builder.PROP_ROBOT_NAME] == "fixture_named"
        assert rig[builder.PROP_XACRO_ARGS] == "name:=fixture_named segment_length:=0.25"

    def test_with_the_sidebars_import_options(self, clone, props):
        props.create_tcp = False
        bpy.ops.kinema.browse_catalog(robot_key="fixture_arm")
        (rig,) = _rigs()
        assert not rig.get(builder.PROP_TCP_BONE)

    def test_an_end_effector_comes_with_its_mesh(self, clone):
        """Named relative to its repository's top, as its mesh_base says."""
        bpy.ops.kinema.browse_catalog(robot_key="fixture_tool")
        (rig,) = _rigs()
        meshes = [obj for obj in rig.children_recursive if obj.type == "MESH"]
        assert meshes
        assert all(obj.data.vertices for obj in meshes)

    def test_a_submodule_not_cloned_says_how_to_clone_it(self, clone, props):
        with pytest.raises(
            RuntimeError,
            match="src/absent_repo isn't cloned: the command to clone it is on the clipboard",
        ):
            bpy.ops.kinema.browse_catalog(robot_key="fixture_absent")
        assert not _rigs()
        # Still the pick, to import once cloned.
        assert import_robot.picked_entry(bpy.context).id == "fixture_absent"

    def test_import_imports_the_pick_again(self, clone):
        bpy.ops.kinema.browse_catalog(robot_key="fixture_tool")
        assert bpy.ops.kinema.import_catalog_pick() == {"FINISHED"}
        assert len(_rigs()) == 2

    def test_import_needs_a_pick_from_the_folder(self, clone, preferences):
        assert not bpy.ops.kinema.import_catalog_pick.poll()
        bpy.ops.kinema.browse_catalog(robot_key="fixture_tool")
        assert bpy.ops.kinema.import_catalog_pick.poll()
        preferences.catalog_folder = ""
        assert not bpy.ops.kinema.import_catalog_pick.poll()

    def test_the_panel_offers_to_import_the_pick(self, clone):
        panel = load_addon_module("ui.panel")
        bpy.ops.kinema.browse_catalog(robot_key="fixture_tool")
        calls = []
        from .test_browse_catalog import _Recorder

        panel._draw_catalog(_Recorder(calls), bpy.context)
        assert ("kinema.import_catalog_pick",) in [
            args for name, args, _kwargs in calls if name == "operator"
        ]


class TestAssetLibrary:
    @pytest.fixture
    def libraries(self, addon):
        libraries = bpy.context.preferences.filepaths.asset_libraries
        before = {library.name for library in libraries}
        yield libraries
        for library in list(libraries):
            if library.name not in before:
                libraries.remove(library)

    def _ours(self, libraries) -> list:
        prefs = load_addon_module("prefs")
        return [library for library in libraries if library.name == prefs.CATALOG_LIBRARY]

    def test_it_adds_the_folder_as_a_library(self, clone, libraries):
        assert bpy.ops.kinema.add_catalog_library() == {"FINISHED"}
        (library,) = self._ours(libraries)
        assert Path(library.path) == clone
        # Appended, so what is dropped is objects to pose, not an instance.
        assert library.import_method == "APPEND"

    def test_once_only_and_moved_with_the_folder(self, clone, preferences, libraries, tmp_path):
        bpy.ops.kinema.add_catalog_library()
        moved = tmp_path / "moved"
        moved.mkdir()
        preferences.catalog_folder = str(moved)
        bpy.ops.kinema.add_catalog_library()
        (library,) = self._ours(libraries)
        assert Path(library.path) == moved

    def test_the_preferences_say_it_is_added(self, clone, preferences, libraries):
        from .test_browse_catalog import _texts

        assert not any("Asset Browser" in text for text in _texts(preferences._draw_catalog_folder))
        bpy.ops.kinema.add_catalog_library()
        assert "In the Asset Browser as Kinema Catalog" in _texts(
            preferences._draw_catalog_folder
        )

    def test_it_needs_a_folder(self, preferences, libraries):
        preferences.catalog_folder = ""
        assert not bpy.ops.kinema.add_catalog_library.poll()
