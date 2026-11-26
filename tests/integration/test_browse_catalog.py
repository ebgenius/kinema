"""The catalogue picker operator: what the popup lists, and what it hands back.

The picker is the only remaining catalogue UI, and its two filters -- supported
and curated-out -- have to move together. They did not at first: an entry with
no resolved file path is *also* unsupported, so the Show All Variants toggle
could never reveal one however it was set.
"""

from __future__ import annotations

import pytest

pytest.importorskip("bpy")

import bpy  # noqa: E402

from ..conftest import load_addon_module  # noqa: E402

import_robot = load_addon_module("ops.import_robot")
catalog = load_addon_module("catalog.index")


@pytest.fixture
def props(addon, clean_scene):
    scene = bpy.context.scene
    scene.kinema.catalog_show_all = False
    scene.kinema.catalog_pick = ""
    scene.kinema.catalog_pick_folder = ""
    yield scene.kinema
    scene.kinema.catalog_show_all = False
    scene.kinema.catalog_pick = ""
    scene.kinema.catalog_pick_folder = ""


def item_keys(props) -> set[str]:
    return {key for key, _label, _description in import_robot._catalog_items(None, bpy.context)}


def test_the_popup_lists_the_catalogue(props):
    assert len(item_keys(props)) > 100


def test_the_category_filter_offers_every_tag(props):
    """Set through Blender, which checks the value against the items callback it
    registered. A lambda there could not see the module it was written in, so
    it offered nothing and every tag was refused."""
    tags = catalog.available_tags()
    assert tags
    for tag in tags:
        props.catalog_tag = tag
        assert props.catalog_tag == tag


def test_curated_out_entries_are_hidden_by_default(props):
    hidden = {e.key for e in catalog.all_entries() if e.is_curated_out}
    assert hidden, "curation.json should cross out at least one entry"
    assert not (hidden & item_keys(props))


def test_show_all_variants_reveals_every_one_of_them(props):
    """The bug: eve_r3_description is curated out *and* has no file path, so
    the toggle had to lift both filters or it revealed nothing."""
    props.catalog_show_all = True
    keys = item_keys(props)
    hidden = {e.key for e in catalog.all_entries() if e.is_curated_out}
    assert hidden <= keys

    unresolved = {e.key for e in catalog.all_entries() if e.file_path is None}
    assert unresolved, "expected at least one entry with no resolved file path"
    assert unresolved <= keys


def test_labels_carry_the_curation_status(props):
    props.catalog_show_all = True
    labels = {
        key: label for key, label, _ in import_robot._catalog_items(None, bpy.context)
    }
    for entry in catalog.all_entries():
        if entry.is_curated_out:
            assert entry.status in labels[entry.key]


class TestPicking:
    """What the operator does. The clipboard itself cannot be asserted here:
    a background Blender has no window system, so ``window_manager.clipboard``
    reads back empty however it was set. The command's *content* is covered
    without bpy in tests/unit/test_catalog.py."""

    def test_records_the_pick_for_the_panel(self, props):
        assert bpy.ops.kinema.browse_catalog(robot_key="panda_description") == {
            "FINISHED"
        }
        assert props.catalog_pick == "panda_description"

    def test_an_entry_without_a_file_path_is_still_pickable(self, props):
        """Reachable only with the toggle on -- and it must not fail there."""
        props.catalog_show_all = True
        assert catalog.get("eve_r3_description").file_path is None
        assert bpy.ops.kinema.browse_catalog(robot_key="eve_r3_description") == {
            "FINISHED"
        }
        assert props.catalog_pick == "eve_r3_description"

    def test_open_repository_needs_a_pick(self, props):
        with pytest.raises(RuntimeError, match="No robot picked"):
            bpy.ops.kinema.open_catalog_repo()


class _Recorder:
    """Anything drawn on a layout, recorded as (call, args, kwargs)."""

    def __init__(self, calls):
        self.__dict__["calls"] = calls

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return _Recorder(self.calls)

        return call

    def __setattr__(self, name, value):
        pass


def _texts(draw, *args) -> list[str]:
    calls = []
    draw(_Recorder(calls), *args)
    return [kwargs.get("text", "") for name, _args, kwargs in calls if name == "label"]


class TestCatalogFolder:
    """With a Catalog Folder set, the search lists its index instead (#105)."""

    @pytest.fixture
    def preferences(self, addon, props):
        import importlib

        found = importlib.import_module(f"{addon.__name__}.prefs").get_prefs()
        before = found.catalog_folder
        yield found
        found.catalog_folder = before

    @pytest.fixture
    def clone(self, preferences, fixture_dir):
        preferences.catalog_folder = str(fixture_dir / "catalog_folder" / "clone")
        return preferences

    def test_the_popup_lists_the_folders_entries(self, clone, props):
        assert item_keys(props) == {"irb1200", "so100", "aura", "ur5e", "gripper_2f"}

    def test_an_item_shows_kind_format_and_licence(self, clone):
        items = {key: (label, text) for key, label, text in
                 import_robot._catalog_items(None, bpy.context)}
        assert items["gripper_2f"] == (
            "Two-Finger Gripper",
            "gripper_2f · urdf — End Effector · arm · gripper · BSD-3-Clause",
        )

    def test_unpacked_zips_are_listed_together(self, preferences, fixture_dir, props):
        preferences.catalog_folder = str(fixture_dir / "catalog_folder" / "zips")
        assert item_keys(props) == {"irb1200", "ur5e", "gripper_2f"}

    def test_picking_records_the_folders_entry(self, clone, props, fixture_dir):
        assert bpy.ops.kinema.browse_catalog(robot_key="ur5e") == {"FINISHED"}
        assert props.catalog_pick == "ur5e"
        entry = import_robot.picked_entry(bpy.context)
        assert entry.path == (
            fixture_dir / "catalog_folder" / "clone"
            / "src/Universal_Robots_ROS2_Description/urdf/ur.urdf.xacro"
        )

    def test_a_bundled_robot_isnt_pickable_from_the_folder(self, clone):
        with pytest.raises(TypeError):
            # Not one of the enum's items while the folder is listed.
            bpy.ops.kinema.browse_catalog(robot_key="panda_description")

    def test_there_is_no_repository_to_open(self, clone):
        assert not bpy.ops.kinema.open_catalog_repo.poll()

    def test_the_panel_shows_the_pick_from_the_folder(self, clone, props):
        panel = load_addon_module("ui.panel")
        assert bpy.ops.kinema.browse_catalog(robot_key="aura") == {"FINISHED"}
        texts = _texts(panel._draw_catalog, bpy.context)
        assert texts[0] == "Catalog Folder: 5 entries"
        assert "Comau AURA" in texts
        assert "Robot · arm · MIT" in texts
        assert any("BSD-2-Clause" in text for text in texts)
        assert any(text.endswith("aura_robot.urdf.xacro") for text in texts)
        assert "Kinema does not download robots." not in texts

    def test_the_file_shown_goes_through_its_unpacked_zip(
        self, preferences, fixture_dir, props
    ):
        preferences.catalog_folder = str(fixture_dir / "catalog_folder" / "zips")
        assert bpy.ops.kinema.browse_catalog(robot_key="ur5e") == {"FINISHED"}
        panel = load_addon_module("ui.panel")
        assert (
            "universal_robots/src/Universal_Robots_ROS2_Description/urdf/ur.urdf.xacro"
            in _texts(panel._draw_catalog, bpy.context)
        )

    def test_a_pick_belongs_to_the_list_it_was_picked_from(
        self, clone, preferences, fixture_dir, props
    ):
        """A pick is no pick once another list is searched, even where its id is
        in both."""
        assert bpy.ops.kinema.browse_catalog(robot_key="ur5e") == {"FINISHED"}
        assert import_robot.picked_entry(bpy.context).id == "ur5e"
        preferences.catalog_folder = str(fixture_dir / "catalog_folder" / "zips")
        assert import_robot.picked_entry(bpy.context) is None
        preferences.catalog_folder = ""
        assert import_robot.picked_entry(bpy.context) is None
        assert bpy.ops.kinema.browse_catalog(robot_key="panda_description") == {"FINISHED"}
        assert import_robot.picked_entry(bpy.context).key == "panda_description"
        preferences.catalog_folder = str(fixture_dir / "catalog_folder" / "clone")
        assert import_robot.picked_entry(bpy.context) is None

    def test_a_folder_pick_is_never_read_from_the_bundled_list(
        self, clone, preferences, props
    ):
        """A pick made from the folder, of an id only the bundled list has."""
        catalog_folder = import_robot.catalog_folder(bpy.context)
        shared = next(e.key for e in catalog.all_entries() if e.key)
        props.catalog_pick = shared
        props.catalog_pick_folder = str(catalog_folder.path)
        # Not in the folder: no pick, rather than the bundled entry of that id.
        assert import_robot.picked_entry(bpy.context) is None

    def test_the_items_outlive_the_call(self, clone):
        """Blender reads a dynamic enum's strings after its callback returns."""
        items = import_robot._catalog_items(None, bpy.context)
        assert items is import_robot._CATALOG_ITEMS
        assert len(items) == 5

    def test_the_preferences_say_what_was_found(self, clone):
        assert "5 entries from 1 catalog.json" in _texts(clone._draw_catalog_folder)

    def test_a_folder_without_an_index_falls_back_to_the_bundled_list(
        self, preferences, tmp_path, props
    ):
        preferences.catalog_folder = str(tmp_path)
        assert len(item_keys(props)) > 100
        panel = load_addon_module("ui.panel")
        assert "Catalog Folder: nothing to list there" in _texts(
            panel._draw_catalog, bpy.context
        )
        texts = _texts(preferences._draw_catalog_folder)
        assert "Nothing to list here: the bundled list is searched." in texts
        assert any("no catalog.json here or one level down" in text for text in texts)

    def test_unset_it_says_where_to_get_the_catalog(self, preferences, props):
        preferences.catalog_folder = ""
        panel = load_addon_module("ui.panel")
        assert "Catalog Folder: get it from github.com/ebgenius/kinema_catalog" in _texts(
            panel._draw_catalog, bpy.context
        )
        assert bpy.ops.kinema.open_catalog_repo.poll()
