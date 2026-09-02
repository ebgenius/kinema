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
    yield scene.kinema
    scene.kinema.catalog_show_all = False
    scene.kinema.catalog_pick = ""


def item_keys(props) -> set[str]:
    return {key for key, _label, _description in import_robot._catalog_items(None, bpy.context)}


def test_the_popup_lists_the_catalogue(props):
    assert len(item_keys(props)) > 100


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
