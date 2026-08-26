"""Catalog browsing. Reads robot_descriptions' metadata table -- no network."""

from __future__ import annotations

import pytest

from ..conftest import load_addon_module

catalog = load_addon_module("catalog.index")


@pytest.fixture(scope="module")
def entries():
    result = catalog.all_entries(refresh=True)
    if not result:
        pytest.skip("robot_descriptions is not installed")
    return result


def test_catalog_is_populated(entries):
    assert len(entries) > 100


def test_entries_carry_display_metadata(entries):
    ur5e = catalog.get("ur5e_description")
    assert ur5e is not None
    assert "arm" in ur5e.tags
    assert ur5e.has_urdf
    assert ur5e.maker == "Universal Robots"
    assert "UR5e" in ur5e.label
    assert "BSD-3-Clause" in ur5e.description


def test_dof_is_omitted_when_the_catalog_does_not_record_it(entries):
    """Not every entry has a dof value upstream (ur5e_description is 0), so the
    description line must not advertise '0 DoF'."""
    missing = [e for e in entries if e.dof == 0]
    assert missing, "expected at least one entry without a recorded dof"
    assert "DoF" not in missing[0].description


def test_dof_is_shown_when_recorded(entries):
    recorded = [e for e in entries if e.dof > 0]
    assert recorded
    assert f"{recorded[0].dof} DoF" in recorded[0].description


def test_sorted_by_maker_then_robot(entries):
    keys = [(e.maker.lower(), e.robot.lower()) for e in entries]
    assert keys == sorted(keys)


class TestSearch:
    def test_free_text_matches_key_robot_or_maker(self):
        assert any(e.key == "ur5e_description" for e in catalog.search("ur5e"))
        assert catalog.search("ur5e") == catalog.search("UR5E"), "should be case-insensitive"

    def test_tag_filter(self):
        arms = catalog.search(tag="arm")
        assert arms
        assert all("arm" in e.tags for e in arms)

    def test_supported_formats_are_the_default(self):
        """Kinema reads URDF and MJCF, so the picker should offer both."""
        results = catalog.search()
        assert all(e.is_supported for e in results)
        assert any(e.has_urdf for e in results)
        assert any(e.has_mjcf and not e.has_urdf for e in results), (
            "MJCF-only robots should be importable"
        )

    def test_nonsense_query_returns_nothing(self):
        assert catalog.search("zzzz-no-such-robot") == []


def test_available_tags_are_ordered_sensibly(entries):
    tags = catalog.available_tags()
    assert "arm" in tags
    # Known tags come first, in KNOWN_TAGS order, before any unexpected ones.
    known = [t for t in tags if t in catalog.KNOWN_TAGS]
    assert known == [t for t in catalog.KNOWN_TAGS if t in tags]


def test_mjcf_only_entries_are_supported_and_labelled(entries):
    """60 of the catalog's 186 robots ship MJCF only -- roughly a third of the
    total, so treating them as unimportable would be a large hole."""
    mjcf_only = [e for e in entries if e.has_mjcf and not e.has_urdf]
    assert len(mjcf_only) > 20
    assert all(e.is_supported for e in mjcf_only)
    assert mjcf_only[0].format_label == "MJCF"


def test_urdf_entries_are_labelled_urdf(entries):
    urdf = [e for e in entries if e.has_urdf]
    assert urdf and urdf[0].format_label == "URDF"
