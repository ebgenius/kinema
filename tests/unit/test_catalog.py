"""The offline catalogue: the shipped JSON, and what the picker makes of it."""

from __future__ import annotations

import json

import pytest

from ..conftest import load_addon_module

catalog = load_addon_module("catalog.index")


@pytest.fixture(scope="module")
def entries():
    return catalog.all_entries(refresh=True)


@pytest.fixture(scope="module")
def raw():
    return json.loads(catalog._ROBOTS_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def curation():
    return json.loads(catalog._CURATION_JSON.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# the data files
# --------------------------------------------------------------------------
def test_catalog_is_populated(entries):
    assert len(entries) > 100


def test_every_entry_can_be_fetched_by_hand(entries):
    """The whole point of the catalogue: a repository, a commit and a file.

    Without all three the user is told to clone something and left to guess
    which of its files to open.
    """
    for entry in entries:
        assert entry.repo_url.startswith("http"), entry.key
        assert entry.commit, entry.key
        assert entry.clone_dir, entry.key


def test_tag_pinned_repositories_keep_their_full_ref(entries):
    """Six repositories pin a tag rather than a SHA. Abbreviating `v0.7.7` the
    way a hash is abbreviated would emit a ref that does not resolve."""
    tagged = [e for e in entries if len(e.commit) != 40]
    assert tagged, "expected at least one tag-pinned repository"
    for entry in tagged:
        assert entry.short_commit == entry.commit, entry.key


def test_only_curated_out_entries_may_lack_a_file_path(entries):
    """One description (eve_r3) rewrites its URDF at import, so its path cannot
    be resolved offline. Any *other* entry losing its path is a generator bug,
    not something to shrug at."""
    unresolved = [e.key for e in entries if e.file_path is None]
    assert all(catalog.get(key).is_curated_out for key in unresolved), unresolved


def test_clone_dir_comes_from_the_url_not_the_cache_path(entries):
    """`git clone` names the directory after the URL. robot_descriptions caches
    Universal_Robots_ROS2_Description under `ur_description`; telling a user to
    look there would send them to a folder that does not exist."""
    ur5e = catalog.get("ur5e_description")
    assert ur5e.clone_dir == "Universal_Robots_ROS2_Description"
    assert ur5e.repo_url.endswith("Universal_Robots_ROS2_Description.git")


def test_curation_keys_all_exist(raw, curation):
    unknown = set(curation) - set(raw["robots"])
    assert not unknown, f"curation.json names robots that are not in robots.json: {unknown}"


def test_curation_statuses_are_known(curation):
    for key, marks in curation.items():
        assert marks.get("status") in catalog.CURATED_OUT, key


def test_prefer_targets_exist_and_are_usable(curation):
    """Pointing a duplicate at another entry that is itself crossed out would
    send the user in a circle."""
    for key, marks in curation.items():
        target = marks.get("prefer")
        if not target:
            continue
        entry = catalog.get(target)
        assert entry is not None, f"{key} prefers unknown robot {target}"
        assert not entry.is_curated_out, f"{key} prefers curated-out robot {target}"


# --------------------------------------------------------------------------
# display
# --------------------------------------------------------------------------
def test_entries_carry_display_metadata(entries):
    ur5e = catalog.get("ur5e_description")
    assert ur5e is not None
    assert "arm" in ur5e.tags
    assert "urdf" in ur5e.formats
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


def test_format_label_follows_the_file_not_the_metadata(entries):
    """ur5e_description is recorded as URDF but ships only a xacro. Labelling it
    URDF would promise a file the repository does not contain."""
    assert catalog.get("ur5e_description").format_label == "xacro"
    assert catalog.get("ur5e_mj_description").format_label == "MJCF"
    assert catalog.get("panda_description").format_label == "URDF"


def test_available_tags_are_ordered_sensibly(entries):
    tags = catalog.available_tags()
    assert "arm" in tags
    # Known tags come first, in KNOWN_TAGS order, before any unexpected ones.
    known = [t for t in tags if t in catalog.KNOWN_TAGS]
    assert known == [t for t in catalog.KNOWN_TAGS if t in tags]


def test_mjcf_robots_are_supported(entries):
    """60 of the catalog's 186 robots ship MJCF only -- roughly a third of the
    total, so treating them as unusable would be a large hole."""
    mjcf_only = [e for e in entries if e.mjcf_path and not (e.urdf_path or e.xacro_path)]
    assert len(mjcf_only) > 20
    assert all(e.is_supported for e in mjcf_only)


# --------------------------------------------------------------------------
# the clipboard handoff
# --------------------------------------------------------------------------
class TestCloneCommand:
    def test_starts_with_a_runnable_git_clone(self):
        command = catalog.get("panda_description").clone_command
        assert command.splitlines()[0] == (
            "git clone https://github.com/Gepetto/example-robot-data.git"
        )

    def test_the_hint_is_a_shell_comment(self):
        """Both lines are pasted together, so anything after the clone must be
        inert in a terminal."""
        for line in catalog.get("panda_description").clone_command.splitlines()[1:]:
            assert line.startswith("#")

    def test_names_the_file_to_open(self):
        command = catalog.get("panda_description").clone_command
        assert "robots/panda_description/urdf/panda.urdf" in command

    def test_carries_the_pinned_revision(self):
        entry = catalog.get("panda_description")
        assert f"git checkout {entry.short_commit}" in entry.clone_command

    def test_omits_the_file_hint_when_no_path_is_known(self):
        entry = catalog.get("eve_r3_description")
        assert entry.file_path is None
        assert "then open" not in entry.clone_command


class TestHandoffHint:
    """The status-bar line the picker reports after copying the command."""

    def test_names_the_file(self):
        entry = catalog.get("panda_description")
        assert entry.handoff_hint == (
            "then open example-robot-data/robots/panda_description/urdf/panda.urdf"
        )

    def test_says_so_when_the_path_is_unknown(self):
        """It used to interpolate None into the message, promising a file
        called 'None' inside the repository."""
        entry = catalog.get("eve_r3_description")
        assert entry.file_path is None
        assert "None" not in entry.handoff_hint
        assert entry.clone_dir in entry.handoff_hint

    def test_never_promises_a_missing_file_for_any_entry(self):
        for entry in catalog.all_entries():
            assert "None" not in entry.handoff_hint, entry.key


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------
class TestSearch:
    def test_free_text_matches_key_robot_or_maker(self):
        assert any(e.key == "ur5e_description" for e in catalog.search("ur5e"))
        assert catalog.search("ur5e") == catalog.search("UR5E"), "should be case-insensitive"

    def test_tag_filter(self):
        arms = catalog.search(tag="arm")
        assert arms
        assert all("arm" in e.tags for e in arms)

    def test_supported_entries_are_the_default(self):
        assert all(e.is_supported for e in catalog.search())

    def test_curated_out_entries_are_hidden_by_default(self):
        hidden = [e for e in catalog.all_entries() if e.is_curated_out]
        assert hidden, "curation.json should cross out at least one entry"
        shown = {e.key for e in catalog.search(supported_only=False)}
        assert not shown & {e.key for e in hidden}

    def test_curated_out_entries_are_revealed_on_request(self):
        hidden = {e.key for e in catalog.all_entries() if e.is_curated_out}
        shown = {
            e.key
            for e in catalog.search(supported_only=False, include_curated_out=True)
        }
        assert hidden <= shown

    def test_nonsense_query_returns_nothing(self):
        assert catalog.search("zzzz-no-such-robot") == []
