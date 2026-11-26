"""A Catalog Folder's ``catalog.json``: the field checks, held to kinema_catalog's
schema, and how a folder's indexes are found and merged (#105)."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import jsonschema
import pytest

from ..conftest import FIXTURE_DIR, load_addon_module

folder = load_addon_module("catalog.folder")

FIXTURES = FIXTURE_DIR / "catalog_folder"
SCHEMA = json.loads((FIXTURES / "catalog.schema.json").read_text(encoding="utf-8"))
ENTRY_SCHEMA = SCHEMA["$defs"]["entry"]

#: A valid entry, to break one field at a time.
UR5E = json.loads((FIXTURES / "clone" / "catalog.json").read_text(encoding="utf-8"))[
    "entries"
][3]


def _index(entries, version=1) -> str:
    return json.dumps({"schema_version": version, "entries": entries})


# --------------------------------------------------------------------------
# the checks against the schema
# --------------------------------------------------------------------------
class TestTheChecksMatchTheSchema:
    """The fields and choices Kinema checks are the schema's, so a change to the
    schema shows here when its copy is refreshed."""

    def test_the_fixture_entry_is_ur5e(self):
        assert UR5E["id"] == "ur5e"

    def test_required_fields(self):
        assert set(folder.REQUIRED) == set(ENTRY_SCHEMA["required"])

    def test_every_field(self):
        assert set(folder.REQUIRED + folder.OPTIONAL) == set(ENTRY_SCHEMA["properties"])

    @pytest.mark.parametrize(
        ("field", "choices"),
        [("kind", "KINDS"), ("category", "CATEGORIES"), ("format", "FORMATS")],
    )
    def test_choices(self, field, choices):
        assert set(getattr(folder, choices)) == set(ENTRY_SCHEMA["properties"][field]["enum"])

    def test_schema_version(self):
        assert SCHEMA["properties"]["schema_version"]["const"] == folder.SCHEMA_VERSION

    def test_every_kind_has_a_label(self):
        assert set(folder.KIND_LABELS) == set(folder.KINDS)


def _without(name):
    record = copy.deepcopy(UR5E)
    del record[name]
    return record


def _with(**fields):
    record = copy.deepcopy(UR5E)
    record.update(fields)
    return record


#: Records, each valid or invalid by the schema, to hold the checks to.
RECORDS = {
    "as it is": UR5E,
    "with a licence note": _with(licence_note="the package.xml says otherwise"),
    "meshes by relative path": _with(mesh_base="src/ur/meshes"),
    "with tags": _with(tags=["gripper", "dual_arm"]),
    "an empty entry name is a suffix only": _with(entry=".urdf"),
    "no args": _with(args={}),
    **{f"no {name}": _without(name) for name in ENTRY_SCHEMA["required"]},
    "an unknown field": _with(colour="blue"),
    "an id with capitals": _with(id="UR5e"),
    "an id starting with a digit": _with(id="5ur"),
    "an id that isn't a string": _with(id=5),
    "a family with a dash": _with(family="universal-robots"),
    "an unknown kind": _with(kind="gripper"),
    "an unknown category": _with(category="humanoid"),
    "an unknown format": _with(format="mjcf"),
    "an entry that isn't URDF or xacro": _with(entry="model.sdf"),
    "an empty name": _with(name=""),
    "an empty maker": _with(maker=""),
    "an empty licence": _with(licence=""),
    "an empty licence note": _with(licence_note=""),
    "an empty submodule": _with(submodule=""),
    "tags that aren't a list": _with(tags="gripper"),
    "an empty tag": _with(tags=[""]),
    "args that aren't an object": _with(args=["ur_type:=ur5e"]),
    "an arg with a space in its value": _with(args={"ur_type": "ur 5e"}),
    "an arg with a space in its name": _with(args={"ur type": "ur5e"}),
    "an arg that isn't a string": _with(args={"safety_limits": True}),
    "a mesh base that's a number": _with(mesh_base=3),
    "not an object": ["ur5e"],
}


@pytest.mark.parametrize("name", list(RECORDS))
def test_kinemas_checks_agree_with_the_schema(name):
    """Valid by the schema exactly when Kinema finds nothing wrong."""
    record = RECORDS[name]
    validator = jsonschema.Draft202012Validator(
        {**ENTRY_SCHEMA, "$defs": SCHEMA["$defs"]}
    )
    schema_says_valid = validator.is_valid(record)
    assert (not folder.entry_problems(record)) == schema_says_valid, (
        name, folder.entry_problems(record)
    )


def test_both_kinds_of_record_are_covered():
    """The agreement above means something only if the schema both accepts and
    refuses some of the records."""
    validator = jsonschema.Draft202012Validator({**ENTRY_SCHEMA, "$defs": SCHEMA["$defs"]})
    verdicts = {validator.is_valid(record) for record in RECORDS.values()}
    assert verdicts == {True, False}


@pytest.mark.parametrize("path", sorted(FIXTURES.rglob("catalog.json")), ids=str)
def test_the_fixture_indexes_are_valid(path):
    jsonschema.validate(json.loads(path.read_text(encoding="utf-8")), SCHEMA)


#: Whole indexes, around one valid entry, to hold the index's own checks to.
INDEXES = {
    "as it is": {"schema_version": 1, "entries": [UR5E]},
    "with its schema and source": {
        "$schema": "./catalog.schema.json", "schema_version": 1,
        "generated_from": "catalog/*.yaml", "entries": [UR5E],
    },
    "another version": {"schema_version": 2, "entries": [UR5E]},
    "a version that's true": {"schema_version": True, "entries": [UR5E]},
    "a version that's text": {"schema_version": "1", "entries": [UR5E]},
    "no version": {"entries": [UR5E]},
    "no entries": {"schema_version": 1},
    "entries that aren't a list": {"schema_version": 1, "entries": {"ur5e": UR5E}},
    "an unknown field": {"schema_version": 1, "entries": [UR5E], "robots": []},
    "a schema that isn't text": {"$schema": 1, "schema_version": 1, "entries": [UR5E]},
    "a source that isn't text": {"generated_from": [], "schema_version": 1, "entries": [UR5E]},
}


@pytest.mark.parametrize("name", list(INDEXES))
def test_an_index_is_read_exactly_when_the_schema_accepts_it(name, tmp_path):
    index = tmp_path / "catalog.json"
    index.write_text(json.dumps(INDEXES[name]), encoding="utf-8")
    entries, problems = folder.read_index(index)
    assert bool(entries) == jsonschema.Draft202012Validator(SCHEMA).is_valid(INDEXES[name])
    assert bool(problems) != bool(entries), problems


def test_a_problem_names_the_field():
    assert folder.entry_problems(_with(kind="gripper")) == [
        "'kind' is 'gripper', not one of robot, end_effector, hand, base, scene"
    ]
    assert folder.entry_problems(_without("licence")) == ["no 'licence'"]


# --------------------------------------------------------------------------
# entries
# --------------------------------------------------------------------------
class TestAnEntry:
    def test_is_read_as_the_index_gives_it(self):
        entries, problems = folder.read_index(FIXTURES / "clone" / "catalog.json")
        assert not problems
        ur5e = next(entry for entry in entries if entry.id == "ur5e")
        assert ur5e.name == "Universal Robots UR5e"
        assert ur5e.format == "xacro"
        assert dict(ur5e.args) == {"ur_type": "ur5e", "name": "ur5e"}
        assert ur5e.mesh_base is None
        assert ur5e.root == FIXTURES / "clone"

    def test_its_file_is_under_its_index(self):
        found = folder.load(FIXTURES / "clone")
        so100 = found.get("so100")
        assert so100.path == FIXTURES / "clone" / "src/SO-ARM100/Simulation/SO100/so100.urdf"
        assert so100.mesh_base == "src/SO-ARM100/Simulation/SO100"

    def test_its_description_has_kind_category_tags_and_licence(self):
        gripper = folder.load(FIXTURES / "clone").get("gripper_2f")
        assert gripper.description == "End Effector · arm · gripper · BSD-3-Clause"

    def test_its_file_from_a_clone_is_as_the_index_names_it(self):
        ur5e = folder.load(FIXTURES / "clone").get("ur5e")
        assert ur5e.relative_path == "src/Universal_Robots_ROS2_Description/urdf/ur.urdf.xacro"

    def test_its_file_from_an_unpacked_zip_goes_through_the_zip(self):
        ur5e = folder.load(FIXTURES / "zips").get("ur5e")
        assert ur5e.relative_path == (
            "universal_robots/src/Universal_Robots_ROS2_Description/urdf/ur.urdf.xacro"
        )
        assert FIXTURES / "zips" / ur5e.relative_path == ur5e.path

    def test_a_licence_note_is_kept(self):
        aura = folder.load(FIXTURES / "clone").get("aura")
        assert "BSD-2-Clause" in aura.licence_note


# --------------------------------------------------------------------------
# a folder
# --------------------------------------------------------------------------
class TestAFolder:
    def test_a_clone_has_its_index_at_the_top(self):
        found = folder.load(FIXTURES / "clone")
        assert found.valid
        assert found.indexes == (FIXTURES / "clone" / "catalog.json",)
        assert not found.problems
        assert len(found.entries) == 5

    def test_unpacked_zips_have_theirs_one_level_down_and_are_merged(self):
        found = folder.load(FIXTURES / "zips")
        assert found.valid
        assert [index.parent.name for index in found.indexes] == ["abb", "universal_robots"]
        assert {entry.id for entry in found.entries} == {"irb1200", "ur5e", "gripper_2f"}
        assert found.get("irb1200").root == FIXTURES / "zips" / "abb"
        assert found.get("ur5e").root == FIXTURES / "zips" / "universal_robots"

    def test_entries_are_sorted_by_maker_then_name(self):
        names = [entry.name for entry in folder.load(FIXTURES / "clone").entries]
        assert names == [
            "ABB IRB 1200-5/0.9",
            "Comau AURA",
            "Two-Finger Gripper",
            "Standard Open Arm SO-100",
            "Universal Robots UR5e",
        ]

    def test_the_top_index_wins_over_those_below(self, tmp_path):
        (tmp_path / "catalog.json").write_text(_index([UR5E]), encoding="utf-8")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "catalog.json").write_text(
            _index([_with(id="other")]), encoding="utf-8"
        )
        found = folder.load(tmp_path)
        assert [entry.id for entry in found.entries] == ["ur5e"]

    def test_two_levels_down_is_not_looked_at(self, tmp_path):
        deep = tmp_path / "a" / "b"
        deep.mkdir(parents=True)
        (deep / "catalog.json").write_text(_index([UR5E]), encoding="utf-8")
        found = folder.load(tmp_path)
        assert not found.valid
        assert found.problems == (f"{tmp_path}: no catalog.json here or one level down",)

    def test_an_id_in_two_indexes_keeps_the_first(self, tmp_path):
        for name, maker in (("a", "First"), ("b", "Second")):
            (tmp_path / name).mkdir()
            (tmp_path / name / "catalog.json").write_text(
                _index([_with(maker=maker)]), encoding="utf-8"
            )
        found = folder.load(tmp_path)
        assert [entry.maker for entry in found.entries] == ["First"]
        assert found.problems == (
            f"{tmp_path / 'b' / 'catalog.json'}: entry ur5e is in "
            f"{tmp_path / 'a' / 'catalog.json'} too; the first is used",
        )

    def test_an_id_twice_in_one_index_keeps_the_first(self, tmp_path):
        index = tmp_path / "catalog.json"
        index.write_text(_index([UR5E, _with(maker="Second")]), encoding="utf-8")
        found = folder.load(tmp_path)
        assert [entry.maker for entry in found.entries] == ["Universal Robots"]
        assert found.problems == (
            f"{index}: entry ur5e is earlier in it too; the first is used",
        )

    def test_a_directory_that_cant_be_looked_into_loses_only_its_own(
        self, tmp_path, monkeypatch
    ):
        for name in ("a", "b"):
            (tmp_path / name).mkdir()
            (tmp_path / name / "catalog.json").write_text(
                _index([_with(id=f"robot_{name}")]), encoding="utf-8"
            )
        locked = tmp_path / "a" / "catalog.json"
        is_file = Path.is_file

        def refusing(path):
            if path == locked:
                raise PermissionError(13, "Permission denied", str(path))
            return is_file(path)

        monkeypatch.setattr(Path, "is_file", refusing)
        found = folder.load(tmp_path)
        assert [entry.id for entry in found.entries] == ["robot_b"]
        assert len(found.problems) == 1
        assert found.problems[0].startswith(f"{tmp_path / 'a'}: can't be read")
        # Not kept: once it can be looked into, its index is found.
        monkeypatch.setattr(Path, "is_file", is_file)
        assert [entry.id for entry in folder.load(tmp_path).entries] == ["robot_a", "robot_b"]

    def test_an_invalid_entry_is_left_out_and_the_rest_kept(self, tmp_path):
        (tmp_path / "catalog.json").write_text(
            _index([UR5E, _with(id="bad", kind="gripper")]), encoding="utf-8"
        )
        found = folder.load(tmp_path)
        assert found.valid
        assert [entry.id for entry in found.entries] == ["ur5e"]
        assert len(found.problems) == 1 and "entry bad: 'kind'" in found.problems[0]

    def test_another_schema_version_lists_nothing_and_says_why(self, tmp_path):
        (tmp_path / "catalog.json").write_text(_index([UR5E], version=2), encoding="utf-8")
        found = folder.load(tmp_path)
        assert not found.valid
        assert "schema version 2; Kinema reads version 1" in found.problems[0]

    def test_an_unreadable_index_says_so(self, tmp_path):
        (tmp_path / "catalog.json").write_text("{not json", encoding="utf-8")
        found = folder.load(tmp_path)
        assert not found.valid
        assert "isn't JSON" in found.problems[0]

    def test_a_missing_folder_says_so(self, tmp_path):
        found = folder.load(tmp_path / "nowhere")
        assert not found.valid
        assert found.problems == (f"{tmp_path / 'nowhere'}: not a folder",)

    def test_an_unset_folder_has_nothing_to_say(self):
        found = folder.load("")
        assert not found.valid and not found.problems

    def test_an_index_is_read_again_when_it_changes(self, tmp_path):
        index = tmp_path / "catalog.json"
        index.write_text(_index([UR5E]), encoding="utf-8")
        assert len(folder.load(tmp_path).entries) == 1
        index.write_text(_index([UR5E, _with(id="ur10e")]), encoding="utf-8")
        # A write within the clock's resolution keeps its time: the size tells.
        stat = index.stat()
        os.utime(index, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        assert len(folder.load(tmp_path).entries) == 2

    def test_an_index_of_the_same_size_is_read_again_when_its_time_changes(self, tmp_path):
        index = tmp_path / "catalog.json"
        index.write_text(_index([_with(maker="Before")]), encoding="utf-8")
        assert folder.load(tmp_path).entries[0].maker == "Before"
        stat = index.stat()
        index.write_text(_index([_with(maker="Aft3r!")]), encoding="utf-8")
        assert index.stat().st_size == stat.st_size
        os.utime(index, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        assert folder.load(tmp_path).entries[0].maker == "Aft3r!"

    def test_a_directory_without_an_index_that_cant_be_looked_into_is_named(
        self, tmp_path, monkeypatch
    ):
        """And no longer, once it can be again: neither changes an index."""
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "catalog.json").write_text(_index([UR5E]), encoding="utf-8")
        (tmp_path / "b").mkdir()
        assert not folder.load(tmp_path).problems
        locked = tmp_path / "b" / "catalog.json"
        is_file = Path.is_file

        def refusing(path):
            if path == locked:
                raise PermissionError(13, "Permission denied", str(path))
            return is_file(path)

        monkeypatch.setattr(Path, "is_file", refusing)
        found = folder.load(tmp_path)
        assert found.valid
        assert len(found.problems) == 1
        assert found.problems[0].startswith(f"{tmp_path / 'b'}: can't be read")
        monkeypatch.setattr(Path, "is_file", is_file)
        assert not folder.load(tmp_path).problems

    def test_a_folder_that_cant_be_looked_into_says_only_that(self, tmp_path, monkeypatch):
        """Not that it has no catalog.json, which nobody could see."""
        iterdir = Path.iterdir

        def refusing(path):
            if path == tmp_path:
                raise PermissionError(13, "Permission denied", str(path))
            return iterdir(path)

        monkeypatch.setattr(Path, "iterdir", refusing)
        found = folder.load(tmp_path)
        assert not found.valid
        assert len(found.problems) == 1
        assert found.problems[0].startswith(f"{tmp_path}: can't be read")

    def test_an_index_that_couldnt_be_read_is_tried_again(self, tmp_path, monkeypatch):
        """Its time and size don't change when it can be read again."""
        index = tmp_path / "catalog.json"
        index.write_text(_index([UR5E]), encoding="utf-8")
        read_text = Path.read_text

        def refusing(path, *args, **kwargs):
            if path == index:
                raise PermissionError(13, "Permission denied", str(path))
            return read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", refusing)
        found = folder.load(tmp_path)
        assert not found.valid
        assert found.problems[0].startswith(f"{index}: can't be read")
        monkeypatch.setattr(Path, "read_text", read_text)
        assert folder.load(tmp_path).valid

    def test_an_index_whose_entries_are_wrong_is_read_once(self, tmp_path, monkeypatch):
        (tmp_path / "catalog.json").write_text(
            _index([_with(kind="gripper")]), encoding="utf-8"
        )
        reads = []
        read_index = folder.read_index
        monkeypatch.setattr(
            folder, "read_index", lambda *args: reads.append(args) or read_index(*args)
        )
        assert not folder.load(tmp_path).valid
        assert not folder.load(tmp_path).valid
        assert len(reads) == 1

    def test_an_index_that_isnt_json_says_so(self, tmp_path):
        (tmp_path / "catalog.json").write_text("{not json", encoding="utf-8")
        assert "isn't JSON" in folder.load(tmp_path).problems[0]

    def test_a_relative_folder_is_its_own_in_each_directory(self, tmp_path, monkeypatch):
        # The same size and time, so only the folder tells them apart.
        for name, maker in (("one", "First"), ("two", "Other")):
            (tmp_path / name / "catalog").mkdir(parents=True)
            index = tmp_path / name / "catalog" / "catalog.json"
            index.write_text(_index([_with(maker=maker)]), encoding="utf-8")
            os.utime(index, ns=(1_700_000_000_000_000_000,) * 2)
        monkeypatch.chdir(tmp_path / "one")
        assert folder.load("catalog").entries[0].maker == "First"
        monkeypatch.chdir(tmp_path / "two")
        assert folder.load("catalog").entries[0].maker == "Other"
