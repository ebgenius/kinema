"""A local kinema_catalog: its ``catalog.json``, read and checked (#105).

kinema_catalog keeps its robots' descriptions in git and lists them in one index,
``catalog.json``, which its ``meshkit index`` builds. Pointed at a clone of it in
the preferences, as the **Catalog Folder**, Kinema lists that index's entries in
place of the bundled catalogue, with what the index says of each: its kind, its
tags and its licence.

The folder is valid when a ``catalog.json`` is at its top, as in a clone, or one
level down, as in the family release zips unpacked side by side, and lists
entries Kinema reads. Several are merged.

Every entry is checked field by field against what ``catalog.schema.json``
(schema version 1) asks, since Blender carries no JSON Schema validator. An entry
that fails is left out and its problem kept, to be shown, instead of the whole
folder failing. ``tests/unit/test_catalog_folder.py`` holds these checks to a copy
of the schema, so they can't drift apart unseen.

Nothing here uses the network, and nothing here imports ``bpy``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

#: The index's file name, at the top of a catalog or of each unpacked zip.
INDEX = "catalog.json"
#: The directory beside an index holding its descriptions' repositories, one
#: submodule each.
SOURCES = "src"
#: The index's schema version that this module reads.
SCHEMA_VERSION = 1

#: The fields an index may have at its top; ``schema_version`` and ``entries`` it must.
INDEX_FIELDS = ("$schema", "schema_version", "generated_from", "entries")
#: Every field an entry must have; ``licence_note`` is the one optional field.
REQUIRED = (
    "id", "name", "kind", "category", "family", "maker", "tags", "licence",
    "submodule", "entry", "format", "args", "mesh_base",
)
OPTIONAL = ("licence_note",)
KINDS = ("robot", "end_effector", "hand", "base", "scene")
CATEGORIES = ("arm", "mobile", "legged")
FORMATS = ("xacro", "urdf")

#: What a kind reads as in a list.
KIND_LABELS = {
    "robot": "Robot",
    "end_effector": "End Effector",
    "hand": "Hand",
    "base": "Base",
    "scene": "Scene",
}

#: What a problem says of an index or directory that couldn't be read, which may
#: be so only for the moment.
UNREAD = "can't be read"

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_ENTRY = re.compile(r"\.(xacro|urdf)$")
_NO_SPACE = re.compile(r"^[^ ]*$")
_NAME_NO_SPACE = re.compile(r"^[^ ]+$")


@dataclass(frozen=True)
class FolderEntry:
    """One entry of a catalog folder's index, as the index gives it."""

    id: str
    name: str
    kind: str
    category: str
    family: str
    maker: str
    tags: tuple[str, ...]
    licence: str
    submodule: str
    entry: str
    format: str
    args: tuple[tuple[str, str], ...]
    mesh_base: str | None
    #: The directory holding the index this entry came from, which ``entry`` and
    #: ``mesh_base`` are relative to.
    root: Path
    #: The Catalog Folder it was found in: ``root``, or the directory above it.
    folder: Path
    licence_note: str = ""

    @property
    def key(self) -> str:
        return self.id

    @property
    def label(self) -> str:
        """Its name, which carries its maker already: "ABB IRB 1200-5/0.9"."""
        return self.name

    @property
    def description(self) -> str:
        """Kind, category, tags and licence, as one line."""
        parts = [KIND_LABELS.get(self.kind, self.kind), self.category]
        parts += [tag.replace("_", " ") for tag in self.tags]
        parts.append(self.licence)
        return " · ".join(parts)

    @property
    def path(self) -> Path:
        """The description file to import."""
        return self.root / self.entry

    @property
    def relative_path(self) -> str:
        """The description file from the Catalog Folder, through the unpacked zip
        it is in, if any, with forward slashes."""
        return Path(self.root.relative_to(self.folder), self.entry).as_posix()


@dataclass(frozen=True)
class Folder:
    """A catalog folder as read: its indexes, their entries, and what was wrong."""

    path: Path
    indexes: tuple[Path, ...] = ()
    entries: tuple[FolderEntry, ...] = ()
    problems: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        """An index was found with entries Kinema reads, so there is a catalog to
        list. One in another schema version lists nothing, and says why."""
        return bool(self.entries)

    def get(self, key: str) -> FolderEntry | None:
        return next((entry for entry in self.entries if entry.id == key), None)


def entry_problems(record) -> list[str]:
    """What is wrong with one record of an index, against schema version 1; none
    when it is a valid entry."""
    if not isinstance(record, dict):
        return ["not an object"]
    problems = [f"no '{name}'" for name in REQUIRED if name not in record]
    problems += [
        f"unknown field '{name}'" for name in record if name not in REQUIRED + OPTIONAL
    ]

    def text(name, pattern=None, choices=None, empty=False) -> None:
        if name not in record:
            return
        value = record[name]
        if not isinstance(value, str):
            problems.append(f"'{name}' is not a string")
        elif choices is not None and value not in choices:
            problems.append(f"'{name}' is '{value}', not one of {', '.join(choices)}")
        elif pattern is not None and not pattern.search(value):
            problems.append(f"'{name}' is '{value}', which isn't allowed")
        elif not empty and not value:
            problems.append(f"'{name}' is empty")

    text("id", _NAME)
    text("name")
    text("kind", choices=KINDS)
    text("category", choices=CATEGORIES)
    text("family", _NAME)
    text("maker")
    text("licence")
    text("licence_note")
    text("submodule")
    text("entry", _ENTRY, empty=True)
    text("format", choices=FORMATS)
    if "tags" in record:
        tags = record["tags"]
        if not isinstance(tags, list) or not all(isinstance(t, str) and t for t in tags):
            problems.append("'tags' is not a list of names")
    if "args" in record:
        args = record["args"]
        if not isinstance(args, dict) or not all(
            _NAME_NO_SPACE.search(name) and isinstance(value, str) and _NO_SPACE.search(value)
            for name, value in args.items()
        ):
            problems.append("'args' is not names to values without spaces")
    if "mesh_base" in record and not (
        record["mesh_base"] is None or isinstance(record["mesh_base"], str)
    ):
        problems.append("'mesh_base' is neither a path nor null")
    return problems


def _entry(record: dict, root: Path, folder: Path) -> FolderEntry:
    return FolderEntry(
        id=record["id"],
        name=record["name"],
        kind=record["kind"],
        category=record["category"],
        family=record["family"],
        maker=record["maker"],
        tags=tuple(record["tags"]),
        licence=record["licence"],
        licence_note=record.get("licence_note", ""),
        submodule=record["submodule"],
        entry=record["entry"],
        format=record["format"],
        args=tuple(record["args"].items()),
        mesh_base=record["mesh_base"],
        root=root,
        folder=folder,
    )


def read_index(
    path: Path, folder: Path | None = None
) -> tuple[list[FolderEntry], list[str]]:
    """The valid entries of one ``catalog.json``, and the problems with the rest.

    An index that can't be read, or that has another schema version, gives no
    entries and one problem."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        return [], [f"{path}: {UNREAD} ({error})"]
    except ValueError as error:
        return [], [f"{path}: isn't JSON ({error})"]
    if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
        return [], [f"{path}: has no list of entries"]
    unknown = [name for name in data if name not in INDEX_FIELDS]
    if unknown:
        return [], [f"{path}: unknown field '{unknown[0]}'"]
    for name in ("$schema", "generated_from"):
        if name in data and not isinstance(data[name], str):
            return [], [f"{path}: '{name}' is not a string"]
    version = data.get("schema_version")
    # JSON's true is no version, though Python's True == 1.
    if isinstance(version, bool) or version != SCHEMA_VERSION:
        return [], [
            f"{path}: schema version {data.get('schema_version')!r}; "
            f"Kinema reads version {SCHEMA_VERSION}"
        ]
    entries, problems = [], []
    for position, record in enumerate(data["entries"]):
        wrong = entry_problems(record)
        if wrong:
            name = record.get("id") if isinstance(record, dict) else None
            problems.append(f"{path}: entry {name or position}: {'; '.join(wrong)}")
        else:
            entries.append(_entry(record, path.parent, folder or path.parent))
    return entries, problems


def find_indexes(folder: Path, problems: list | None = None) -> list[Path]:
    """The folder's ``catalog.json``: at its top, else one in each directory one
    level down, in name order. What couldn't be looked into goes in ``problems``."""
    problems = [] if problems is None else problems
    top = folder / INDEX
    if _is_file(top, problems):
        return [top]
    try:
        children = sorted(folder.iterdir())
    except OSError as error:
        problems.append(f"{folder}: {UNREAD} ({error})")
        return []
    # One directory that can't be looked into loses only its own index.
    return [child / INDEX for child in children if _is_file(child / INDEX, problems)]


def _is_file(path: Path, problems: list) -> bool:
    try:
        return path.is_file()
    except OSError as error:
        problems.append(f"{path.parent}: {UNREAD} ({error})")
        return False


def _stamp(indexes) -> tuple:
    stamps = []
    for index in indexes:
        try:
            info = index.stat()
        except OSError:
            continue
        stamps.append((str(index), info.st_mtime_ns, info.st_size))
    return tuple(stamps)


_cache: dict[str, tuple[tuple, Folder]] = {}


def load(folder: str | Path) -> Folder:
    """The catalog in ``folder``, read again only when one of its indexes changes.

    Entries keep the first index's where two share an id; the second is named in
    the problems."""
    path = Path(folder).absolute()
    if not str(folder).strip() or not path.is_dir():
        return Folder(path, problems=(f"{path}: not a folder",) if str(folder).strip() else ())
    probed: list[str] = []
    indexes = find_indexes(path, probed)
    # What couldn't be looked into is part of the stamp: a directory without an
    # index that can't be, or can be again, changes no index.
    stamp = (_stamp(indexes), tuple(probed))
    cached = _cache.get(str(path))
    if cached is not None and cached[0] == stamp:
        return cached[1]

    entries, problems, seen = [], list(probed), {}
    # An index that couldn't be read may be so only for the moment, and its time
    # and size don't change when it can be read again.
    unread = False
    for index in indexes:
        found, wrong = read_index(index, path)
        problems += wrong
        unread = unread or any(problem.startswith(f"{index}: {UNREAD}") for problem in wrong)
        for entry in found:
            if entry.id in seen:
                where = "earlier in it" if seen[entry.id] == index else f"in {seen[entry.id]}"
                problems.append(
                    f"{index}: entry {entry.id} is {where} too; the first is used"
                )
                continue
            seen[entry.id] = index
            entries.append(entry)
    if not indexes and not probed:
        problems.append(f"{path}: no {INDEX} here or one level down")
    entries.sort(key=lambda entry: (entry.maker.lower(), entry.name.lower()))
    result = Folder(path, tuple(indexes), tuple(entries), tuple(problems))
    # Read again on the next look; one whose entries are wrong is read once.
    if not unread:
        _cache[str(path)] = (stamp, result)
    return result


def containing(path: str | Path) -> tuple[Path, FolderEntry | None] | None:
    """The catalog a description file is in: the directory of the nearest index
    above it, and the entry naming the file there, or None for a file it doesn't
    name. None when no index is above it.

    Found from the file's place alone, so the solver's later reload of a rig
    finds what its import found, whatever the preferences say by then. A
    ``catalog.json`` that lists nothing Kinema reads isn't a catalog's, and the
    search goes on above it."""
    path = Path(path).resolve()
    for root in path.parents:
        index = root / INDEX
        if not _is_file(index, []):
            continue
        found = load(root)
        if not found.valid or found.indexes != (index,):
            continue
        name = _same(path.relative_to(root))
        for entry in found.entries:
            if _same(entry.entry) == name:
                return root, entry
        return root, None
    return None


def _same(path) -> str:
    """A relative path as compared: one spelling of its separators and, where
    the file system ignores case, of its case."""
    return os.path.normcase(os.path.normpath(path))
