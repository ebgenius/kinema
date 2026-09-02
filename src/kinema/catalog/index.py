"""The offline robot catalogue.

186 robots with maker, degrees of freedom, licence, tags, formats -- and, for
185 of them, the exact path of the description file *inside* its repository.
That is enough to present a real picker instead of an alphabetical wall of
module names, and enough to tell a user precisely what to clone and which file
to open afterwards.

Nothing here reaches the network, and nothing here imports
``robot_descriptions``. The data is generated from it by
``tools/build_catalog.py`` and shipped as JSON; see that script for how the
paths are resolved.

Two files back this module:

``robots.json``
    Generated. Never edit by hand -- ``build_catalog.py --check`` will notice.

``curation.json``
    Hand-maintained, and *subtractive*: a robot absent from it is shown
    normally, and reviewing the catalogue means crossing entries out. Each
    value is ``{"status": ..., "note": ..., "prefer": ...}`` where status is one
    of :data:`CURATED_OUT` and ``prefer`` names the entry to use instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent
_ROBOTS_JSON = _DATA_DIR / "robots.json"
_CURATION_JSON = _DATA_DIR / "curation.json"

#: Tags the catalog uses, in the order worth showing them.
KNOWN_TAGS = (
    "arm",
    "dual_arm",
    "end_effector",
    "humanoid",
    "quadruped",
    "biped",
    "mobile_manipulator",
    "wheeled",
    "drone",
)

#: Curation statuses that hide an entry from the default listing.
CURATED_OUT = ("duplicate", "broken", "partial")


@dataclass(frozen=True)
class CatalogEntry:
    """One robot in the catalog, with everything needed to show and fetch it."""

    key: str
    robot: str
    maker: str
    dof: int
    tags: tuple[str, ...]
    formats: tuple[str, ...]
    license_spdx: str | None
    repo_url: str
    clone_dir: str
    commit: str
    urdf_path: str | None
    xacro_path: str | None
    mjcf_path: str | None
    status: str = ""
    note: str = ""
    prefer: str = ""

    @property
    def label(self) -> str:
        return f"{self.maker} {self.robot}".strip()

    @property
    def description(self) -> str:
        parts = [f"{self.dof} DoF"] if self.dof else []
        parts += [t.replace("_", " ") for t in self.tags]
        if self.license_spdx:
            parts.append(self.license_spdx)
        return " · ".join(parts)

    @property
    def file_path(self) -> str | None:
        """The description file to open after cloning, relative to the repo.

        URDF first, then xacro, then MJCF -- the order Kinema handles them best.
        None for the one description whose path could not be resolved.
        """
        return self.urdf_path or self.xacro_path or self.mjcf_path

    @property
    def format_label(self) -> str:
        path = self.file_path or ""
        if path.endswith(".xacro"):
            return "xacro"
        return "MJCF" if path.endswith(".xml") else "URDF"

    @property
    def is_supported(self) -> bool:
        """Kinema reads all three formats, so any resolved path can be rigged."""
        return self.file_path is not None

    @property
    def is_curated_out(self) -> bool:
        return self.status in CURATED_OUT

    @property
    def short_commit(self) -> str:
        """The pinned revision, abbreviated only if it is a hash.

        Six of the 83 repositories are pinned to a tag rather than a SHA
        (``v0.7.7``, ``release-1.0.0``); truncating those would produce a ref
        that does not resolve.
        """
        is_sha = len(self.commit) == 40 and all(c in "0123456789abcdef" for c in self.commit)
        return self.commit[:12] if is_sha else self.commit

    @property
    def clone_command(self) -> str:
        """What goes on the clipboard.

        Two lines, but paste-safe: the second is a shell comment, so pasting
        the block into a terminal clones the repository and does nothing else.
        It carries the two facts the user cannot get from the clone -- the
        revision the description was written against, and which file to open
        out of up to 2466 of them.
        """
        command = f"git clone {self.repo_url}"
        if self.commit:
            command += f"\n# cd {self.clone_dir} && git checkout {self.short_commit}"
        if self.file_path:
            command += f"\n# then open: {self.file_path}"
        return command

    @property
    def handoff_hint(self) -> str:
        """One line naming what to do after cloning, for the status bar.

        Kept beside :attr:`clone_command` because it has the same conditional:
        one description's path could not be resolved offline, and naming a file
        we do not know is worse than admitting we do not know it.
        """
        if self.file_path:
            return f"then open {self.clone_dir}/{self.file_path}"
        return f"file unknown; look inside {self.clone_dir} after cloning"


def _entry(key: str, record: dict, curation: dict) -> CatalogEntry:
    marks = curation.get(key) or {}
    return CatalogEntry(
        key=key,
        robot=str(record.get("robot") or key),
        maker=str(record.get("maker") or ""),
        dof=int(record.get("dof") or 0),
        tags=tuple(record.get("tags") or ()),
        formats=tuple(record.get("formats") or ()),
        license_spdx=record.get("license_spdx"),
        repo_url=str(record.get("repo_url") or ""),
        clone_dir=str(record.get("clone_dir") or ""),
        commit=str(record.get("commit") or ""),
        urdf_path=record.get("urdf_path"),
        xacro_path=record.get("xacro_path"),
        mjcf_path=record.get("mjcf_path"),
        status=str(marks.get("status") or ""),
        note=str(marks.get("note") or ""),
        prefer=str(marks.get("prefer") or ""),
    )


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # A corrupt or missing data file degrades the picker to empty rather
        # than raising out of a UI callback.
        return {}


@lru_cache(maxsize=1)
def _entries() -> tuple[CatalogEntry, ...]:
    robots = _read(_ROBOTS_JSON).get("robots") or {}
    curation = _read(_CURATION_JSON)
    entries = [_entry(key, record, curation) for key, record in robots.items()]
    entries.sort(key=lambda e: (e.maker.lower(), e.robot.lower()))
    return tuple(entries)


def all_entries(refresh: bool = False) -> list[CatalogEntry]:
    """Every catalog entry, sorted by maker then robot name."""
    if refresh:
        _entries.cache_clear()
    return list(_entries())


def search(
    text: str = "",
    *,
    tag: str = "",
    supported_only: bool = True,
    include_curated_out: bool = False,
) -> list[CatalogEntry]:
    """Filter the catalog by free text and tag.

    Curated-out entries are hidden by default: of the three UR5e entries only
    the one worth using shows up until the caller asks for the rest.
    """
    needle = text.strip().lower()
    results = []
    for entry in all_entries():
        if supported_only and not entry.is_supported:
            continue
        if entry.is_curated_out and not include_curated_out:
            continue
        if tag and tag not in entry.tags:
            continue
        if needle and needle not in f"{entry.key} {entry.robot} {entry.maker}".lower():
            continue
        results.append(entry)
    return results


def get(key: str) -> CatalogEntry | None:
    return next((e for e in all_entries() if e.key == key), None)


def available_tags() -> list[str]:
    """Tags actually present in the catalog, in a stable, sensible order."""
    present = {tag for entry in all_entries() for tag in entry.tags}
    ordered = [tag for tag in KNOWN_TAGS if tag in present]
    return ordered + sorted(present - set(ordered))
