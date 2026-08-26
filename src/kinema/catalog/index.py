"""Browse the robot_descriptions catalog.

``robot_descriptions`` ships a metadata table for 186 robots -- maker, degrees
of freedom, licence, tags, and which formats each provides. That is enough to
present a real picker instead of an alphabetical wall of module names, so this
module exposes it in a form Blender's UI can consume directly.

Nothing here downloads anything. Importing a *description module* triggers a
download at import time, but the metadata table is a plain dict that is safe to
read offline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

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


@dataclass(frozen=True)
class CatalogEntry:
    """One robot in the catalog, with everything needed to show and load it."""

    key: str
    robot: str
    maker: str
    dof: int
    tags: tuple[str, ...]
    has_urdf: bool
    has_mjcf: bool
    license_spdx: str | None

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
    def is_supported(self) -> bool:
        """Kinema builds rigs from URDF; MJCF-only entries need the MJCF reader."""
        return self.has_urdf


def _load_table() -> dict:
    # GitPython refuses to import without a git binary; the catalog metadata
    # itself needs no git at all, so quiet that check before touching it.
    os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
    from robot_descriptions._descriptions import DESCRIPTIONS

    return DESCRIPTIONS


_cache: list[CatalogEntry] | None = None


def all_entries(refresh: bool = False) -> list[CatalogEntry]:
    """Every catalog entry, sorted by maker then robot name."""
    global _cache
    if _cache is not None and not refresh:
        return _cache

    try:
        table = _load_table()
    except Exception:  # noqa: BLE001 - catalog is optional; UI degrades gracefully
        _cache = []
        return _cache

    entries = [
        CatalogEntry(
            key=key,
            robot=str(getattr(value, "robot", key)),
            maker=str(getattr(value, "maker", "") or ""),
            dof=int(getattr(value, "dof", 0) or 0),
            tags=tuple(sorted(getattr(value, "tags", ()) or ())),
            has_urdf=bool(getattr(value, "has_urdf", False)),
            has_mjcf=bool(getattr(value, "has_mjcf", False)),
            license_spdx=getattr(value, "license_spdx", None),
        )
        for key, value in table.items()
    ]
    entries.sort(key=lambda e: (e.maker.lower(), e.robot.lower()))
    _cache = entries
    return _cache


def search(
    text: str = "",
    *,
    tag: str = "",
    urdf_only: bool = True,
) -> list[CatalogEntry]:
    """Filter the catalog by free text and tag."""
    needle = text.strip().lower()
    results = []
    for entry in all_entries():
        if urdf_only and not entry.is_supported:
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


def load_urdf(key: str, progress=None):
    """Download (if needed) and parse one description into a ``yourdfpy.URDF``.

    Raises RuntimeError if the description is not available offline and Blender
    is in offline mode. The extension declares the ``network`` permission, and
    the guidelines require honouring ``bpy.app.online_access`` before using it.
    """
    from .fetch import install_git_free_loader

    install_git_free_loader(progress=progress)

    from robot_descriptions.loaders.yourdfpy import load_robot_description

    return load_robot_description(key)
