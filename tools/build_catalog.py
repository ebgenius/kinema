"""Generate the add-on's offline robot catalogue from ``robot_descriptions``.

The add-on used to import ``robot_descriptions`` at runtime and download a
robot on demand. That is not something a Blender extension may do: it needs
network access, it pulls in GitPython, and resolving *which file inside the
repository* to load required importing a description module with its cloning
function monkey-patched out. All of that now happens here, once, on a
developer's machine, and the add-on ships the result as static JSON.

The interesting half is the probe. A description module looks like::

    REPOSITORY_PATH: str = _clone_to_cache("unitree_ros", commit=...)
    PACKAGE_PATH: str = _path.join(REPOSITORY_PATH, "robots", "a1_description")
    URDF_PATH: str = _path.join(PACKAGE_PATH, "urdf", "a1.urdf")

so the paths are pure ``os.path.join`` calls over whatever ``clone_to_cache``
returned. Substituting a sentinel string for that return value and importing the
module therefore yields every path relative to the repository root, without
fetching anything. The module is then evicted from ``sys.modules``, because its
paths point at a sentinel and must not be mistaken for a real checkout.

This is not universal: a handful of descriptions do real work at import time --
``eve_r3_description`` parses and rewrites its URDF to fix invalid joint limits,
which cannot work against a path that does not exist. Those raise, and are
recorded with a null path and reported at the end so they can be curated by
hand.

Usage::

    uv run python tools/build_catalog.py           # rewrite robots.json
    uv run python tools/build_catalog.py --check    # CI: fail if out of date
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_DIR = REPO_ROOT / "src" / "kinema" / "catalog"
ROBOTS_JSON = CATALOG_DIR / "robots.json"

#: Stand-in for the repository root during a probe import. Never touches disk.
SENTINEL_ROOT = "__KINEMA_REPO_ROOT__"

#: Description module attributes worth recording, and the JSON key for each.
#:
#: ``XACRO_PATH`` matters as much as ``URDF_PATH``: 38 descriptions ship only a
#: xacro, which Kinema reads directly. Without it those entries would have no
#: file to point the user at. ``SRDF_XACRO_PATH`` is deliberately absent -- an
#: SRDF describes planning groups, not kinematics.
PATH_ATTRS = {
    "URDF_PATH": "urdf_path",
    "XACRO_PATH": "xacro_path",
    "MJCF_PATH": "mjcf_path",
}


def clone_dir(repo_url: str) -> str:
    """The directory ``git clone <repo_url>`` creates.

    Not the same as the repository's ``cache_path``: ``robot_descriptions``
    caches Universal_Robots_ROS2_Description under ``ur_description``, but a
    user cloning it by hand gets a directory named after the URL. The catalogue
    tells the user where their own files will land, so the URL is what counts.
    """
    return repo_url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")


def _relative_to_root(path: str) -> str | None:
    """A probed absolute path, relative to the sentinel root, as POSIX.

    Returns None when the path escaped the repository or was never built from
    the sentinel at all -- both mean "this probe told us nothing".
    """
    if not path:
        return None
    try:
        relative = os.path.relpath(path, SENTINEL_ROOT)
    except ValueError:
        return None
    if relative in (".", "", os.curdir) or relative.startswith(os.pardir):
        return None
    return PurePosixPath(*Path(relative).parts).as_posix()


def probe(key: str) -> tuple[dict[str, str], str | None]:
    """Import one description under the sentinel and read its paths back.

    Returns the paths found and, on failure, the exception text. Never raises:
    a description that cannot be probed still belongs in the catalogue, just
    without a file path to point the user at.
    """
    from robot_descriptions import _cache

    module_name = f"robot_descriptions.{key}"
    original = _cache.clone_to_cache
    _cache.clone_to_cache = lambda name, commit=None: SENTINEL_ROOT
    try:
        module = importlib.import_module(module_name)
    except BaseException as exc:  # noqa: BLE001 - a bad probe must not stop the run
        return {}, f"{type(exc).__name__}: {exc}"
    else:
        found = {}
        for attr, json_key in PATH_ATTRS.items():
            relative = _relative_to_root(getattr(module, attr, ""))
            if relative:
                found[json_key] = relative
        return found, None
    finally:
        _cache.clone_to_cache = original
        # The probe leaves a module whose paths point at the sentinel. It must
        # not survive: anything importing it later would believe them.
        sys.modules.pop(module_name, None)


def build() -> dict:
    """The whole catalogue, ready to serialise."""
    from robot_descriptions._descriptions import DESCRIPTIONS
    from robot_descriptions._repositories import REPOSITORIES

    robots: dict[str, dict] = {}
    failures: list[tuple[str, str]] = []

    for key in sorted(DESCRIPTIONS):
        description = DESCRIPTIONS[key]
        repository = REPOSITORIES.get(description.repository)
        if repository is None:
            failures.append((key, f"unknown repository {description.repository!r}"))
            continue

        paths, error = probe(key)
        if error is not None:
            failures.append((key, error))

        formats = []
        if getattr(description, "has_urdf", False):
            formats.append("urdf")
        if getattr(description, "has_mjcf", False):
            formats.append("mjcf")

        robots[key] = {
            "robot": description.robot,
            "maker": description.maker or "",
            "dof": int(description.dof or 0),
            "tags": sorted(description.tags or ()),
            "formats": formats,
            "license_spdx": description.license_spdx,
            "repo_url": repository.url,
            "clone_dir": clone_dir(repository.url),
            "commit": repository.commit,
            **{json_key: paths.get(json_key) for json_key in PATH_ATTRS.values()},
        }

    if failures:
        print(f"\n{len(failures)} description(s) could not be probed:", file=sys.stderr)
        for key, reason in failures:
            print(f"  {key}: {reason}", file=sys.stderr)
        print(
            "\nThese are listed without a file path. Curate them by hand in "
            "curation.json if they are not usable.",
            file=sys.stderr,
        )

    return {
        "robot_descriptions_version": importlib.metadata.version("robot_descriptions"),
        "robots": robots,
    }


def serialise(catalog: dict) -> str:
    return json.dumps(catalog, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit non-zero if robots.json is out of date",
    )
    args = parser.parse_args()

    # GitPython refuses to import without a git binary on PATH, and importing
    # any description module reaches it. Nothing here clones, so quiet it.
    os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

    text = serialise(build())

    if args.check:
        current = ROBOTS_JSON.read_text(encoding="utf-8") if ROBOTS_JSON.exists() else ""
        if current != text:
            print(
                f"{ROBOTS_JSON.relative_to(REPO_ROOT)} is out of date; "
                "run: uv run python tools/build_catalog.py",
                file=sys.stderr,
            )
            return 1
        print(f"{ROBOTS_JSON.relative_to(REPO_ROOT)} is up to date")
        return 0

    ROBOTS_JSON.write_text(text, encoding="utf-8")
    count = len(json.loads(text)["robots"])
    print(f"wrote {ROBOTS_JSON.relative_to(REPO_ROOT)}: {count} robots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
