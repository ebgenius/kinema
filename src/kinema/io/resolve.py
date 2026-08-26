"""Resolve URDF mesh references to real files on disk.

URDF mesh filenames are rarely plain paths. ROS descriptions overwhelmingly use
``package://<pkg>/<rest>``, which only means something to a ROS workspace
lookup. Kinema has no ROS installation to ask, so it locates the package
directory by searching the description's own directory tree -- which is exactly
where a downloaded or checked-out description keeps it.

Handled forms:

* ``package://pkg/path``  -- ROS package-relative
* ``model://name/path``   -- Gazebo model database
* ``file:///abs/path``    -- explicit file URI
* absolute and relative paths
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import unquote, urlparse

#: A directory containing one of these is a ROS package root.
_PACKAGE_MARKERS = ("package.xml", "manifest.xml", "CATKIN_IGNORE")
#: How far above the URDF to look for sibling packages.
_MAX_SEARCH_DEPTH = 6


def _index_packages(roots: list[Path]) -> dict[str, Path]:
    """Map package name -> directory, by scanning for package markers."""
    found: dict[str, Path] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for marker in _PACKAGE_MARKERS:
            for path in root.rglob(marker):
                package_dir = path.parent
                found.setdefault(package_dir.name, package_dir)
    return found


def _search_roots(urdf_path: Path, extra: list[Path] | None) -> list[Path]:
    roots: list[Path] = list(extra or [])
    current = urdf_path.parent if urdf_path.is_file() else urdf_path
    for _ in range(_MAX_SEARCH_DEPTH):
        roots.append(current)
        if current.parent == current:
            break
        current = current.parent
    return roots


def make_mesh_resolver(
    urdf_path: str | Path,
    *,
    extra_search_paths: list[str | Path] | None = None,
):
    """Build a callable mapping a URDF mesh filename to an absolute path.

    The package index is built lazily and cached: a full ``rglob`` over a large
    description tree is not free, and most URDFs reference only a handful of
    packages.

    The resolver returns its best guess even when the file does not exist, so
    the caller can report a specific missing path rather than a vague failure.
    """
    urdf_path = Path(urdf_path).resolve()
    base_dir = urdf_path.parent if urdf_path.is_file() else urdf_path
    extra = [Path(p).resolve() for p in (extra_search_paths or [])]

    cache: dict[str, Path] = {}
    package_index: dict[str, Path] | None = None

    def packages() -> dict[str, Path]:
        nonlocal package_index
        if package_index is None:
            package_index = _index_packages(_search_roots(urdf_path, extra))
        return package_index

    def find_package(name: str) -> Path | None:
        if name in packages():
            return packages()[name]
        # Fall back to any directory of that name; descriptions vendored into a
        # repo often omit package.xml.
        for root in _search_roots(urdf_path, extra):
            candidate = root / name
            if candidate.is_dir():
                packages()[name] = candidate
                return candidate
        return None

    def resolve(filename: str) -> str:
        if filename in cache:
            return str(cache[filename])

        raw = str(filename).strip()
        parsed = urlparse(raw)
        result: Path | None = None

        if parsed.scheme in ("package", "model"):
            package_name = parsed.netloc
            relative = unquote(parsed.path).lstrip("/")
            package_dir = find_package(package_name)
            if package_dir is not None:
                result = package_dir / relative
            else:
                # Some descriptions self-reference their own package; the URDF
                # directory is then the package root.
                result = base_dir / relative
        elif parsed.scheme == "file":
            path = unquote(parsed.path)
            # urlparse leaves a leading slash on Windows drive paths.
            if os.name == "nt" and path.startswith("/") and len(path) > 2 and path[2] == ":":
                path = path[1:]
            result = Path(path)
        else:
            candidate = Path(raw)
            result = candidate if candidate.is_absolute() else (base_dir / candidate)

        result = Path(os.path.normpath(result))
        if not result.is_file():
            # Last resort: match by basename anywhere under the description.
            matches = list(base_dir.rglob(result.name))
            if len(matches) == 1:
                result = matches[0]

        cache[filename] = result
        return str(result)

    return resolve
