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
#: A directory containing one of these is the top of a checkout. Package
#: discovery stops here: whatever is above it belongs to somebody else.
_BOUNDARY_MARKERS = (".git", ".hg", ".svn", ".repo", "COLCON_IGNORE")


def is_package_dir(directory: Path) -> bool:
    return any((directory / marker).exists() for marker in _PACKAGE_MARKERS)


def package_search_root(path: str | Path) -> Path:
    """The one directory worth scanning for ROS packages near ``path``.

    A ROS description repository puts sibling packages side by side, so the
    place to look is the *parent* of the package containing the file:
    ``kuka_lbr_iiwa_support`` and ``kuka_resources`` are neighbours, and a robot
    in the first reaches materials in the second.

    Bounded deliberately, and that is the point rather than a detail. The
    obvious implementation -- walk up a few levels and scan each -- reaches the
    filesystem root for a description saved anywhere shallow, and scanning a
    whole drive for ``package.xml`` does not fail, it just never finishes.

    So: stop at the checkout boundary, never return a filesystem root, and fall
    back to the file's own directory when there is no containing package at all
    -- a lone URDF beside a ``meshes/`` folder has no packages to find, and its
    references resolve relatively.
    """
    path = Path(path)
    start = path.parent if path.is_file() else path

    current = start
    for _ in range(_MAX_SEARCH_DEPTH):
        if any((current / marker).exists() for marker in _BOUNDARY_MARKERS):
            # The checkout root. Its children are the packages.
            return current
        if is_package_dir(current):
            parent = current.parent
            # A package directly at the filesystem root has no siblings worth
            # scanning, and its parent is the root itself.
            return current if parent == current else parent
        if current.parent == current:
            break
        current = current.parent

    return start


def _index_packages(root: Path) -> dict[str, Path]:
    """Map package name -> directory, by scanning one root for markers.

    One root, not a list of ancestors: this is the expensive call, and its cost
    is the size of whatever it is pointed at. See :func:`package_search_root`.
    """
    found: dict[str, Path] = {}
    if not root.is_dir():
        return found
    for marker in _PACKAGE_MARKERS:
        for path in root.rglob(marker):
            package_dir = path.parent
            found.setdefault(package_dir.name, package_dir)
    return found


def _search_roots(urdf_path: Path, extra: list[Path] | None) -> list[Path]:
    """Ancestors to try for the cheap ``root / name`` lookup.

    Still walks broadly, unlike :func:`package_search_root`, because every use
    of this list is a single ``is_dir()`` check rather than a tree scan. It is
    also what rescues a description vendored into a repo without a
    ``package.xml`` anywhere.
    """
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
            package_index = _index_packages(package_search_root(urdf_path))
            for root in extra:
                # Caller-supplied roots are explicit, so scan them too.
                for name, directory in _index_packages(root).items():
                    package_index.setdefault(name, directory)
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
            # Two shapes arrive here. A well-formed URI carries everything in
            # ``path``: ``file:///home/u/x`` and ``file:///C:/x``, the latter
            # keeping a leading slash urlparse does not strip.
            #
            # xacrodoc emits the other shape. Rendering a xacro with
            # resolve_packages=True rewrites every ``package://`` into
            # ``file://<absolute path>`` -- and on Windows that is
            # ``file://C:\...``, which has no slash after the authority marker,
            # so urlparse puts the entire path in ``netloc`` and leaves ``path``
            # empty. Reading ``path`` alone yielded "" and every mesh resolved
            # to the URDF's own directory, which exists, so the robot imported
            # in silence with no geometry at all.
            path = unquote(parsed.netloc + parsed.path)
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
