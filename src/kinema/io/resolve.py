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
import re
from pathlib import Path
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree

#: ``C:`` at the start of what urlparse took for an authority: the signature of
#: a Windows path in a ``file://`` URI that has no third slash.
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")

#: A directory containing one of these is a ROS package root.
_PACKAGE_MARKERS = ("package.xml", "manifest.xml", "CATKIN_IGNORE")
#: How far above the URDF to look for sibling packages.
_MAX_SEARCH_DEPTH = 6
#: A directory containing one of these is the top of a checkout. Package
#: discovery stops here: whatever is above it belongs to somebody else.
_BOUNDARY_MARKERS = (".git", ".hg", ".svn", ".repo", "COLCON_IGNORE")


def is_package_dir(directory: Path) -> bool:
    return any((directory / marker).exists() for marker in _PACKAGE_MARKERS)


def declared_package_name(package_dir: Path) -> str | None:
    """The ``<name>`` a ``package.xml`` declares, or None.

    Not the same as the directory name, and assuming otherwise is a real bug:
    Universal_Robots_ROS2_Description declares ``ur_description``, so every
    ``$(find ur_description)`` in it failed to resolve against an index keyed on
    the folder.

    Parsed with ``ElementTree``, which reads bytes and honours the XML
    declaration. Reading the text first and decoding with the system codec is
    what made this file crash on Windows -- it contains a maintainer name with
    a Danish ø, and cp1252 cannot decode UTF-8. Forcing UTF-8 instead would only
    move the failure to a file that declares something else.

    Returns None rather than raising for anything unreadable: an odd package.xml
    somewhere in a checkout must not stop an unrelated robot from importing.
    """
    manifest = package_dir / "package.xml"
    if not manifest.is_file():
        # ROS 1 manifest.xml carries no <name>; the directory *is* the package.
        return None
    try:
        node = ElementTree.parse(manifest).getroot().find("name")
    except (ElementTree.ParseError, OSError, ValueError):
        return None
    if node is None or not node.text:
        return None
    return node.text.strip() or None


def package_names(package_dir: Path) -> list[str]:
    """Every name a package should answer to, best first.

    Both, when they differ: descriptions in the wild reference each other by
    either, and indexing the declared name alone would break a repository whose
    own files say ``$(find Universal_Robots_ROS2_Description)``.
    """
    declared = declared_package_name(package_dir)
    if declared and declared != package_dir.name:
        return [declared, package_dir.name]
    return [declared or package_dir.name]


def _is_filesystem_root(directory: Path) -> bool:
    """``C:\\``, ``/``, or a UNC share root -- never safe to scan."""
    return bool(directory.anchor) and directory == Path(directory.anchor)


def package_search_root(path: str | Path) -> Path | None:
    """The one directory worth scanning for ROS packages near ``path``.

    A ROS description repository puts sibling packages side by side, so the
    place to look is the *parent* of the package containing the file:
    ``kuka_lbr_iiwa_support`` and ``kuka_resources`` are neighbours, and a robot
    in the first reaches materials in the second.

    Bounded deliberately, and that is the point rather than a detail. The
    obvious implementation -- walk up a few levels and scan each -- reaches the
    filesystem root for a description saved anywhere shallow, and scanning a
    whole drive for ``package.xml`` does not fail, it just never finishes.

    Returns **None** when no directory qualifies, rather than falling back to
    something plausible. A package unpacked directly at ``C:\\`` has siblings,
    but they are every top-level folder on the drive; there is no honest answer
    there, and returning the root would hand callers precisely the scan this
    exists to prevent. Callers skip indexing instead, which costs cross-package
    resolution for that one layout and keeps the import responsive.
    """
    path = Path(path).resolve()
    start = path.parent if path.is_file() else path

    current = start
    for _ in range(_MAX_SEARCH_DEPTH):
        # A root is never an answer, but walking *through* one is fine and says
        # nothing about the file: on POSIX any loose file a few levels down
        # reaches ``/`` within six steps, and giving up there would refuse to
        # search the directory the file is actually sitting in.
        if not _is_filesystem_root(current):
            if any((current / marker).exists() for marker in _BOUNDARY_MARKERS):
                # The checkout root. Its children are the packages.
                return current
            if is_package_dir(current):
                parent = current.parent
                return None if _is_filesystem_root(parent) else parent
        if current.parent == current:
            break
        current = current.parent

    return None if _is_filesystem_root(start) else start


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
            for name in package_names(package_dir):
                found.setdefault(name, package_dir)
    return found


def package_map(
    path: str | Path,
    *,
    extra_search_paths: list[str | Path] | None = None,
) -> dict[str, Path]:
    """Every ROS package reachable from ``path``, name -> directory.

    The single source of truth for "what packages exist near this file", shared
    by the mesh resolver and by the xacro renderer. They used to answer that
    question separately and could disagree -- one keyed on directory names, the
    other on what ``package.xml`` declared.

    ``extra_search_paths`` is how a cell spanning repositories works: the
    robot's own checkout is found automatically, and the shared macro
    repositories come from the user's preferences.
    """
    found: dict[str, Path] = {}
    root = package_search_root(path)
    if root is not None:
        found.update(_index_packages(root))
    for extra in extra_search_paths or []:
        for name, directory in _index_packages(Path(extra).resolve()).items():
            found.setdefault(name, directory)
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
            package_index = package_map(urdf_path, extra_search_paths=extra)
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
            # Three shapes arrive here, and the authority is what separates
            # them.
            #
            # A well-formed URI has no authority and carries everything in
            # ``path``: ``file:///home/u/x``, or ``file:///C:/x`` keeping a
            # leading slash urlparse does not strip. ``localhost`` is defined as
            # equivalent to empty.
            #
            # A real authority means UNC: ``file://server/share/x`` is
            # ``\\server\share\x``, and dropping the leading slashes would turn
            # it into a relative path.
            #
            # xacrodoc emits the third, in two spellings. Rendering a xacro
            # with resolve_packages=True rewrites every ``package://`` into
            # ``file://<absolute path>``, and on Windows the drive letter lands
            # in the authority either way:
            #
            #   file://C:\pkg\mesh.stl   netloc='C:\\pkg\\mesh.stl'  path=''
            #   file://C:/pkg/mesh.stl   netloc='C:'  path='/pkg/mesh.stl'
            #
            # Which one appears depends on how the package was registered:
            # ``look_in`` keeps the native path, while ``update_package_cache``
            # normalises through ``as_posix``. Measured on the current bundle,
            # the path Kinema takes produces the first -- but both are real, and
            # the branch has to survive a change of registration mechanism.
            #
            # Reading ``path`` alone handled neither: it gave "" for the first
            # and silently dropped the drive from the second. Every mesh then
            # resolved to the URDF's own directory, which exists, so the robot
            # imported in silence with no geometry at all.
            authority = unquote(parsed.netloc)
            path = unquote(parsed.path)
            if _WINDOWS_DRIVE.match(authority):
                path = authority + path
            elif authority and authority.lower() != "localhost":
                path = f"//{authority}{path}"
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
