"""What importing a Catalog Folder entry reads (#105).

:func:`importable` is the one place an entry becomes something to import, so
every route that imports one -- the catalog search, and later the Asset Browser
-- reads entries the same way. Today that is always the entry's description
file, with the xacro arguments its index names. An entry with a built asset
``.blend`` (kinema_catalog's release zips, #118) is to be answered here too, and
nothing that calls this needs to change for it.

Packages aren't part of the answer. A description in a catalog is searched with
every repository in the catalog's ``src/``, which ``io.resolve`` works out from
the file's place (:func:`kinema.catalog.folder.containing`): one description
reaches into another's repository, as TurtleBot 4 does into Create 3's, and the
solver's later reload of the rig has to find the same packages.

Nothing here uses the network, and nothing here imports ``bpy``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .folder import SOURCES, FolderEntry

#: The clone command for one submodule, over anonymous HTTPS as kinema_catalog's
#: launcher clones them: every fork is public, and ``.gitmodules`` names SSH URLs,
#: which need a key.
CLONE = (
    'git -C "{root}" -c url."https://github.com/".insteadOf=git@github.com: '
    "submodule update --init -- {submodule}"
)


@dataclass(frozen=True)
class Importable:
    """An entry, as it is imported. ``problem`` is set when it can't be: the
    other fields then say nothing."""

    #: The description file.
    path: Path
    #: The entry's xacro arguments, in the syntax the import takes:
    #: ``name:=ur5e ur_type:=ur5e``.
    xacro_args: str = ""
    #: Why it can't be imported, in a line.
    problem: str = ""
    #: The command that would make it importable, to copy, where there is one.
    command: str = ""


def importable(entry: FolderEntry) -> Importable:
    """What importing ``entry`` reads, or why it can't."""
    from ..io.xacro_args import format_args

    submodule = Path(SOURCES, entry.submodule)
    if not _has_files(entry.root / submodule):
        # kinema_catalog clones its submodules on demand; an unpacked zip has
        # no git to clone them with.
        is_clone = (entry.root / ".git").exists()
        return Importable(
            entry.path,
            problem=(
                f"{entry.name}: {submodule.as_posix()} isn't cloned"
                if is_clone
                else f"{entry.name}: {submodule.as_posix()} isn't in {entry.root}"
            ),
            command=(
                CLONE.format(root=entry.root, submodule=submodule.as_posix())
                if is_clone
                else ""
            ),
        )
    try:
        present = entry.path.is_file()
    except OSError:
        present = False
    if not present:
        return Importable(entry.path, problem=f"{entry.name}: no {entry.relative_path}")
    return Importable(entry.path, xacro_args=format_args(dict(entry.args)))


def _has_files(directory: Path) -> bool:
    """Whether ``directory`` holds anything: a submodule not yet cloned is an
    empty directory, or none."""
    try:
        return any(directory.iterdir())
    except OSError:
        return False
