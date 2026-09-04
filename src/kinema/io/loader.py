"""Turn a file path into a :class:`RobotModel`.

Everything here is deliberately free of ``bpy``: reading a description is a
parsing problem, and keeping Blender out of it means this half can be tested in
a plain Python venv, which is where most of the suite runs.

Failures are returned, not raised. Every entry point answers with a
:class:`LoadResult` carrying either a model or a message the operator can put
in front of the user, so a malformed URDF never raises out of an operator.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..rig.kinematics import RobotModel

#: File suffixes that are unambiguously URDF rather than MJCF.
_URDF_SUFFIXES = {".urdf", ".xacro"}


@dataclass
class LoadResult:
    """The outcome of one load. Exactly one of ``model`` or ``error`` is set."""

    model: RobotModel | None = None
    #: ``(kind, value)`` written onto the rig so the solver can reload the
    #: description later: ("file", path) or ("mjcf", path).
    source: tuple[str, str] | None = None
    error: str | None = None
    cancelled: bool = False
    #: yourdfpy's own mesh-path resolver, when the model came from a URDF.
    resolver: Callable[[str], str] | None = None


def looks_like_mjcf(path: Path) -> bool:
    """True if the file's root element is ``<mujoco>``.

    MJCF and URDF both commonly use .xml, so the extension decides nothing.
    Only the opening bytes are read -- some MJCF scenes are large.
    """
    if path.suffix.lower() in _URDF_SUFFIXES:
        return False
    try:
        with open(path, "rb") as handle:
            head = handle.read(4096).decode("utf-8", "ignore")
    except OSError:
        return False
    return "<mujoco" in head


def _cancelled_result() -> LoadResult:
    return LoadResult(cancelled=True, error="Import cancelled")


def load_file(
    filepath: str | Path,
    *,
    should_cancel: Callable[[], bool] | None = None,
    xacro_args: dict[str, str] | None = None,
    extra_search_paths: list[str] | None = None,
) -> LoadResult:
    """Parse a local URDF, xacro or MJCF file.

    No network, but the parse alone is slow enough on a large robot to be worth
    keeping off the main thread -- and it puts both import routes through the
    same operator.

    ``xacro_args`` are substitution arguments, as ``xacro name:=value`` takes
    them; ignored for a plain URDF. ``extra_search_paths`` are additional
    directories to find ROS packages in, for a cell whose macros live in a
    different repository from the robot.
    """
    from ..rig import kinematics

    path = Path(filepath)
    if not path.is_file():
        return LoadResult(error=f"No such file: {path}")

    if looks_like_mjcf(path):
        return _model_from_mjcf(path, ("mjcf", str(path)))

    try:
        import yourdfpy
    except ImportError:
        return LoadResult(error="yourdfpy is unavailable; check Kinema's dependencies")

    from .resolve import make_mesh_resolver

    resolver = make_mesh_resolver(path, extra_search_paths=extra_search_paths)
    is_xacro = looks_like_xacro(path)
    try:
        if is_xacro:
            urdf = load_xacro_urdf(
                path, resolver, xacro_args=xacro_args,
                extra_search_paths=extra_search_paths,
            )
        else:
            urdf = yourdfpy.URDF.load(
                str(path),
                build_scene_graph=True,
                load_meshes=False,
                filename_handler=lambda name: resolver(name),
            )
    except Exception as exc:  # noqa: BLE001
        if is_xacro:
            # "Undefined substitution argument name" tells a user nothing --
            # 'name' even reads as a noun. Name the argument and list what the
            # file declares.
            from .xacro_args import describe

            return LoadResult(error=describe(path, str(exc)))
        return LoadResult(error=f"Could not parse {path.name}: {exc}")

    try:
        model = kinematics.model_from_urdf(urdf, mesh_resolver=resolver)
    except kinematics.UnsupportedJointError as exc:
        return LoadResult(error=str(exc))
    except Exception as exc:  # noqa: BLE001
        return LoadResult(error=f"Could not read robot: {exc}")

    return LoadResult(model=model, source=("file", str(path)), resolver=resolver)


def looks_like_xacro(path: str | Path) -> bool:
    """Whether this file needs rendering before a URDF parser sees it.

    Shared with the solver, which reloads the description later to build
    PyRoki's model and must make the same decision. When it did not, a xacro
    reached yourdfpy raw and the rig silently dropped to the NumPy backend.
    """
    path = Path(path)
    return path.suffix.lower() == ".xacro" or path.name.endswith(".urdf.xacro")


def load_xacro_urdf(
    path: Path,
    resolver,
    xacro_args: dict[str, str] | None = None,
    extra_search_paths: list[str] | None = None,
):
    """Render a xacro to URDF first; many ROS descriptions ship only xacro.

    xacro reaches other packages by name -- a KUKA arm pulls its materials from
    a sibling ``kuka_resources`` -- and xacrodoc cannot find those on its own.
    It resolves the file's *own* package and nothing beside it, so any
    cross-package ``$(find …)`` failed with ``PackageNotFoundError``.

    Kinema hands it a finished map rather than a directory to search, which buys
    two things beyond the lookup. The two resolvers can no longer disagree about
    what a package is called -- the map is built by the same indexer that
    resolves meshes, so both know Universal_Robots_ROS2_Description declares
    itself ``ur_description``. And xacrodoc never opens ``package.xml`` itself,
    which is what stops it decoding a UTF-8 file with the system codec: that
    file's maintainer name contains a Danish ø, and on Windows the render died
    with ``'charmap' codec can't decode byte 0x9d``.

    ``reset()`` matters as much as the map. xacrodoc's package finder is
    module-global, so without it one import's packages stay resolvable in the
    next, and two robots from different repos that both ship a
    ``common_materials.xacro`` would quietly render against whichever was
    loaded first.

    xacrodoc's own ``temp_urdf_file_path`` cannot be used. It yields the path of
    a NamedTemporaryFile it is still holding open, and Windows refuses to open a
    file that another handle has -- so every xacro import failed there with
    ``WinError 32``. Writing the render ourselves and closing it first is the
    fix; the temp directory is fine for it, because ``resolver`` was built from
    the *xacro's* path and is what yourdfpy defers every mesh filename to.
    """
    import tempfile

    import yourdfpy
    from xacrodoc import XacroDoc, packages

    from .resolve import package_map

    packages.reset()
    known = package_map(path, extra_search_paths=extra_search_paths)
    if known:
        packages.update_package_cache({name: str(p) for name, p in known.items()})

    doc = XacroDoc.from_file(
        str(path), resolve_packages=True, subargs=dict(xacro_args or {})
    )

    handle, rendered = tempfile.mkstemp(suffix=".urdf", prefix=f"kinema-{path.stem}-")
    os.close(handle)
    rendered_path = Path(rendered)
    try:
        rendered_path.write_text(doc.to_urdf_string(), encoding="utf-8")
        return yourdfpy.URDF.load(
            str(rendered_path),
            build_scene_graph=True,
            load_meshes=False,
            filename_handler=lambda name: resolver(name),
        )
    finally:
        rendered_path.unlink(missing_ok=True)


def _model_from_mjcf(path: str | Path, source: tuple[str, str]) -> LoadResult:
    from . import mjcf

    try:
        return LoadResult(model=mjcf.model_from_mjcf(path), source=source)
    except mjcf.MjcfError as exc:
        return LoadResult(error=str(exc))
    except Exception as exc:  # noqa: BLE001
        return LoadResult(error=f"Could not parse {Path(path).name}: {exc}")
