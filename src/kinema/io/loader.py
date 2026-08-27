"""Turn a catalog key or a file path into a :class:`RobotModel`, off-thread.

Everything here is deliberately free of ``bpy``. That is the whole point: it
lets the import operator run this entire half -- HTTPS download, tarball or
sparse fetch, xacro render, URDF/MJCF parse -- on a worker thread, and keep
Blender's event loop running meanwhile.

The remaining half, building the armature and importing meshes, cannot follow:
``bpy`` is not thread-safe. It is chunked across modal ticks instead
(``rig.builder.build_rig_iter``).

Failures are returned, not raised. A worker thread has no useful way to report
an exception to Blender's UI, so every entry point answers with a
:class:`LoadResult` carrying either a model or a message the operator can put
in front of the user.
"""

from __future__ import annotations

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
    #: description later: ("catalog", key), ("catalog-mjcf", key), ("file",
    #: path) or ("mjcf", path).
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


def load_catalog(
    key: str,
    *,
    progress: Callable[[float, int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> LoadResult:
    """Download (if needed) and parse one catalog description."""
    from ..catalog import fetch, index
    from ..rig import kinematics

    entry = index.get(key)
    if entry is None:
        return LoadResult(error=f"Unknown robot '{key}'")

    try:
        if entry.has_urdf:
            urdf = index.load_urdf(key, progress=progress, should_cancel=should_cancel)
            model = kinematics.model_from_urdf(
                urdf, mesh_resolver=getattr(urdf, "_filename_handler", None)
            )
            return LoadResult(
                model=model,
                source=("catalog", key),
                resolver=getattr(urdf, "_filename_handler", None),
            )

        path = index.mjcf_path(key, progress=progress, should_cancel=should_cancel)
        return _model_from_mjcf(path, ("catalog-mjcf", key))
    except fetch.FetchCancelled:
        return _cancelled_result()
    except kinematics.UnsupportedJointError as exc:
        return LoadResult(error=str(exc))
    except Exception as exc:  # noqa: BLE001 - surfaced in the UI by the operator
        return LoadResult(error=f"Could not load '{key}': {exc}")


def load_file(
    filepath: str | Path,
    *,
    should_cancel: Callable[[], bool] | None = None,
) -> LoadResult:
    """Parse a local URDF, xacro or MJCF file.

    No network, but the parse alone is slow enough on a large robot to be worth
    keeping off the main thread -- and it puts both import routes through the
    same operator.
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

    resolver = make_mesh_resolver(path)
    try:
        if path.suffix.lower() == ".xacro" or path.name.endswith(".urdf.xacro"):
            urdf = _load_xacro(path, resolver)
        else:
            urdf = yourdfpy.URDF.load(
                str(path),
                build_scene_graph=True,
                load_meshes=False,
                filename_handler=lambda name: resolver(name),
            )
    except Exception as exc:  # noqa: BLE001
        return LoadResult(error=f"Could not parse {path.name}: {exc}")

    try:
        model = kinematics.model_from_urdf(urdf, mesh_resolver=resolver)
    except kinematics.UnsupportedJointError as exc:
        return LoadResult(error=str(exc))
    except Exception as exc:  # noqa: BLE001
        return LoadResult(error=f"Could not read robot: {exc}")

    return LoadResult(model=model, source=("file", str(path)), resolver=resolver)


def _load_xacro(path: Path, resolver):
    """Render a xacro to URDF first; many ROS descriptions ship only xacro."""
    import yourdfpy
    from xacrodoc import XacroDoc

    doc = XacroDoc.from_file(str(path), resolve_packages=True)
    with doc.temp_urdf_file_path() as urdf_path:
        return yourdfpy.URDF.load(
            urdf_path,
            build_scene_graph=True,
            load_meshes=False,
            filename_handler=lambda name: resolver(name),
        )


def _model_from_mjcf(path: str | Path, source: tuple[str, str]) -> LoadResult:
    from . import mjcf

    try:
        return LoadResult(model=mjcf.model_from_mjcf(path), source=source)
    except mjcf.MjcfError as exc:
        return LoadResult(error=str(exc))
    except Exception as exc:  # noqa: BLE001
        return LoadResult(error=f"Could not parse {Path(path).name}: {exc}")
