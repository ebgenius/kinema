"""Runtime bootstrap for Kinema's heavy dependencies.

Everything in this module exists to make the JAX/PyRoki stack behave inside a
DCC application, which is not what it was written for. The findings encoded
here came out of the M0 feasibility spike against Blender 5.2.0 LTS.

Design rules:

* **Nothing heavy is imported at add-on registration time.** Importing JAX costs
  roughly 2-5 s and Blender registers add-ons on the main thread during startup.
  The solver stack is imported on first use, and can be warmed in the
  background; until then the NumPy fallback backend answers every solve.
* **``JAX_PLATFORMS`` is set before JAX is ever imported.** Left unset, JAX
  probes for GPU/TPU backends, which inside Blender produces console noise and
  can stall. Kinema is a CPU workload -- IK on a 6-DoF arm is microseconds of
  actual maths.
* **jaxls logs at INFO through loguru on every ``analyze()``.** In a live
  viewport handler that is several lines of console spam per mouse move, so its
  sink is muted unless the user turns on debug logging.
"""

from __future__ import annotations

import importlib
import os
import sys
import threading
from pathlib import Path
from types import ModuleType

ADDON_DIR = Path(__file__).resolve().parent
VENDOR_DIR = ADDON_DIR / "vendor"

#: Populated by :func:`load_solver_stack`. Never import these at module scope.
_stack: dict[str, ModuleType] = {}
_load_lock = threading.Lock()
_load_error: str | None = None


# --------------------------------------------------------------------------
# sys.path and environment
# --------------------------------------------------------------------------
def ensure_vendor_path() -> None:
    """Put the vendored packages on ``sys.path``.

    ``pyroki`` and ``jaxls`` are vendored rather than bundled as wheels because
    neither is installable from PyPI (see ``tools/vendor.py``). They must be
    importable under their canonical top-level names: pyroki does ``import
    jaxls``, and jaxls does ``from jaxls._preconditioning import ...``.

    Aliasing them into ``sys.modules`` under ``kinema.vendor.*`` instead would
    let the same source be imported under two names, producing two distinct
    module objects -- and jax_dataclasses registers JAX pytree nodes at import
    time, so a double import raises a duplicate-registration error. One path
    entry, one module identity.
    """
    path = str(VENDOR_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)


def remove_vendor_path() -> None:
    """Undo :func:`ensure_vendor_path` when the add-on is disabled."""
    path = str(VENDOR_DIR)
    while path in sys.path:
        sys.path.remove(path)


def configure_jax_env() -> None:
    """Pin JAX to CPU. Must run before the first ``import jax``."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    # JAX preallocates and probes at import; neither helps a CPU IK workload.
    os.environ.setdefault("JAX_ENABLE_X64", "0")


def silence_vendor_logging(debug: bool = False) -> None:
    """Mute jaxls' per-solve loguru output unless debugging.

    jaxls emits four INFO lines from ``LeastSquaresProblem.analyze()``. That is
    fine in a script and intolerable from a viewport handler firing on every
    mouse move.
    """
    try:
        from loguru import logger
    except ImportError:
        return
    logger.remove()
    if debug:
        logger.add(sys.stderr, level="DEBUG")


# --------------------------------------------------------------------------
# Solver stack
# --------------------------------------------------------------------------
def solver_available() -> bool:
    """True if the PyRoki stack is loaded and ready to solve."""
    return bool(_stack)


def solver_error() -> str | None:
    """The reason the stack failed to load, if it did."""
    return _load_error


def load_solver_stack(debug: bool = False) -> dict[str, ModuleType] | None:
    """Import JAX + jaxls + PyRoki. Returns the modules, or None on failure.

    Safe to call repeatedly and from any thread; the work happens once. Failure
    is not fatal -- the caller falls back to the NumPy backend, which is why
    this returns None rather than raising.
    """
    global _load_error
    if _stack:
        return _stack
    with _load_lock:
        if _stack:
            return _stack
        try:
            ensure_vendor_path()
            configure_jax_env()

            import jax
            import jax.numpy as jnp
            import jax_dataclasses as jdc
            import jaxlie
            import jaxls
            import pyroki

            silence_vendor_logging(debug=debug)
            _stack.update(
                jax=jax, jnp=jnp, jdc=jdc, jaxlie=jaxlie, jaxls=jaxls, pyroki=pyroki
            )
            _load_error = None
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI, never fatal
            _load_error = f"{type(exc).__name__}: {exc}"
            _stack.clear()
            return None
    return _stack


def warm_up_async(on_done=None, debug: bool = False) -> threading.Thread:
    """Load the solver stack on a worker thread.

    Only the import is backgrounded. JIT warmup needs a built robot model, so
    it happens in the solver backend once a robot exists. Nothing here touches
    ``bpy`` -- Blender's API is not thread-safe, so the callback must marshal
    back to the main thread (a one-shot timer) before touching scene data.
    """
    def _work() -> None:
        load_solver_stack(debug=debug)
        if on_done is not None:
            on_done(solver_available())

    thread = threading.Thread(target=_work, name="kinema-solver-warmup", daemon=True)
    thread.start()
    return thread


def unload_solver_stack() -> None:
    """Drop references so a disabled add-on does not pin ~500 MB of JAX."""
    _stack.clear()


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------
def dependency_report() -> list[tuple[str, str, str]]:
    """(name, status, detail) rows for the preferences panel.

    Used to tell a user with a broken install *which* piece is missing, rather
    than failing with an opaque ImportError deep in a handler.
    """
    ensure_vendor_path()
    rows: list[tuple[str, str, str]] = []
    for name, why in (
        ("numpy", "array maths (provided by Blender)"),
        ("scipy", "required by JAX"),
        ("jax", "solver core"),
        ("jaxlib", "solver core (compiled)"),
        ("jaxls", "least-squares optimiser (vendored)"),
        ("pyroki", "robot kinematics (vendored)"),
        ("yourdfpy", "URDF parsing"),
        ("collada", "COLLADA meshes, from pycollada (Blender 5 removed its own)"),
        ("robot_descriptions", "robot catalog"),
        ("trimesh", "mesh utilities"),
    ):
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "")
            rows.append((name, "ok", version or why))
        except Exception as exc:  # noqa: BLE001
            rows.append((name, "missing", f"{type(exc).__name__}: {exc}"[:80]))
    return rows
