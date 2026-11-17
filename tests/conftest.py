"""Shared pytest fixtures.

Tests split into two groups:

* ``tests/unit`` -- pure Python (URL mapping, parsers). Runs anywhere.
* ``tests/integration`` -- needs a real ``bpy``, so it only runs inside Blender
  via ``tools/dev.py test``. Collected but skipped elsewhere.

The add-on is imported by its extension module name (``bl_ext.<repo>.kinema``)
rather than as ``kinema``: that is how Blender actually loads it, so importing
it any other way would test a module layout that never exists in production.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"
EXTENSION_ID = "bl_ext.user_default.kinema"

try:
    import bpy  # noqa: F401
    HAVE_BPY = True
except ImportError:
    HAVE_BPY = False
    # Outside Blender, make `import kinema...` resolve for the pure-Python tests.
    sys.path.insert(0, str(REPO_ROOT / "src"))

requires_bpy = pytest.mark.skipif(not HAVE_BPY, reason="requires Blender's bpy")


def load_addon_module(dotted: str):
    """Import an add-on submodule, with or without Blender.

    Inside Blender the add-on is a package whose ``__init__`` imports ``bpy``,
    so a submodule is reached normally through the extension module name.
    Outside Blender that import fails, so any submodule that is itself
    bpy-free (the parsers, the fetcher) is loaded directly from its file --
    the same source, without executing the package ``__init__``.

    Args:
        dotted: submodule path relative to the add-on, e.g. ``"catalog.fetch"``.
    """
    if HAVE_BPY:
        return importlib.import_module(f"{EXTENSION_ID}.{dotted}")
    return importlib.import_module(f"{OFFLINE_PACKAGE}.{dotted}")


#: Synthetic package standing in for the add-on outside Blender.
OFFLINE_PACKAGE = "kinema_offline"


def _install_offline_package() -> None:
    """Make ``src/kinema`` importable without running its ``__init__``.

    The real package's ``__init__`` imports ``bpy``, so it cannot execute here.
    But loading individual files standalone breaks any module using a relative
    import (``from .base import ...``). The fix is a stub package object whose
    ``__path__`` points at the source tree: submodules then import normally and
    their relative imports resolve, while the bpy-importing ``__init__`` is
    never run.
    """
    if OFFLINE_PACKAGE in sys.modules:
        return
    package = types.ModuleType(OFFLINE_PACKAGE)
    package.__path__ = [str(REPO_ROOT / "src" / "kinema")]
    sys.modules[OFFLINE_PACKAGE] = package


if not HAVE_BPY:
    _install_offline_package()


@pytest.fixture(scope="session")
def addon():
    """The enabled Kinema add-on module, as Blender loads it."""
    if not HAVE_BPY:
        pytest.skip("requires Blender")
    import addon_utils
    import bpy

    bpy.ops.preferences.addon_refresh()
    addon_utils.enable(EXTENSION_ID, default_set=True, persistent=True)
    if not addon_utils.check(EXTENSION_ID)[1]:
        pytest.fail(f"could not enable {EXTENSION_ID}; run 'dev.py link' first")
    return importlib.import_module(EXTENSION_ID)


@pytest.fixture
def clean_scene():
    """An empty scene, without resetting preferences.

    ``wm.read_factory_settings`` would reset preferences, which disables the
    extension and uninstalls its bundled wheels mid-run -- so datablocks are
    removed by hand instead.
    """
    if not HAVE_BPY:
        pytest.skip("requires Blender")
    import bpy

    def purge() -> None:
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        for mesh in list(bpy.data.meshes):
            bpy.data.meshes.remove(mesh)
        for material in list(bpy.data.materials):
            bpy.data.materials.remove(material)
        for armature in list(bpy.data.armatures):
            bpy.data.armatures.remove(armature)

    purge()
    yield bpy.context.scene
    purge()


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    return FIXTURE_DIR


# --------------------------------------------------------------------------
# running a share of the suite, and timing it: tools/dev.py test -j, --record-durations
# --------------------------------------------------------------------------
@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(session, config, items):
    """Keep this process's share of the suite, where ``KINEMA_TEST_SHARD`` names one.

    First, before ``-k``, ``-m`` or ``--deselect`` take any away: what was
    collected is then every test in the paths asked for.
    """
    import os

    from .sharding import covers_suite, parse_shard, read_durations, shares

    if os.environ.get("KINEMA_RECORD_DURATIONS"):
        # For dev.py to drop the times of tests that are gone: only where this
        # is the whole suite, and collecting it failed nowhere.
        _collected[:] = [item.nodeid for item in items]
        source = getattr(config, "args_source", None)
        base = config.rootpath if source == config.ArgsSource.TESTPATHS else (
            config.invocation_params.dir
        )
        _whole[:] = [covers_suite(config.args, base) and not session.testsfailed]
    shard = os.environ.get("KINEMA_TEST_SHARD")
    if not shard:
        return

    index, count = parse_shard(shard)
    owners = shares([item.nodeid for item in items], read_durations(), count)
    kept = [item for item, owner in zip(items, owners, strict=True) if owner == index]
    dropped = [item for item, owner in zip(items, owners, strict=True) if owner != index]
    if dropped:
        config.hook.pytest_deselected(items=dropped)
    items[:] = kept


_measured: dict[str, float] = {}
_collected: list[str] = []
_whole: list[bool] = [False]


def pytest_runtest_logreport(report):
    """Add up each test's setup, call and teardown, where they are being recorded."""
    import os

    if os.environ.get("KINEMA_RECORD_DURATIONS"):
        _measured[report.nodeid] = _measured.get(report.nodeid, 0.0) + float(report.duration)


def pytest_sessionfinish(session, exitstatus):
    """Write the times measured, and every test collected, to the file
    ``KINEMA_RECORD_DURATIONS`` names."""
    import json
    import os

    path = os.environ.get("KINEMA_RECORD_DURATIONS")
    if path:
        times = {k: round(v, 3) for k, v in sorted(_measured.items())}
        Path(path).write_text(
            json.dumps({"whole": _whole[0], "collected": _collected, "times": times}, indent=0),
            encoding="utf-8",
        )
