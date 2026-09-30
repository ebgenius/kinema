"""Compiled solvers are kept on disk, in the add-on's own folder. Needs a real ``bpy``.

The claims:

* the folder is inside the add-on's user folder -- JAX runs what its cache holds,
  so the cache must never sit where another user could write;
* a session's IK compile lands in it, which is what lets the next session skip
  the compile and load the solver back instead;
* unticking *Keep Compiled Solvers* turns it off, and leaves the folder there to
  be cleared.

That the same solver comes back under the same key in a new process is the
vendored jaxls patch's doing: see ``tests/unit/test_jaxls_group_order.py``.
"""

from __future__ import annotations

import importlib
import os

import pytest

from ..conftest import requires_bpy

pytestmark = requires_bpy


@pytest.fixture
def runtime(addon):
    return importlib.import_module(f"{addon.__name__}.runtime")


@pytest.fixture
def manager(addon):
    return importlib.import_module(f"{addon.__name__}.solver.manager")


@pytest.fixture
def preferences(addon):
    prefs = importlib.import_module(f"{addon.__name__}.prefs").get_prefs()
    before = prefs.keep_compiled
    yield prefs
    prefs.keep_compiled = before


def _solver_entries(folder: str) -> set[str]:
    return {
        name for name in os.listdir(folder)
        if name.startswith("jit__solve") and name.endswith("-cache")
    }


def test_the_folder_is_the_addons_own(addon, runtime):
    import bpy

    folder = runtime.compile_cache_folder(create=True)
    own = bpy.utils.extension_path_user(addon.__name__)
    assert folder is not None and os.path.isdir(folder)
    assert os.path.commonpath([folder, own]) == own


def test_unticked_it_is_off_and_still_clearable(runtime, preferences):
    preferences.keep_compiled = False
    assert runtime.compile_cache_dir() is None
    assert runtime.compile_cache_folder() is not None


def test_an_ik_compile_is_kept_on_disk(
    addon, runtime, manager, preferences, fixture_dir, clean_scene
):
    """Cleared first, and solved on a fresh solver, so an entry left by an
    earlier run -- or a kernel already compiled in this process -- cannot pass
    for this compile's.

    JAX keeps only compiles over a second. arm6's takes about four here, but a
    faster machine could come in under, so the thresholds are lifted for the
    test and put back after: what is tested is that a compile lands, not how
    long it took."""
    import bpy

    preferences.keep_compiled = True
    stack = runtime.load_solver_stack()
    assert stack is not None, runtime.solver_error()
    folder = runtime.compile_cache_dir()
    config = stack["jax"].config
    assert config.jax_compilation_cache_dir == folder, (
        "the solver stack was loaded without the cache"
    )
    thresholds = (
        config.jax_persistent_cache_min_compile_time_secs,
        config.jax_persistent_cache_min_entry_size_bytes,
    )
    try:
        config.update("jax_persistent_cache_min_compile_time_secs", 0)
        config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        runtime.clear_compile_cache(folder)

        assert "FINISHED" in bpy.ops.kinema.build_robot(
            filepath=str(fixture_dir / "arm6.urdf")
        )
        rig = next(o for o in bpy.data.objects if o.get("kinema_rig"))
        bpy.context.view_layer.objects.active = rig
        rig.kinema_solver_mode = "PYROKI"
        manager.invalidate()
        assert "FINISHED" in bpy.ops.kinema.add_ik()  # warms the solver: the compile
        solver = manager.get_solver(rig)
        assert solver.pyroki(rig, build=False) is not None, "PyRoki did not compile"
        assert _solver_entries(folder), "the compiled IK solver was not kept"
    finally:
        config.update("jax_persistent_cache_min_compile_time_secs", thresholds[0])
        config.update("jax_persistent_cache_min_entry_size_bytes", thresholds[1])
        manager.invalidate()


def test_a_cache_folder_that_fails_still_loads_pyroki(runtime, preferences, monkeypatch):
    """Trimming the folder happens while the solver stack loads. An exception
    from it used to fail the load, and the whole session fell back to NumPy. A
    cache problem must mean no cache at most, and a failed trim not even that."""

    def refuse(path):
        raise PermissionError(path)

    preferences.keep_compiled = True
    monkeypatch.setattr(runtime.os, "scandir", refuse)
    runtime.unload_solver_stack()
    try:
        stack = runtime.load_solver_stack()
        assert stack is not None, runtime.solver_error()
        assert stack["jax"].config.jax_compilation_cache_dir == runtime.compile_cache_dir()
    finally:
        monkeypatch.undo()
        runtime.unload_solver_stack()
        assert runtime.load_solver_stack() is not None
