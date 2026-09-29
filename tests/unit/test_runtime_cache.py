"""Compiled solvers kept on disk: the configuration, and the folder's upkeep.

What happens inside Blender -- the folder being the add-on's own, a compile
landing in it -- is in ``tests/integration/test_compile_cache.py``.
"""

from __future__ import annotations

import os

import pytest

from ..conftest import HAVE_BPY, load_addon_module

runtime = load_addon_module("runtime")


@pytest.fixture
def jax():
    """JAX, with the cache settings put back after: they are process-wide."""
    module = pytest.importorskip("jax")
    before = (
        module.config.jax_compilation_cache_dir,
        module.config.jax_compilation_cache_max_size,
    )
    yield module
    module.config.update("jax_compilation_cache_dir", before[0])
    module.config.update("jax_compilation_cache_max_size", before[1])


def test_a_folder_turns_the_cache_on_and_leaves_jaxs_cap_alone(jax, tmp_path):
    """JAX's own cap needs the ``filelock`` package, which Kinema does not
    bundle. Set without it, every compile in the session fails -- IK included --
    and a dev venv that happens to have it would never show that here. So the
    cap must stay at JAX's default, -1: none."""
    jax.config.update("jax_compilation_cache_max_size", -1)
    runtime.configure_jax(jax, str(tmp_path))
    assert jax.config.jax_compilation_cache_dir == str(tmp_path)
    assert jax.config.jax_compilation_cache_max_size == -1


def test_no_folder_leaves_it_off(jax):
    jax.config.update("jax_compilation_cache_dir", None)
    runtime.configure_jax(jax, None)
    assert jax.config.jax_compilation_cache_dir is None


@pytest.mark.skipif(HAVE_BPY, reason="outside Blender only")
def test_outside_blender_there_is_no_folder():
    assert runtime.compile_cache_folder() is None
    assert runtime.compile_cache_dir() is None


def test_usage_counts_entries_and_bytes(tmp_path):
    (tmp_path / "jit__solve-abc-cache").write_bytes(b"x" * 100)
    (tmp_path / "jit__solve-abc-atime").write_bytes(b"y" * 8)
    assert runtime.compile_cache_usage(str(tmp_path)) == (1, 108)
    assert runtime.compile_cache_usage(str(tmp_path / "missing")) == (0, 0)
    assert runtime.compile_cache_usage(None) == (0, 0)


def test_pruning_keeps_the_folder_under_its_ceiling_oldest_first(tmp_path):
    """JAX records no access times without its lock, so written time decides."""
    for age, name in enumerate(["newest", "middle", "oldest"]):
        path = tmp_path / f"jit__solve-{name}-cache"
        path.write_bytes(b"x" * 100)
        stamp = 1_700_000_000 - age * 1000
        os.utime(path, (stamp, stamp))

    assert runtime.prune_compile_cache(str(tmp_path), max_bytes=250) == 1
    assert sorted(os.listdir(tmp_path)) == ["jit__solve-middle-cache", "jit__solve-newest-cache"]
    assert runtime.prune_compile_cache(str(tmp_path), max_bytes=250) == 0
    assert runtime.prune_compile_cache(None) == 0


def test_clearing_removes_files_and_nothing_else(tmp_path):
    """The folder is flat and JAX's: files go, and a folder or a link found in
    it is left alone rather than followed."""
    (tmp_path / "jit__solve-abc-cache").write_bytes(b"x")
    (tmp_path / "jit__solve-abc-atime").write_bytes(b"y")
    (tmp_path / "stray").mkdir()
    (tmp_path / "stray" / "keep.txt").write_text("keep")

    assert runtime.clear_compile_cache(str(tmp_path)) == 2
    assert sorted(os.listdir(tmp_path)) == ["stray"]
    assert (tmp_path / "stray" / "keep.txt").read_text() == "keep"
    assert runtime.clear_compile_cache(None) == 0
