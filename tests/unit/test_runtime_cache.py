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
    it is left alone rather than followed. Solvers are counted as the button
    counts them: an access-time file is not one."""
    (tmp_path / "jit__solve-abc-cache").write_bytes(b"x")
    (tmp_path / "jit__solve-abc-atime").write_bytes(b"y")
    (tmp_path / "stray").mkdir()
    (tmp_path / "stray" / "keep.txt").write_text("keep")

    assert runtime.clear_compile_cache(str(tmp_path)) == (1, 0)
    assert sorted(os.listdir(tmp_path)) == ["stray"]
    assert (tmp_path / "stray" / "keep.txt").read_text() == "keep"
    assert runtime.clear_compile_cache(None) == (0, 0)


# A cache problem must mean no cache: never a solver stack that fails to load,
# a preferences panel that fails to draw, or a Clear that stops half-way. The
# folder is shared with every other Blender running Kinema, and JAX writes to it.


class _Vanished:
    """A directory entry whose file went between listing and looking."""

    name = path = "jit__solve-gone-cache"

    def is_file(self, follow_symlinks=True):
        return True

    def stat(self, follow_symlinks=True):
        raise FileNotFoundError(self.path)


def _listing_with_a_vanished_file(real_scandir):
    class _Listing:
        def __init__(self, path):
            self.inner = real_scandir(path)

        def __enter__(self):
            return iter([*self.inner.__enter__(), _Vanished()])

        def __exit__(self, *exc):
            return self.inner.__exit__(*exc)

    return _Listing


def test_a_file_that_goes_mid_listing_is_skipped(tmp_path, monkeypatch):
    """Another Blender pruning, or JAX replacing an entry, between the listing
    and the stat."""
    (tmp_path / "jit__solve-kept-cache").write_bytes(b"x" * 10)
    monkeypatch.setattr(runtime.os, "scandir", _listing_with_a_vanished_file(os.scandir))

    assert runtime.compile_cache_usage(str(tmp_path)) == (1, 10)
    assert runtime.prune_compile_cache(str(tmp_path), max_bytes=100) == 0


def test_a_folder_that_cannot_be_read_is_empty(tmp_path, monkeypatch):
    def refuse(path):
        raise PermissionError(path)

    monkeypatch.setattr(runtime.os, "scandir", refuse)
    assert runtime.compile_cache_usage(str(tmp_path)) == (0, 0)
    assert runtime.prune_compile_cache(str(tmp_path), max_bytes=0) == 0
    assert runtime.clear_compile_cache(str(tmp_path)) == (0, 0)


def test_clearing_goes_on_past_a_file_in_use(tmp_path, monkeypatch):
    """On Windows a file JAX or another Blender has open can't be deleted."""
    for name in ("jit__solve-a-cache", "jit__solve-busy-cache", "jit__solve-c-cache"):
        (tmp_path / name).write_bytes(b"x")
    real_remove = os.remove

    def remove(path):
        if path.endswith("busy-cache"):
            raise PermissionError(path)
        real_remove(path)

    monkeypatch.setattr(runtime.os, "remove", remove)
    assert runtime.clear_compile_cache(str(tmp_path)) == (2, 1)
    assert os.listdir(tmp_path) == ["jit__solve-busy-cache"]
