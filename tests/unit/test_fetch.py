"""Git-free description fetching. Pure Python -- no Blender needed."""

from __future__ import annotations

import io
import json
import tarfile

import pytest

from ..conftest import load_addon_module

fetch = load_addon_module("catalog.fetch")


def _targz(entries: dict[str, bytes]) -> bytes:
    """Build a .tar.gz in memory from {member name: content}."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, payload in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def _stream_of(payload: bytes, chunk: int = 512):
    """A _ChunkReader over `payload`, as _open_stream would produce."""
    chunks = (payload[i : i + chunk] for i in range(0, len(payload), chunk))
    return fetch._ChunkReader(chunks, len(payload), fetch._hooks)


class TestTarballUrl:
    """Every catalog repository is on GitHub (82) or Codeberg (1)."""

    def test_github_uses_codeload(self):
        url = fetch.tarball_url(
            "https://github.com/ANYbotics/anymal_b_simple_description.git", "988b5df"
        )
        assert url == (
            "https://codeload.github.com/ANYbotics/"
            "anymal_b_simple_description/tar.gz/988b5df"
        )

    def test_github_without_dot_git_suffix(self):
        url = fetch.tarball_url("https://github.com/owner/repo", "abc")
        assert url == "https://codeload.github.com/owner/repo/tar.gz/abc"

    def test_codeberg_uses_forgejo_archive_route(self):
        url = fetch.tarball_url("https://codeberg.org/upkie/cookie_description.git", "c0ffee")
        assert url == "https://codeberg.org/upkie/cookie_description/archive/c0ffee.tar.gz"

    def test_nested_group_path_keeps_owner_and_repo(self):
        url = fetch.tarball_url("https://github.com/org/repo/extra", "sha")
        assert url == "https://codeload.github.com/org/repo/tar.gz/sha"

    @pytest.mark.parametrize("bad", ["https://github.com/", "https://github.com/lonely"])
    def test_unparseable_url_raises(self, bad):
        with pytest.raises(fetch.FetchError):
            fetch.tarball_url(bad, "sha")


class TestCacheRoot:
    def test_honours_environment_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ROBOT_DESCRIPTIONS_CACHE", str(tmp_path))
        assert fetch.cache_root() == tmp_path

    def test_defaults_to_robot_descriptions_cache(self, monkeypatch):
        monkeypatch.delenv("ROBOT_DESCRIPTIONS_CACHE", raising=False)
        assert fetch.cache_root().name == "robot_descriptions"


class TestIsComplete:
    """One predicate, so the offline gate and the fetcher cannot disagree.

    They used to: is_cached tested only that the directory existed, so a stale
    cache reported as present, the online-access check was skipped, and a
    download started anyway -- in offline mode.
    """

    def test_bare_directory_is_not_complete(self, tmp_path):
        (tmp_path / "repo").mkdir()
        assert not fetch.is_complete(tmp_path / "repo", "sha")

    def test_matching_stamp_is_complete(self, tmp_path):
        target = tmp_path / "repo"
        target.mkdir()
        (target / fetch.STAMP_NAME).write_text("sha", encoding="utf-8")
        assert fetch.is_complete(target, "sha")

    def test_wrong_commit_stamp_is_not_complete(self, tmp_path):
        target = tmp_path / "repo"
        target.mkdir()
        (target / fetch.STAMP_NAME).write_text("oldsha", encoding="utf-8")
        assert not fetch.is_complete(target, "newsha")

    def test_git_clone_is_complete(self, tmp_path):
        target = tmp_path / "repo"
        (target / ".git").mkdir(parents=True)
        assert fetch.is_complete(target, "anything")

    def test_full_checkout_satisfies_a_subtree_request(self, tmp_path):
        """The 1.6 GB someone already downloaded must not be re-fetched."""
        target = tmp_path / "repo"
        target.mkdir()
        (target / fetch.STAMP_NAME).write_text("sha", encoding="utf-8")
        assert fetch.is_complete(target, "sha", "unitree_go2")

    def test_sparse_manifest_hit(self, tmp_path):
        target = tmp_path / "repo"
        target.mkdir()
        (target / fetch.MANIFEST_NAME).write_text(
            json.dumps({"commit": "sha", "subtrees": ["unitree_go2"]}), encoding="utf-8"
        )
        assert fetch.is_complete(target, "sha", "unitree_go2")

    def test_sparse_manifest_miss(self, tmp_path):
        """A cache holding one robot must still fetch the next one."""
        target = tmp_path / "repo"
        target.mkdir()
        (target / fetch.MANIFEST_NAME).write_text(
            json.dumps({"commit": "sha", "subtrees": ["unitree_go2"]}), encoding="utf-8"
        )
        assert not fetch.is_complete(target, "sha", "unitree_g1")

    def test_sparse_manifest_at_a_stale_commit_is_not_complete(self, tmp_path):
        target = tmp_path / "repo"
        target.mkdir()
        (target / fetch.MANIFEST_NAME).write_text(
            json.dumps({"commit": "oldsha", "subtrees": ["unitree_go2"]}), encoding="utf-8"
        )
        assert not fetch.is_complete(target, "newsha", "unitree_go2")

    def test_sparse_cache_does_not_satisfy_a_whole_repo_request(self, tmp_path):
        target = tmp_path / "repo"
        target.mkdir()
        (target / fetch.MANIFEST_NAME).write_text(
            json.dumps({"commit": "sha", "subtrees": ["unitree_go2"]}), encoding="utf-8"
        )
        assert not fetch.is_complete(target, "sha", None)


class TestHooks:
    """Regression: callbacks used to stick at whatever the first caller passed.

    install_git_free_loader captured `progress` in a closure and returned early
    once patched, so the progress bar worked exactly once per Blender session.
    """

    def test_hooks_are_read_per_call_not_captured(self):
        seen = []
        with fetch.hooks(progress=lambda f, d, t: seen.append(("a", d))):
            fetch._hooks.report(1, 10)
        with fetch.hooks(progress=lambda f, d, t: seen.append(("b", d))):
            fetch._hooks.report(2, 10)
        assert seen == [("a", 1), ("b", 2)]

    def test_hooks_are_restored_on_exit(self):
        before = fetch._hooks
        with fetch.hooks(progress=lambda *a: None):
            assert fetch._hooks is not before
        assert fetch._hooks is before

    def test_hooks_are_restored_after_an_exception(self):
        before = fetch._hooks
        with pytest.raises(ValueError):
            with fetch.hooks(progress=lambda *a: None):
                raise ValueError("boom")
        assert fetch._hooks is before

    def test_report_sends_negative_fraction_when_total_is_unknown(self):
        """codeload may answer with chunked encoding and no Content-Length."""
        seen = []
        with fetch.hooks(progress=lambda f, d, t: seen.append(f)):
            fetch._hooks.report(100, 0)
        assert seen == [-1.0]

    def test_check_cancelled_raises(self):
        with fetch.hooks(should_cancel=lambda: True):
            with pytest.raises(fetch.FetchCancelled):
                fetch._hooks.check_cancelled()


class TestStreamExtract:
    """Unpacking straight from the response, one pass instead of three."""

    def test_single_root_archive_extracts(self, tmp_path, monkeypatch):
        payload = _targz({"repo-sha/README.md": b"hi", "repo-sha/urdf/a.urdf": b"<x/>"})
        monkeypatch.setattr(fetch, "_open_stream",
                            lambda url, hooks_: (_stream_of(payload), None))
        target = tmp_path / "repo"
        fetch._fetch_whole("https://github.com/o/r.git", "sha", target)

        assert (target / "README.md").read_bytes() == b"hi"
        assert (target / "urdf" / "a.urdf").read_bytes() == b"<x/>"
        assert (target / fetch.STAMP_NAME).read_text(encoding="utf-8") == "sha"

    def test_two_top_level_directories_are_rejected(self, tmp_path, monkeypatch):
        payload = _targz({"one/a": b"a", "two/b": b"b"})
        monkeypatch.setattr(fetch, "_open_stream",
                            lambda url, hooks_: (_stream_of(payload), None))
        with pytest.raises(fetch.FetchError, match="one top-level directory"):
            fetch._fetch_whole("https://github.com/o/r.git", "sha", tmp_path / "repo")

    def test_target_is_untouched_when_the_stream_fails(self, tmp_path, monkeypatch):
        target = tmp_path / "repo"
        target.mkdir()
        (target / "keep.txt").write_text("original", encoding="utf-8")

        def explode(url, hooks_):
            raise fetch.FetchError("network went away")

        monkeypatch.setattr(fetch, "_open_stream", explode)
        with pytest.raises(fetch.FetchError):
            fetch._fetch_whole("https://github.com/o/r.git", "sha", target)
        assert (target / "keep.txt").read_text(encoding="utf-8") == "original"

    def test_no_staging_directories_are_left_behind(self, tmp_path, monkeypatch):
        payload = _targz({"repo-sha/a": b"a"})
        monkeypatch.setattr(fetch, "_open_stream",
                            lambda url, hooks_: (_stream_of(payload), None))
        fetch._fetch_whole("https://github.com/o/r.git", "sha", tmp_path / "repo")
        assert not [p for p in tmp_path.iterdir() if p.name.startswith(".kinema-")]

    def test_cancel_propagates_out_of_the_reader(self, tmp_path, monkeypatch):
        payload = _targz({"repo-sha/a": b"a" * 4096})
        monkeypatch.setattr(fetch, "_open_stream",
                            lambda url, hooks_: (_stream_of(payload, chunk=16), None))
        with fetch.hooks(should_cancel=lambda: True):
            with pytest.raises(fetch.FetchCancelled):
                fetch._fetch_whole("https://github.com/o/r.git", "sha", tmp_path / "repo")


class TestChunkReader:
    def test_reports_bytes_as_they_arrive(self):
        seen = []
        with fetch.hooks(progress=lambda f, d, t: seen.append(d)):
            reader = fetch._ChunkReader(iter([b"a" * 10, b"b" * 10]), 20, fetch._hooks)
            assert reader.read(15) == b"a" * 10 + b"b" * 5
        assert seen[-1] == 20

    def test_short_read_at_end_of_stream(self):
        reader = fetch._ChunkReader(iter([b"abc"]), 3, fetch.FetchHooks())
        assert reader.read(100) == b"abc"
        assert reader.read(1) == b""


class TestSelectBlobs:
    def test_takes_the_subtree_and_top_level_files_only(self):
        blobs = [
            fetch._Blob("unitree_go2/go2.xml", 10),
            fetch._Blob("unitree_go2/assets/base.obj", 20),
            fetch._Blob("unitree_g1/g1.xml", 30),
            fetch._Blob("LICENSE", 5),
            fetch._Blob("README.md", 5),
        ]
        chosen = {b.path for b in fetch._select_blobs(blobs, "unitree_go2")}
        assert chosen == {
            "unitree_go2/go2.xml",
            "unitree_go2/assets/base.obj",
            "LICENSE",
            "README.md",
        }

    def test_prefix_does_not_match_a_sibling_with_the_same_start(self):
        blobs = [fetch._Blob("go2_extra/x", 1), fetch._Blob("go2/y", 1)]
        assert {b.path for b in fetch._select_blobs(blobs, "go2")} == {"go2/y"}


class TestFetchDescription:
    def test_reuses_a_completed_checkout(self, monkeypatch, tmp_path):
        """A matching stamp must short-circuit before any network access."""
        monkeypatch.setenv("ROBOT_DESCRIPTIONS_CACHE", str(tmp_path))
        target = tmp_path / "some_robot"
        target.mkdir()
        (target / fetch.STAMP_NAME).write_text("deadbeef", encoding="utf-8")

        def explode(*args, **kwargs):
            raise AssertionError("should not download when cache is valid")

        monkeypatch.setattr(fetch, "_fetch_whole", explode)
        monkeypatch.setattr(fetch, "_fetch_sparse", explode)
        assert fetch.fetch_description("https://github.com/o/r.git", "deadbeef",
                                       "some_robot") == str(target)

    def test_stale_stamp_triggers_refetch(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ROBOT_DESCRIPTIONS_CACHE", str(tmp_path))
        target = tmp_path / "some_robot"
        target.mkdir()
        (target / fetch.STAMP_NAME).write_text("oldsha", encoding="utf-8")

        called: list[str] = []
        monkeypatch.setattr(fetch, "_fetch_whole",
                            lambda url, commit, tgt: called.append(commit))
        fetch.fetch_description("https://github.com/o/r.git", "newsha", "some_robot")
        assert called == ["newsha"], "stale commit stamp did not trigger a refetch"

    def test_existing_git_clone_is_reused(self, monkeypatch, tmp_path):
        """Users who already ran robot_descriptions with git keep their cache."""
        monkeypatch.setenv("ROBOT_DESCRIPTIONS_CACHE", str(tmp_path))
        target = tmp_path / "some_robot"
        (target / ".git").mkdir(parents=True)

        monkeypatch.setattr(fetch, "_fetch_whole", lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("should not download over an existing clone")))
        assert fetch.fetch_description("https://github.com/o/r.git", "sha",
                                       "some_robot") == str(target)

    def test_sparse_fetch_is_tried_first_when_a_subtree_is_given(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("ROBOT_DESCRIPTIONS_CACHE", str(tmp_path))
        calls = []
        monkeypatch.setattr(fetch, "_fetch_sparse",
                            lambda url, commit, tgt, sub: calls.append(("sparse", sub)))
        monkeypatch.setattr(fetch, "_fetch_whole",
                            lambda *a: calls.append(("whole", None)))
        fetch.fetch_description("https://github.com/o/r.git", "sha", "repo",
                                subtree="unitree_go2")
        assert calls == [("sparse", "unitree_go2")]

    def test_sparse_failure_falls_back_to_the_whole_repository(
        self, monkeypatch, tmp_path
    ):
        """A rate-limited or unavailable tree API must not break the import."""
        monkeypatch.setenv("ROBOT_DESCRIPTIONS_CACHE", str(tmp_path))
        calls = []

        def sparse_fails(url, commit, tgt, sub):
            raise fetch.FetchError("tree API returned 403")

        monkeypatch.setattr(fetch, "_fetch_sparse", sparse_fails)
        monkeypatch.setattr(fetch, "_fetch_whole",
                            lambda *a: calls.append("whole"))
        fetch.fetch_description("https://github.com/o/r.git", "sha", "repo",
                                subtree="unitree_go2")
        assert calls == ["whole"]

    def test_cancelling_a_sparse_fetch_does_not_fall_back(self, monkeypatch, tmp_path):
        """Esc means stop, not "try the 1.6 GB version instead"."""
        monkeypatch.setenv("ROBOT_DESCRIPTIONS_CACHE", str(tmp_path))

        def cancelled(url, commit, tgt, sub):
            raise fetch.FetchCancelled("cancelled")

        monkeypatch.setattr(fetch, "_fetch_sparse", cancelled)
        monkeypatch.setattr(fetch, "_fetch_whole", lambda *a: (_ for _ in ()).throw(
            AssertionError("must not fall back after a cancel")))
        with pytest.raises(fetch.FetchCancelled):
            fetch.fetch_description("https://github.com/o/r.git", "sha", "repo",
                                    subtree="unitree_go2")

    def test_non_github_host_skips_the_sparse_path(self, monkeypatch, tmp_path):
        """The tree API is GitHub's; Codeberg gets the tarball."""
        monkeypatch.setenv("ROBOT_DESCRIPTIONS_CACHE", str(tmp_path))
        calls = []
        monkeypatch.setattr(fetch, "_fetch_sparse", lambda *a: calls.append("sparse"))
        monkeypatch.setattr(fetch, "_fetch_whole", lambda *a: calls.append("whole"))
        fetch.fetch_description("https://codeberg.org/o/r.git", "sha", "repo",
                                subtree="something")
        assert calls == ["whole"]


class TestPackageSubtree:
    """Deriving the one directory a robot needs, offline."""

    def setup_method(self):
        fetch._subtree_cache.clear()

    def test_menagerie_robot_resolves_to_its_own_directory(self):
        pytest.importorskip("robot_descriptions")
        assert fetch.package_subtree("go2_mj_description") == "unitree_go2"

    def test_probe_does_not_leave_the_stubbed_module_behind(self):
        """The probe's paths point at a sentinel; it must not survive."""
        import sys

        pytest.importorskip("robot_descriptions")
        fetch.package_subtree("go2_mj_description")
        assert "robot_descriptions.go2_mj_description" not in sys.modules

    def test_probe_restores_clone_to_cache(self):
        pytest.importorskip("robot_descriptions")
        from robot_descriptions import _cache

        before = _cache.clone_to_cache
        fetch.package_subtree("go2_mj_description")
        assert _cache.clone_to_cache is before

    def test_repository_outside_the_allowlist_returns_none(self):
        """Sparse fetching is opt-in per repository; see SPARSE_REPOSITORIES."""
        pytest.importorskip("robot_descriptions")
        assert fetch.package_subtree("ur5e_description") is None

    def test_unknown_description_returns_none(self):
        assert fetch.package_subtree("no_such_description") is None


class TestIsCached:
    """A description's cache directory is named after its *repository*.

    ur5e_description lives in Universal_Robots_ROS2_Description, and several
    descriptions share one repository. Looking for a directory named after the
    description key never matches, which made Blender demand online access even
    for robots already on disk.
    """

    def test_requires_a_stamp_not_merely_a_directory(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ROBOT_DESCRIPTIONS_CACHE", str(tmp_path))
        pytest.importorskip("robot_descriptions")
        from robot_descriptions._descriptions import DESCRIPTIONS
        from robot_descriptions._repositories import REPOSITORIES

        entry = DESCRIPTIONS["ur5e_description"]
        repo = REPOSITORIES[entry.repository]
        assert repo.cache_path != "ur5e_description", "test premise no longer holds"

        assert not fetch.is_cached("ur5e_description")

        # A bare directory is what a half-finished download leaves. Reporting it
        # as cached skipped the offline gate and then downloaded anyway.
        (tmp_path / repo.cache_path).mkdir(parents=True)
        assert not fetch.is_cached("ur5e_description")

        (tmp_path / repo.cache_path / fetch.STAMP_NAME).write_text(
            repo.commit, encoding="utf-8"
        )
        assert fetch.is_cached("ur5e_description")

    def test_sparse_cache_reports_cached_for_the_robot_it_holds(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("ROBOT_DESCRIPTIONS_CACHE", str(tmp_path))
        pytest.importorskip("robot_descriptions")
        from robot_descriptions._descriptions import DESCRIPTIONS
        from robot_descriptions._repositories import REPOSITORIES

        fetch._subtree_cache.clear()
        repo = REPOSITORIES[DESCRIPTIONS["go2_mj_description"].repository]
        target = tmp_path / repo.cache_path
        target.mkdir(parents=True)
        (target / fetch.MANIFEST_NAME).write_text(
            json.dumps({"commit": repo.commit, "subtrees": ["unitree_go2"]}),
            encoding="utf-8",
        )
        assert fetch.is_cached("go2_mj_description")

    def test_unknown_description_is_not_cached(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ROBOT_DESCRIPTIONS_CACHE", str(tmp_path))
        assert not fetch.is_cached("no_such_description")
