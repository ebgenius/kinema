"""Git-free description fetching. Pure Python -- no Blender needed."""

from __future__ import annotations

import pytest

from ..conftest import load_addon_module

fetch = load_addon_module("catalog.fetch")


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


class TestFetchDescription:
    def test_reuses_a_completed_checkout(self, monkeypatch, tmp_path):
        """A matching stamp must short-circuit before any network access."""
        monkeypatch.setenv("ROBOT_DESCRIPTIONS_CACHE", str(tmp_path))
        target = tmp_path / "some_robot"
        target.mkdir()
        (target / fetch.STAMP_NAME).write_text("deadbeef", encoding="utf-8")

        def explode(*args, **kwargs):
            raise AssertionError("should not download when cache is valid")

        monkeypatch.setattr(fetch, "_download", explode)
        assert fetch.fetch_description("https://github.com/o/r.git", "deadbeef",
                                       "some_robot") == str(target)

    def test_stale_stamp_triggers_refetch(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ROBOT_DESCRIPTIONS_CACHE", str(tmp_path))
        target = tmp_path / "some_robot"
        target.mkdir()
        (target / fetch.STAMP_NAME).write_text("oldsha", encoding="utf-8")

        called: list[str] = []
        monkeypatch.setattr(fetch, "_download",
                            lambda url, dest, progress: called.append(url))
        # _download is a no-op here, so extraction fails -- we only assert that
        # a stale stamp gets as far as attempting the download.
        with pytest.raises(Exception):
            fetch.fetch_description("https://github.com/o/r.git", "newsha", "some_robot")
        assert called, "stale commit stamp did not trigger a refetch"

    def test_existing_git_clone_is_reused(self, monkeypatch, tmp_path):
        """Users who already ran robot_descriptions with git keep their cache."""
        monkeypatch.setenv("ROBOT_DESCRIPTIONS_CACHE", str(tmp_path))
        target = tmp_path / "some_robot"
        (target / ".git").mkdir(parents=True)

        monkeypatch.setattr(fetch, "_download", lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("should not download over an existing clone")))
        assert fetch.fetch_description("https://github.com/o/r.git", "sha",
                                       "some_robot") == str(target)
