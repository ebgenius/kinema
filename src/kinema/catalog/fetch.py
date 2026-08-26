"""Fetch robot descriptions without requiring a git executable.

``robot_descriptions`` downloads model repositories with GitPython, which shells
out to a real ``git`` binary. That assumption is fine for roboticists and wrong
for Kinema's audience: a Blender artist on Windows very often has no git
installed, and "install git, add it to PATH, restart Blender" is not an
acceptable first-run experience for an add-on.

Rather than fork the catalog, Kinema replaces one function.
``robot_descriptions._cache.clone_to_cache`` is the single choke point every
description module calls at import time::

    REPOSITORY_PATH: str = _clone_to_cache("Universal_Robots_ROS2_Description", ...)

The catalog's ``REPOSITORIES`` table already carries everything needed to fetch
without git -- repository URL, exact pinned commit, and cache directory name --
so the replacement downloads a source tarball over HTTPS and unpacks it.

This cannot be done by pre-populating the normal cache: ``clone_to_directory``
calls ``Repo(target_dir)`` on any existing directory and, on
``InvalidGitRepositoryError``, deletes it and re-clones. A plain extracted
tree would be wiped on next use. The function has to be replaced outright.

Of the 83 repositories in the catalog, 82 are on GitHub and one is on Codeberg;
both serve commit tarballs from stable URLs.
"""

from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

#: Written into a completed cache directory so a partial download is never
#: mistaken for a usable checkout.
STAMP_NAME = ".kinema-fetch-complete"


class FetchError(RuntimeError):
    """A robot description could not be downloaded."""


def cache_root() -> Path:
    """Where descriptions are unpacked.

    Honours ``ROBOT_DESCRIPTIONS_CACHE`` so Kinema shares a cache with any
    existing robot_descriptions install rather than downloading twice.
    """
    return Path(
        os.path.expanduser(
            os.environ.get("ROBOT_DESCRIPTIONS_CACHE", "~/.cache/robot_descriptions")
        )
    )


def tarball_url(repo_url: str, commit: str) -> str:
    """Map a git clone URL plus commit to a source tarball URL."""
    parsed = urlparse(repo_url)
    host = parsed.netloc.lower()
    parts = parsed.path.removesuffix(".git").strip("/").split("/")
    if len(parts) < 2:
        raise FetchError(f"cannot parse repository URL: {repo_url}")
    owner, repo = parts[0], parts[1]

    if host.endswith("github.com"):
        return f"https://codeload.github.com/{owner}/{repo}/tar.gz/{commit}"
    # Codeberg runs Forgejo; Gitea/Forgejo share this archive route.
    return f"https://{parsed.netloc}/{owner}/{repo}/archive/{commit}.tar.gz"


def _download(url: str, dest: Path, progress: Callable[[float], None] | None) -> None:
    """Stream a URL to disk. Uses requests (Blender ships it) or urllib."""
    try:
        import requests

        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            with open(dest, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 16):
                    handle.write(chunk)
                    done += len(chunk)
                    if progress and total:
                        progress(done / total)
    except ImportError:
        import urllib.request

        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
            dest.write_bytes(response.read())


def _safe_extract(archive: Path, into: Path) -> Path:
    """Extract a tarball and return its single top-level directory.

    Uses the ``data`` filter so member paths cannot escape the destination
    (CVE-2007-4559); Python 3.13 defaults to this, but naming it keeps the
    intent explicit for anyone auditing the bundled code.
    """
    with tarfile.open(archive, "r:gz") as tar:
        roots = {Path(m.name).parts[0] for m in tar.getmembers() if m.name.strip("./")}
        if len(roots) != 1:
            raise FetchError(f"expected one top-level directory, found {sorted(roots)}")
        tar.extractall(into, filter="data")
    return into / roots.pop()


def fetch_description(
    repo_url: str,
    commit: str,
    cache_path: str,
    *,
    progress: Callable[[float], None] | None = None,
    force: bool = False,
) -> str:
    """Download and unpack one description; return its local directory.

    Idempotent: an already-complete checkout at the same commit is reused.
    """
    target = cache_root() / cache_path
    stamp = target / STAMP_NAME

    if not force and stamp.is_file() and stamp.read_text(encoding="utf-8").strip() == commit:
        return str(target)
    # A git clone from a previous robot_descriptions run is equally usable.
    if not force and (target / ".git").is_dir():
        return str(target)

    url = tarball_url(repo_url, commit)
    target.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="kinema-fetch-") as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "source.tar.gz"
        try:
            _download(url, archive, progress)
        except Exception as exc:  # noqa: BLE001
            raise FetchError(f"could not download {url}: {exc}") from exc

        extracted = _safe_extract(archive, tmp_path / "unpacked")

        # Replace atomically-ish: never leave a half-written cache behind.
        staging = target.with_name(target.name + ".incoming")
        shutil.rmtree(staging, ignore_errors=True)
        shutil.move(str(extracted), str(staging))
        (staging / STAMP_NAME).write_text(commit, encoding="utf-8")
        shutil.rmtree(target, ignore_errors=True)
        staging.rename(target)

    return str(target)


def install_git_free_loader(progress: Callable[[float], None] | None = None) -> bool:
    """Patch ``robot_descriptions`` to fetch over HTTPS instead of git.

    Must run *before* importing any description module, because those call
    ``clone_to_cache`` at module import time.

    Returns True if the patch was applied. Idempotent.
    """
    # GitPython raises ImportError at *import* time when it cannot find a git
    # executable ("Bad git executable"), so importing robot_descriptions at all
    # fails on a machine without git -- the exact case this function exists to
    # handle. "quiet" downgrades that to a warning; we never call git anyway.
    os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

    try:
        from robot_descriptions import _cache
        from robot_descriptions._repositories import REPOSITORIES
    except ImportError:
        return False

    if getattr(_cache, "_kinema_patched", False):
        return True

    original = _cache.clone_to_cache

    def clone_to_cache(description_name: str, commit: str | None = None) -> str:
        repository = REPOSITORIES.get(description_name)
        if repository is None:
            # Unknown key: let the original raise its own clear error.
            return original(description_name, commit=commit)
        try:
            return fetch_description(
                repository.url,
                commit or repository.commit,
                repository.cache_path,
                progress=progress,
            )
        except FetchError:
            # A real git install may still succeed (private mirrors, proxies).
            if shutil.which("git"):
                return original(description_name, commit=commit)
            raise

    _cache.clone_to_cache = clone_to_cache
    _cache._kinema_patched = True
    return True
