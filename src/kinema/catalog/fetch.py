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

Sparse fetching
---------------
One repository dominates: ``mujoco_menagerie`` is 1.64 GB across 2466 files and
backs **49 of the 186** catalog robots. Downloading all of it to obtain one
robot -- ``unitree_go2`` is 29.4 MB -- is a 56x overfetch, and on a laptop it is
minutes of waiting plus a gigabyte of disk that never gets used.

It is avoidable. Description modules do no filesystem access at import time,
only ``os.path.join``::

    PACKAGE_PATH: str = _path.join(REPOSITORY_PATH, "unitree_go2")

so the directory a description needs is derivable offline (:func:`package_subtree`),
and GitHub's tree API plus ``raw.githubusercontent.com`` can fetch just that.

Crucially the **cache layout does not change**: the repository-named directory
stays, and only the one subdirectory inside it is populated. Every path a
description module computes still resolves, an existing full checkout is still
honoured, and importing a second robot from the same repository adds its
directory rather than re-downloading.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

#: Written into a completed cache directory so a partial download is never
#: mistaken for a usable checkout. Plain text holding the commit; its presence
#: means the *whole* repository is present.
STAMP_NAME = ".kinema-fetch-complete"

#: Records which subdirectories of a repository have been fetched, for caches
#: populated sparsely. ``{"commit": "<sha>", "subtrees": ["unitree_go2", ...]}``
MANIFEST_NAME = ".kinema-cache.json"

#: Repositories whose per-robot directories are known to be self-contained, so
#: fetching one directory yields a working model.
#:
#: This is an allowlist rather than a general rule because a subtree that
#: references shared assets outside itself would fetch cleanly and then fail at
#: mesh-load time with nothing but a warning -- a silently incomplete robot is
#: worse than a slow download. Menagerie is verified: no XML under any robot
#: directory refers to a parent path, and each carries its own ``assets/``.
#:
#: Widening this list means checking the same thing for the candidate
#: repository; :func:`package_subtree` already resolves a subtree for 143 of the
#: 186 descriptions, so the mechanism is there when a repository earns it.
SPARSE_REPOSITORIES = frozenset({"mujoco_menagerie"})

#: Parallel blob downloads for a sparse fetch. These files are small and the
#: cost is almost entirely round-trip latency, so a handful of workers helps a
#: lot and more stops helping.
_SPARSE_WORKERS = 8

_CHUNK = 1 << 16


class FetchError(RuntimeError):
    """A robot description could not be downloaded."""


class FetchCancelled(FetchError):
    """The user cancelled an in-progress download."""


# --------------------------------------------------------------------------
# hooks
# --------------------------------------------------------------------------
@dataclass
class FetchHooks:
    """Callbacks and context for one import job.

    Held in a module global rather than threaded through as parameters because
    the call actually doing the work is ``clone_to_cache``, which Kinema does
    not call -- a description module calls it, at import time, several frames
    below anything Kinema wrote. A parameter has nowhere to enter.

    ``progress`` is invoked from the download thread and must not touch ``bpy``.
    """

    progress: Callable[[float, int, int], None] | None = None  # fraction, done, total
    should_cancel: Callable[[], bool] | None = None
    #: Repository subdirectory to fetch, resolved by the caller before the
    #: description module is imported (see :func:`package_subtree`).
    subtree: str | None = None
    #: Human-readable label for the thing being downloaded, for status text.
    label: str = ""

    def report(self, done: int, total: int) -> None:
        if self.progress is not None:
            self.progress((done / total) if total else -1.0, done, total)

    def check_cancelled(self) -> None:
        if self.should_cancel is not None and self.should_cancel():
            raise FetchCancelled("download cancelled")


_hooks = FetchHooks()


@contextlib.contextmanager
def hooks(
    *,
    progress: Callable[[float, int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    subtree: str | None = None,
    label: str = "",
):
    """Install fetch callbacks for the duration of one import job.

    Restores the previous hooks on exit, so a nested or failed job cannot leave
    a stale callback installed -- which is exactly the bug the earlier
    closure-captured ``progress`` argument had: ``install_git_free_loader``
    returns early once patched, so every import after the first kept reporting
    to the *first* caller's callback.
    """
    global _hooks
    previous = _hooks
    _hooks = FetchHooks(
        progress=progress, should_cancel=should_cancel, subtree=subtree, label=label
    )
    try:
        yield _hooks
    finally:
        _hooks = previous


# --------------------------------------------------------------------------
# cache location and completeness
# --------------------------------------------------------------------------
def cache_root() -> Path:
    """Where descriptions are unpacked.

    Honours ``ROBOT_DESCRIPTIONS_CACHE`` so Kinema shares a cache with any
    existing robot_descriptions install rather than downloading twice. Kinema's
    own preference is projected onto that variable by ``prefs.apply_cache_dir``,
    which keeps this module free of ``bpy`` -- it runs on the download thread,
    where Blender's API is off limits.
    """
    return Path(
        os.path.expanduser(
            os.environ.get("ROBOT_DESCRIPTIONS_CACHE", "~/.cache/robot_descriptions")
        )
    )


def read_manifest(target: Path) -> dict:
    """The sparse-cache manifest for a repository directory, or ``{}``."""
    try:
        return json.loads((target / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_manifest(target: Path, commit: str, subtrees: set[str]) -> None:
    (target / MANIFEST_NAME).write_text(
        json.dumps({"commit": commit, "subtrees": sorted(subtrees)}, indent=1),
        encoding="utf-8",
    )


def is_complete(target: Path, commit: str, subtree: str | None = None) -> bool:
    """The single "usable without the network" test.

    Shared by :func:`is_cached` and :func:`fetch_description` so the offline
    gate and the fetcher can never disagree. They used to: ``is_cached`` tested
    only that the directory existed, so a stale or half-written cache reported
    as present, the "enable online access" check was skipped, and a download
    started anyway -- in offline mode.

    Order matters. A full checkout satisfies any subtree request, which is what
    keeps a cache downloaded before sparse fetching existed (or by
    robot_descriptions itself) valid rather than re-fetching it.
    """
    if (target / ".git").is_dir():
        return True  # a real clone from robot_descriptions; always complete

    stamp = target / STAMP_NAME
    try:
        if stamp.read_text(encoding="utf-8").strip() == commit:
            return True  # whole repository at the pinned commit
    except OSError:
        pass

    if subtree is None:
        return False
    manifest = read_manifest(target)
    return manifest.get("commit") == commit and subtree in manifest.get("subtrees", ())


def is_cached(description_key: str) -> bool:
    """True if a description is already on disk, so no network is needed.

    The cache directory is named after the *repository*, not the description --
    ``ur5e_description`` lives in ``Universal_Robots_ROS2_Description``, and
    several descriptions often share one repository. Checking for a directory
    named after the description key therefore never matches, which made Blender
    demand online access even for robots already downloaded.
    """
    os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
    try:
        from robot_descriptions._descriptions import DESCRIPTIONS
        from robot_descriptions._repositories import REPOSITORIES
    except ImportError:
        return False

    entry = DESCRIPTIONS.get(description_key)
    repository = REPOSITORIES.get(getattr(entry, "repository", None)) if entry else None
    if repository is None:
        return False
    return is_complete(
        cache_root() / repository.cache_path,
        repository.commit,
        package_subtree(description_key),
    )


# --------------------------------------------------------------------------
# subtree resolution
# --------------------------------------------------------------------------
#: Guards against re-entering the probe import from inside itself.
_probing = threading.local()

_subtree_cache: dict[str, str | None] = {}

#: Stand-in for the repository root during a probe import. Never touches disk.
_SENTINEL_ROOT = "__KINEMA_REPO_ROOT__"


def _relative_package(repository_path: str, package_path: str) -> str | None:
    """``PACKAGE_PATH`` relative to ``REPOSITORY_PATH``, or None if they match."""
    try:
        relative = os.path.relpath(package_path, repository_path)
    except ValueError:
        return None
    if relative in (".", "", os.curdir):
        return None
    if relative.startswith(os.pardir):
        return None  # escapes the repository; treat as "needs everything"
    return PurePosixPath(*Path(relative).parts).as_posix()


def package_subtree(description_key: str) -> str | None:
    """Repository subdirectory a description needs, or None for the whole repo.

    Imports the description module with ``clone_to_cache`` stubbed to a sentinel
    and reads back ``PACKAGE_PATH``. That is safe and entirely offline for the
    ordinary case, because these modules only call ``os.path.join`` at import
    time -- but not universally: ``eve_r3_description`` parses and rewrites its
    URDF at import to fix invalid joint limits. Any such module raises here, and
    a failed probe correctly means "fetch the whole repository".

    Returns None for repositories not in :data:`SPARSE_REPOSITORIES`, so callers
    can use this as the single "should this be sparse?" question.
    """
    if description_key in _subtree_cache:
        return _subtree_cache[description_key]
    if getattr(_probing, "active", False):
        return None  # re-entered from inside a probe import

    os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
    try:
        from robot_descriptions._descriptions import DESCRIPTIONS
    except ImportError:
        return None

    entry = DESCRIPTIONS.get(description_key)
    if entry is None or getattr(entry, "repository", None) not in SPARSE_REPOSITORIES:
        _subtree_cache[description_key] = None
        return None

    module_name = f"robot_descriptions.{description_key}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        # Already imported for real: read the live paths rather than re-probing,
        # and above all do not evict it -- its REPOSITORY_PATH is the real one.
        subtree = _relative_package(
            getattr(existing, "REPOSITORY_PATH", ""),
            getattr(existing, "PACKAGE_PATH", ""),
        )
        _subtree_cache[description_key] = subtree
        return subtree

    from robot_descriptions import _cache

    original = _cache.clone_to_cache
    _cache.clone_to_cache = lambda name, commit=None: _SENTINEL_ROOT
    _probing.active = True
    try:
        module = importlib.import_module(module_name)
        subtree = _relative_package(
            _SENTINEL_ROOT, getattr(module, "PACKAGE_PATH", _SENTINEL_ROOT)
        )
    except Exception:  # noqa: BLE001 - any failure means "fetch everything"
        subtree = None
    finally:
        _probing.active = False
        _cache.clone_to_cache = original
        # The probe left a module whose paths point at the sentinel. It must not
        # survive, or the real import would be skipped and MJCF_PATH would be
        # nonsense.
        sys.modules.pop(module_name, None)

    _subtree_cache[description_key] = subtree
    return subtree


# --------------------------------------------------------------------------
# URLs
# --------------------------------------------------------------------------
def _owner_repo(repo_url: str) -> tuple[str, str, str]:
    """(host, owner, repo) from a git clone URL."""
    parsed = urlparse(repo_url)
    parts = parsed.path.removesuffix(".git").strip("/").split("/")
    if len(parts) < 2:
        raise FetchError(f"cannot parse repository URL: {repo_url}")
    return parsed.netloc, parts[0], parts[1]


def tarball_url(repo_url: str, commit: str) -> str:
    """Map a git clone URL plus commit to a source tarball URL."""
    netloc, owner, repo = _owner_repo(repo_url)
    if netloc.lower().endswith("github.com"):
        return f"https://codeload.github.com/{owner}/{repo}/tar.gz/{commit}"
    # Codeberg runs Forgejo; Gitea/Forgejo share this archive route.
    return f"https://{netloc}/{owner}/{repo}/archive/{commit}.tar.gz"


def _is_github(repo_url: str) -> bool:
    return urlparse(repo_url).netloc.lower().endswith("github.com")


# --------------------------------------------------------------------------
# sparse fetch (GitHub tree API + raw CDN)
# --------------------------------------------------------------------------
@dataclass
class _Blob:
    path: str
    size: int


def _requests():
    """``requests``, or a FetchError so the caller falls back to the tarball.

    Blender ships requests, which is why it is not among the bundled wheels.
    The tarball path can still manage on urllib alone; the sparse path cannot,
    so a missing requests has to read as "sparse is unavailable" rather than
    escaping as a ModuleNotFoundError past the fallback in fetch_description.
    """
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - requests ships with Blender
        raise FetchError("requests is unavailable; cannot fetch sparsely") from exc
    return requests


def _tree_blobs(owner: str, repo: str, commit: str) -> list[_Blob]:
    """Every file in a repository at one commit, with its size.

    One unauthenticated API call. GitHub's unauthenticated limit is 60/hour per
    IP and this costs one per repository per import, which is not a constraint
    in practice -- but any failure here simply falls back to the tarball, so it
    never becomes one.
    """
    requests = _requests()

    url = (
        f"https://api.github.com/repos/{owner}/{repo}/git/trees/{commit}"
        "?recursive=1"
    )
    response = requests.get(
        url, timeout=30, headers={"Accept": "application/vnd.github+json"}
    )
    if response.status_code != 200:
        raise FetchError(f"tree API returned {response.status_code} for {owner}/{repo}")
    payload = response.json()
    if payload.get("truncated"):
        raise FetchError(f"{owner}/{repo} tree is truncated; cannot fetch sparsely")
    return [
        _Blob(entry["path"], int(entry.get("size") or 0))
        for entry in payload.get("tree", ())
        if entry.get("type") == "blob"
    ]


def _select_blobs(
    blobs: list[_Blob], subtree: str
) -> tuple[list[_Blob], list[_Blob]]:
    """Split into (files under ``subtree``, the repository's top-level files).

    Returned separately rather than concatenated so the caller can tell "this
    subtree does not exist" from "this subtree is empty apart from the repo's
    own furniture". Merged, the two are indistinguishable: every repository has
    a LICENSE and a README, so a non-empty result proved nothing about the
    subtree and the caller's emptiness check could never fire.

    The top-level files are a few hundred KB and referenced by no model, but
    their absence makes a cache directory look corrupt to anyone who opens it.
    """
    prefix = subtree.rstrip("/") + "/"
    matched = [blob for blob in blobs if blob.path.startswith(prefix)]
    top_level = [blob for blob in blobs if "/" not in blob.path]
    return matched, top_level


#: One requests.Session per worker thread. A Session is not documented as
#: thread-safe, but a bare ``requests.get`` per file would build a fresh
#: connection pool and repeat the TLS handshake for every one of ~34 files on
#: the same host. Per-thread sessions keep the connection reuse without sharing
#: one object across threads. Freed when the pool's threads exit.
_sessions = threading.local()


def _session():
    session = getattr(_sessions, "session", None)
    if session is None:
        session = _sessions.session = _requests().Session()
    return session


def _fetch_blob(
    owner: str, repo: str, commit: str, blob: _Blob, into: Path, hooks_: FetchHooks
) -> int:
    # Checked here, not only in the collecting loop: a cancelled job must stop
    # workers from starting new downloads, not merely stop reporting them.
    hooks_.check_cancelled()

    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{commit}/{blob.path}"
    destination = into / Path(*PurePosixPath(blob.path).parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = _session().get(url, timeout=60)
    if response.status_code != 200:
        raise FetchError(f"could not download {blob.path}: HTTP {response.status_code}")
    destination.write_bytes(response.content)
    return len(response.content)


def _fetch_sparse(repo_url: str, commit: str, target: Path, subtree: str) -> None:
    """Materialise one subdirectory of a repository into ``target``.

    Atomicity is per-subtree, not per-repository: adding a second robot must not
    disturb the first, so this stages the new directory and moves it into place
    rather than replacing ``target`` wholesale.
    """
    _, owner, repo = _owner_repo(repo_url)
    matched, top_level = _select_blobs(_tree_blobs(owner, repo, commit), subtree)
    if not matched:
        # Fail before spending any requests. Downloading the top-level files
        # first and only discovering the problem at the move below would waste
        # a round of requests and report it as a FileNotFoundError.
        raise FetchError(f"'{subtree}' matched no files in {owner}/{repo}")
    blobs = matched + top_level

    total = sum(blob.size for blob in blobs)
    done = 0
    hooks_ = _hooks
    hooks_.report(0, total)

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=target.parent, prefix=".kinema-sparse-"))
    try:
        with ThreadPoolExecutor(max_workers=_SPARSE_WORKERS) as pool:
            futures = [
                pool.submit(_fetch_blob, owner, repo, commit, blob, staging, hooks_)
                for blob in blobs
            ]
            try:
                # as_completed, not submission order: waiting on futures in the
                # order they were queued blocks on the first one even when the
                # other thirty have already landed, so progress arrives in
                # lumps and a cancel sits behind the slowest early file.
                for future in as_completed(futures):
                    done += future.result()
                    hooks_.report(done, total)
            except BaseException:
                # Drop whatever has not started. Leaving this to the `with`
                # block would call shutdown(wait=True), which runs the entire
                # remaining queue before the exception surfaces -- so Esc looked
                # like a hang until the whole download finished anyway.
                pool.shutdown(wait=False, cancel_futures=True)
                raise

        target.mkdir(parents=True, exist_ok=True)
        # Move the subtree into place, then the top-level files beside it.
        staged_subtree = staging / Path(*PurePosixPath(subtree).parts)
        final = target / Path(*PurePosixPath(subtree).parts)
        final.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(final, ignore_errors=True)
        shutil.move(str(staged_subtree), str(final))
        for item in staging.iterdir():
            if item.is_file():
                shutil.move(str(item), str(target / item.name))

        manifest = read_manifest(target)
        known = set(manifest.get("subtrees", ())) if manifest.get("commit") == commit else set()
        _write_manifest(target, commit, known | {subtree})
    finally:
        shutil.rmtree(staging, ignore_errors=True)


# --------------------------------------------------------------------------
# whole-repository fetch (streaming tarball)
# --------------------------------------------------------------------------
class _ChunkReader:
    """Adapts an iterator of byte chunks to a ``read()``-able stream.

    Lets ``tarfile`` unpack straight from the HTTP response instead of writing a
    1.6 GB archive to disk and reading it back -- one pass rather than three,
    and roughly half the peak disk. Also the only place that can notice a cancel
    mid-download, so a big fetch aborts on the next 64 KB boundary rather than
    at the end.
    """

    def __init__(self, chunks: Iterator[bytes], total: int, hooks_: FetchHooks) -> None:
        self._chunks = chunks
        self._buffer = bytearray()
        self._total = total
        self._done = 0
        self._hooks = hooks_

    def read(self, size: int = -1) -> bytes:
        # One loop for both the sized and read-everything cases. Draining the
        # iterator separately for size < 0 meant that path neither reported
        # bytes nor honoured a cancel, and kept a second copy of the accounting
        # that could drift from this one.
        read_all = size is None or size < 0
        while read_all or len(self._buffer) < size:
            self._hooks.check_cancelled()
            try:
                chunk = next(self._chunks)
            except StopIteration:
                break
            self._buffer.extend(chunk)
            self._done += len(chunk)
            self._hooks.report(self._done, self._total)

        if read_all:
            size = len(self._buffer)
        taken = bytes(self._buffer[:size])
        del self._buffer[: len(taken)]
        return taken


def _single_root(tar: tarfile.TarFile, seen: list[str]) -> Iterator[tarfile.TarInfo]:
    """Yield members, enforcing that they share one top-level directory.

    The streaming mode ``r|gz`` cannot call ``getmembers()`` -- it never seeks --
    so the invariant the old code checked up front becomes a per-member check.
    """
    for member in tar:
        parts = PurePosixPath(member.name).parts
        if not parts or parts[0] in (".", "/"):
            continue
        if not seen:
            seen.append(parts[0])
        elif parts[0] != seen[0]:
            raise FetchError(
                f"expected one top-level directory, found {seen[0]!r} and {parts[0]!r}"
            )
        yield member


def _open_stream(url: str, hooks_: FetchHooks):
    """A file-like object over the archive at ``url``, and the thing to close."""
    try:
        import requests
    except ImportError:
        import urllib.request

        # urllib's response is already file-like, so hand it straight to
        # tarfile. Reading it whole would mean a 1.6 GB bytes object in RAM.
        response = urllib.request.urlopen(url, timeout=60)  # noqa: S310
        return response, response

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("Content-Length") or 0)
    # iter_content, not response.raw: some hosts set Content-Encoding: gzip on a
    # .tar.gz, and raw would hand tarfile doubly-compressed bytes.
    reader = _ChunkReader(response.iter_content(chunk_size=_CHUNK), total, hooks_)
    return reader, response


def _fetch_whole(repo_url: str, commit: str, target: Path) -> None:
    """Download and unpack an entire repository at one commit."""
    url = tarball_url(repo_url, commit)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Inside the cache root, not the system temp directory: the final move is
    # then a rename rather than a full copy across volumes, which matters now
    # that the cache location is a preference and may well be another drive.
    staging_root = Path(tempfile.mkdtemp(dir=target.parent, prefix=".kinema-fetch-"))
    try:
        unpacked = staging_root / "unpacked"
        stream, response = _open_stream(url, _hooks)
        seen: list[str] = []
        try:
            with tarfile.open(fileobj=stream, mode="r|gz") as tar:
                # filter="data" keeps member paths from escaping the destination
                # (CVE-2007-4559); Python 3.13 defaults to it, but naming it
                # keeps the intent explicit for anyone auditing the bundled code.
                tar.extractall(unpacked, members=_single_root(tar, seen), filter="data")
        except FetchCancelled:
            raise
        except FetchError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise FetchError(f"could not download {url}: {exc}") from exc
        finally:
            if response is not None:
                response.close()

        if not seen:
            raise FetchError(f"archive at {url} was empty")
        extracted = unpacked / seen[0]
        (extracted / STAMP_NAME).write_text(commit, encoding="utf-8")

        staging = target.with_name(target.name + ".incoming")
        shutil.rmtree(staging, ignore_errors=True)
        shutil.move(str(extracted), str(staging))
        shutil.rmtree(target, ignore_errors=True)
        staging.rename(target)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def fetch_description(
    repo_url: str,
    commit: str,
    cache_path: str,
    *,
    subtree: str | None = None,
    force: bool = False,
) -> str:
    """Download and unpack one description; return its local directory.

    Idempotent: an already-complete checkout at the same commit is reused, and
    so is one populated sparsely that already holds ``subtree``.
    """
    target = cache_root() / cache_path
    if not force and is_complete(target, commit, subtree):
        return str(target)

    if subtree and _is_github(repo_url):
        try:
            _fetch_sparse(repo_url, commit, target, subtree)
            return str(target)
        except FetchCancelled:
            raise
        except (FetchError, OSError) as exc:
            # The tree API is unavailable, rate-limited, or the layout is not
            # what we expected. The whole repository always works -- but say so,
            # loudly. Silence here means a robot that should have cost 30 MB
            # quietly costs 1.6 GB instead, with nothing to explain the wait.
            print(
                f"Kinema: sparse fetch of '{subtree}' from {cache_path} failed "
                f"({exc}); falling back to the whole repository, which is much "
                f"larger."
            )

    _fetch_whole(repo_url, commit, target)
    return str(target)


def install_git_free_loader() -> bool:
    """Patch ``robot_descriptions`` to fetch over HTTPS instead of git.

    Must run *before* importing any description module, because those call
    ``clone_to_cache`` at module import time.

    Returns True if the patch was applied. Idempotent -- and unlike the earlier
    version it takes no callbacks, because the early return below made them
    stick at whatever the first caller passed. Callbacks live in :data:`_hooks`
    and are read on every call instead; see :func:`hooks`.
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
                subtree=_hooks.subtree,
            )
        except FetchCancelled:
            raise
        except FetchError:
            # A real git install may still succeed (private mirrors, proxies).
            if shutil.which("git"):
                return original(description_name, commit=commit)
            raise

    _cache.clone_to_cache = clone_to_cache
    _cache._kinema_patched = True
    return True
