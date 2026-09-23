"""The bundled wheel payload, checked from the committed manifest alone.

The wheels themselves are gitignored -- only ``blender_manifest.toml`` is
committed -- so this is the one place a bad payload can be caught without a
network round trip. It has to be: a mismatched payload installs perfectly and
only fails at the first ``import jax``, in front of the user.
"""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "src" / "kinema" / "blender_manifest.toml"
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _load_fetch_wheels():
    """Load ``tools/fetch_wheels.py`` by path (``tools`` is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "kinema_fetch_wheels", REPO_ROOT / "tools" / "fetch_wheels.py"
    )
    if spec is None or spec.loader is None:
        pytest.skip("tools/fetch_wheels.py not available")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wheel_tags(name: str) -> tuple[str, str, str, str]:
    """``(dist, python_tag, abi_tag, platform_tag)`` from a wheel filename.

    PEP 427 puts an *optional* build tag between the version and the python
    tag, so a wheel name has five segments or six. The trailing three are the
    only reliable anchor: unpacking into a fixed five raises ValueError on a
    perfectly legal ``pkg-1.0.0-1-py3-none-any.whl``.
    """
    parts = name[: -len(".whl")].split("-")
    python_tag, abi_tag, platform_tag = parts[-3:]
    return parts[0], python_tag, abi_tag, platform_tag


@pytest.mark.parametrize(
    "name, expected",
    [
        ("jax-0.11.1-py3-none-any.whl", ("jax", "py3", "none", "any")),
        ("six-1.17.0-py2.py3-none-any.whl", ("six", "py2.py3", "none", "any")),
        (
            "jaxlib-0.11.1-cp313-cp313-manylinux_2_27_x86_64.whl",
            ("jaxlib", "cp313", "cp313", "manylinux_2_27_x86_64"),
        ),
        # The build tag none of our dependencies uses today.
        ("pkg-1.0.0-1-py3-none-any.whl", ("pkg", "py3", "none", "any")),
    ],
)
def test_wheel_name_parsing_tolerates_a_build_tag(name, expected):
    assert wheel_tags(name) == expected


@pytest.fixture(scope="module")
def manifest() -> dict:
    return tomllib.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def wheel_names(manifest) -> list[str]:
    names = [Path(entry).name for entry in manifest["wheels"]]
    assert names, "manifest declares no wheels"
    return names


def test_version_matches_pyproject(manifest):
    """The two version fields are maintained by hand and must not drift.

    The manifest's version is the one that ships, and it is what the extensions
    platform keys a release on -- a release cut from a pyproject bump that never
    reached the manifest would publish under the old number, silently.
    """
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    assert manifest["version"] == pyproject["project"]["version"], (
        f"blender_manifest.toml is {manifest['version']} but pyproject.toml is "
        f"{pyproject['project']['version']}; bump both"
    )


def test_no_version_skew_across_platforms(wheel_names):
    """One package, one version -- on every platform.

    jaxlib once resolved to 0.7.0 on Linux while jax stayed at 0.11.1, because
    jaxlib's Linux wheel carries a platform tag that was missing from
    PLATFORM_TAGS and pip quietly fell back through releases until one matched.
    Users got "solver unavailable (jaxlib is version 0.7.0, but this version of
    jax requires version >= 0.11.1)".
    """
    _load_fetch_wheels().check_version_skew(wheel_names)


def test_every_compiled_package_covers_every_platform(manifest, wheel_names):
    """A package with compiled wheels must ship one per declared platform.

    Version skew is the loud failure; a platform missing from the payload
    entirely is the quiet one -- the extension installs and the import fails.
    """
    #: Blender platform id -> predicate over a wheel's platform tag.
    matches = {
        "windows-x64": lambda tag: "win_amd64" in tag,
        "linux-x64": lambda tag: "linux" in tag and "x86_64" in tag,
        "macos-arm64": lambda tag: "macosx" in tag
        and ("arm64" in tag or "universal2" in tag),
    }
    platforms = manifest["platforms"]
    assert set(platforms) <= set(matches), "unhandled platform in the manifest"

    compiled: dict[str, list[str]] = {}
    for name in wheel_names:
        dist, python_tag, abi_tag, platform_tag = wheel_tags(name)
        if platform_tag == "any" and abi_tag == "none" and python_tag.startswith("py"):
            continue  # pure Python: one wheel serves every platform
        compiled.setdefault(dist.lower().replace("_", "-"), []).append(platform_tag)

    missing = {
        dist: [p for p in platforms if not any(matches[p](t) for t in tags)]
        for dist, tags in compiled.items()
    }
    missing = {dist: gaps for dist, gaps in missing.items() if gaps}
    assert not missing, f"compiled packages missing a platform wheel: {missing}"


def test_bundled_versions_match_the_lock(wheel_names):
    """Every bundled wheel is the version ``uv.lock`` pins: what the tests ran against.

    The dev venv is built from the lock and the zip from the manifest, and in a
    dev checkout the bundled wheels are never imported, so nothing else notices
    when they part. 0.4.0 shipped lxml 6.1.3 over a locked 6.1.2, and a later
    build rewrote the manifest to rospkg 1.6.3 over a locked 1.6.2.
    """
    fetch_wheels = _load_fetch_wheels()
    locked = fetch_wheels.locked_versions()
    apart = {}
    for name in wheel_names:
        dist, version = name.split("-")[:2]
        if locked.get(fetch_wheels.normalise(dist)) != version:
            apart[dist] = (version, locked.get(fetch_wheels.normalise(dist)))
    assert not apart, f"bundled (manifest, lock) versions disagree: {apart}"


def test_every_bundled_package_is_locked():
    """fetch_wheels downloads the lock's versions, so each package has to be in it."""
    fetch_wheels = _load_fetch_wheels()
    try:
        fetch_wheels.pinned_requirements()
    except SystemExit as exc:
        pytest.fail(str(exc))


def test_every_fetched_package_is_in_the_manifest(wheel_names):
    """The other direction from the version check: nothing fetch_wheels names is left out.

    Blender installs only the wheels the manifest lists, so a package dropped
    from it -- by a hand edit, or a fetch limited to one platform -- installs
    cleanly and fails at its first import.
    """
    fetch_wheels = _load_fetch_wheels()
    expected = {fetch_wheels.normalise(name) for name in fetch_wheels.PACKAGES}
    bundled = {fetch_wheels.normalise(wheel_tags(name)[0]) for name in wheel_names}
    missing = sorted(expected - bundled)
    assert not missing, f"fetched packages missing from the manifest: {missing}"


def _lock_with(tmp_path, *records: tuple[str, str]) -> Path:
    body = "".join(
        f'[[package]]\nname = "{name}"\nversion = "{version}"\n\n' for name, version in records
    )
    path = tmp_path / "uv.lock"
    path.write_text(body, encoding="utf-8")
    return path


def test_a_bundled_package_locked_at_two_versions_is_refused(tmp_path, monkeypatch):
    """A lock forked by markers must not have its last record quietly win."""
    fetch_wheels = _load_fetch_wheels()
    lock = _lock_with(tmp_path, ("rospkg", "1.6.2"), ("rospkg", "1.6.3"))
    monkeypatch.setattr(fetch_wheels, "LOCKFILE", lock)
    with pytest.raises(SystemExit, match="rospkg at both 1.6.2 and 1.6.3"):
        fetch_wheels.locked_versions()


def test_agreeing_or_unbundled_duplicates_are_fine(tmp_path, monkeypatch):
    fetch_wheels = _load_fetch_wheels()
    lock = _lock_with(
        tmp_path,
        ("rospkg", "1.6.3"), ("rospkg", "1.6.3"),
        ("mujoco", "3.1.0"), ("mujoco", "3.2.0"),
    )
    monkeypatch.setattr(fetch_wheels, "LOCKFILE", lock)
    assert fetch_wheels.locked_versions()["rospkg"] == "1.6.3"


def test_pyyaml_is_bundled(wheel_names):
    """Load Joint Limits imports yaml, and Blender's own Python has none.

    It arrives as a wheel or not at all. Checked by name rather than trusted to
    xacro and rospkg, which need it too but are not why Kinema imports it. The
    compiled-package test above then holds it to every platform.
    """
    dists = {wheel_tags(name)[0].lower() for name in wheel_names}
    assert "pyyaml" in dists, "PyYAML is missing from the bundled wheels"


def test_no_wheel_requires_a_newer_glibc_than_blender(wheel_names):
    """Linux wheels must stay within Blender's own glibc baseline.

    A manylinux_2_34 wheel imports fine on the machine that built the zip and
    fails on a supported host, which is the same class of bug as the version
    skew above but only reproducible on an older distro.
    """
    cap = _load_fetch_wheels().MAX_GLIBC_MINOR
    too_new = [
        name
        for name in wheel_names
        for tag in wheel_tags(name)[3].split(".")
        if tag.startswith("manylinux_2_") and int(tag.split("_")[2]) > cap
    ]
    assert not too_new, f"wheels requiring glibc newer than 2.{cap}: {too_new}"
