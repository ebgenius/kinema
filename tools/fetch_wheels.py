"""Download the runtime wheel payload and write it into blender_manifest.toml.

Blender extensions ship their Python dependencies as unmodified wheels bundled
in the extension zip; installing packages at runtime is forbidden by the
extension guidelines. This script materialises that payload.

Three things here are load-bearing and easy to get wrong:

1. **``--no-deps`` with an explicit package list.** ``yourdfpy`` depends on
   ``trimesh[easy]``, and those extras drag in embreex (37 MB), Pillow,
   networkx, shapely, rtree, manifold3d and more -- about 90 MB of payload that
   Kinema never imports. Resolving dependencies automatically blows the size
   budget; naming every package explicitly keeps it at ~112 MB.

2. **NumPy is deliberately absent.** Blender 5.2 ships NumPy 2.3.4 in its own
   ``site-packages``. Bundling a second copy would shadow Blender's, risking an
   ABI mismatch against modules compiled against theirs. Verified in the M0
   spike: the JAX stack runs fine on Blender's NumPy.

3. **cp313 only.** Blender 5.2 LTS embeds CPython 3.13.13. A wheel built for
   any other ABI silently fails to load at install time.

Usage::

    uv run python tools/fetch_wheels.py            # all platforms
    uv run python tools/fetch_wheels.py --platform windows-x64
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ADDON_DIR = REPO_ROOT / "src" / "kinema"
WHEEL_DIR = ADDON_DIR / "wheels"
MANIFEST = ADDON_DIR / "blender_manifest.toml"

PYTHON_VERSION = "3.13"
ABI = "cp313"

#: Blender platform id -> pip ``--platform`` tags to try, most specific first.
PLATFORM_TAGS: dict[str, tuple[str, ...]] = {
    "windows-x64": ("win_amd64",),
    "linux-x64": (
        "manylinux_2_28_x86_64",
        "manylinux_2_17_x86_64",
        "manylinux2014_x86_64",
    ),
    # Several macOS baselines: projects raise their minimum over time, and
    # SciPy's cp313 arm64 wheels are not built for macosx_11_0 at all. pip
    # accepts every tag and picks whichever the package actually publishes.
    "macos-arm64": (
        "macosx_15_0_arm64",
        "macosx_14_0_arm64",
        "macosx_13_0_arm64",
        "macosx_12_0_arm64",
        "macosx_11_0_arm64",
    ),
}

#: Every runtime import, named explicitly. Grouped by why it is here.
PACKAGES: tuple[str, ...] = (
    # --- solver core (vendored pyroki + jaxls import these) ---
    "jax", "jaxlib", "ml_dtypes", "opt_einsum", "scipy",
    "jaxlie", "jax_dataclasses", "jaxtyping", "wadler-lindig", "typeguard",
    # --- jaxls runtime deps ---
    "loguru", "termcolor", "tqdm", "rich", "markdown-it-py", "mdurl",
    "pygments", "typing_extensions", "win32-setctime", "colorama",
    # --- URDF parsing ---
    "yourdfpy", "lxml", "trimesh", "six",
    # --- xacro descriptions (ur5e and many other catalog entries need this) ---
    "xacrodoc", "xacro", "rospkg", "pyyaml", "pyparsing", "packaging",
    "python-dateutil", "docutils", "setuptools",
    # --- COLLADA meshes: Blender 5.0 removed its own .dae importer ---
    "pycollada",
    # --- robot catalog ---
    "robot_descriptions", "GitPython", "gitdb", "smmap",
)

#: Provided by Blender itself -- bundling these would shadow the host's copies.
EXCLUDED = ("numpy", "requests", "certifi", "idna", "charset-normalizer", "urllib3")


def download(platform: str, dest: Path) -> None:
    tags = PLATFORM_TAGS[platform]
    cmd = [
        sys.executable, "-m", "pip", "download",
        "--only-binary=:all:", "--no-deps",
        "--python-version", PYTHON_VERSION,
        "--implementation", "cp", "--abi", ABI,
        "-d", str(dest),
    ]
    for tag in tags:
        cmd += ["--platform", tag]
    cmd += list(PACKAGES)

    print(f"  downloading {len(PACKAGES)} packages for {platform} ({tags[0]})")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if "No module named pip" in (result.stderr or ""):
        raise SystemExit(
            "fetch_wheels: this interpreter has no pip. "
            "Run `uv sync --group dev` (pip is a declared dev dependency)."
        )
    if result.returncode != 0:
        # pip fails the whole batch if one package has no matching wheel; retry
        # individually so the offender is named rather than hidden.
        print("  batch failed, retrying individually to identify the cause:")
        for pkg in PACKAGES:
            one = [c for c in cmd if c not in PACKAGES] + [pkg]
            r = subprocess.run(one, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"    FAILED {pkg}: {r.stderr.strip().splitlines()[-1][:160]}")
        raise SystemExit("fetch_wheels: could not resolve the full payload")


def write_manifest(wheel_names: list[str]) -> None:
    entries = "\n".join(f'  "./wheels/{n}",' for n in sorted(wheel_names))
    block = f"wheels = [\n{entries}\n]"
    text = MANIFEST.read_text(encoding="utf-8")
    new, count = re.subn(r"wheels = \[[^\]]*\]", block, text, count=1)
    if count != 1:
        raise SystemExit("fetch_wheels: could not locate the wheels list in the manifest")
    MANIFEST.write_text(new, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=sorted(PLATFORM_TAGS),
                        action="append", dest="platforms",
                        help="limit to one platform (repeatable); default: all")
    parser.add_argument("--clean", action="store_true",
                        help="remove existing wheels before downloading")
    args = parser.parse_args()

    platforms = args.platforms or list(PLATFORM_TAGS)
    if args.clean:
        shutil.rmtree(WHEEL_DIR, ignore_errors=True)
    WHEEL_DIR.mkdir(parents=True, exist_ok=True)

    for platform in platforms:
        download(platform, WHEEL_DIR)

    for wheel in WHEEL_DIR.glob("*.whl"):
        stem = wheel.name.split("-")[0].lower().replace("_", "-")
        if stem in {e.lower().replace("_", "-") for e in EXCLUDED}:
            print(f"  removing host-provided package: {wheel.name}")
            wheel.unlink()

    names = [w.name for w in WHEEL_DIR.glob("*.whl")]
    if not names:
        raise SystemExit("fetch_wheels: no wheels downloaded")
    write_manifest(names)

    total_mb = sum(w.stat().st_size for w in WHEEL_DIR.glob("*.whl")) / 1e6
    print(f"\n{len(names)} wheels, {total_mb:.1f} MB total across {len(platforms)} platform(s)")
    print(f"manifest updated: {MANIFEST.relative_to(REPO_ROOT)}")
    if len(platforms) > 1:
        print("Build with --split-platforms so each zip carries only its own wheels.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
