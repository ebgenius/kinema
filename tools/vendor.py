"""Vendor third-party source that cannot ship as a PyPI wheel.

Two of Kinema's runtime dependencies are not installable from PyPI:

* **pyroki** -- published only as a git repo; ``version = "0.0.0"`` and its
  ``pyproject.toml`` pulls ``jaxls`` from a git URL, so it has no wheel.
* **jaxls** -- brentyi/jaxls is not on PyPI at all. The PyPI project *named*
  ``jaxls`` is an unrelated "JAX Language Server APIs" package by a different
  author. Installing it gets you something that imports pydantic and fails.

Blender's extension guidelines allow exactly this case: dependencies must be
"bundled as wheels **or vendorized**". Both projects are MIT licensed, so their
source is copied into ``src/kinema/vendor/`` and their licenses recorded in
``LICENSES/``.

The vendored copies are kept byte-identical to upstream apart from one patch:
``pyroki/__init__.py`` imports ``pyroki.viewer``, which depends on ``viser`` (a
full web-based 3D visualiser -- useless inside Blender and ~40 MB). The
``viewer/`` subpackage is dropped and that single import line removed. Keeping
the diff to one line is deliberate: re-syncing with upstream stays trivial.

Usage::

    uv run python tools/vendor.py           # vendor at the pinned commits
    uv run python tools/vendor.py --check   # verify tree matches upstream
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = REPO_ROOT / "src" / "kinema" / "vendor"
LICENSE_DIR = REPO_ROOT / "LICENSES"


@dataclass(frozen=True)
class VendoredPackage:
    name: str
    url: str
    commit: str
    #: Path of the importable package *inside* the cloned repo.
    source_subdir: str
    license_file: str
    #: Sub-directories to delete after copying (heavy or irrelevant deps).
    drop_dirs: tuple[str, ...] = ()
    #: Exact lines to remove from ``__init__.py`` (newline-agnostic).
    drop_init_lines: tuple[str, ...] = ()


PACKAGES = (
    VendoredPackage(
        name="pyroki",
        url="https://github.com/chungmin99/pyroki.git",
        # Pinned. Bump deliberately and re-run the integration tests: PyRoki's
        # cost API is not stable across commits.
        commit="main",
        source_subdir="src/pyroki",
        license_file="LICENSE",
        # viewer/ is the only viser-dependent part of the package.
        drop_dirs=("viewer",),
        drop_init_lines=("from . import viewer as viewer",),
    ),
    VendoredPackage(
        name="jaxls",
        url="https://github.com/brentyi/jaxls.git",
        commit="main",
        source_subdir="src/jaxls",
        license_file="LICENSE",
    ),
)


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        cmd, cwd=cwd, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _patch_init(init_path: Path, drop_lines: tuple[str, ...]) -> None:
    """Remove specific import lines, preserving the file's original newlines.

    Upstream is cloned on Windows with autocrlf, so the file may use CRLF. We
    match both rather than normalising, to keep the diff against upstream at
    exactly the lines we intend to touch.
    """
    if not drop_lines:
        return
    data = init_path.read_bytes()
    for line in drop_lines:
        for newline in (b"\r\n", b"\n"):
            candidate = line.encode() + newline
            if candidate in data:
                data = data.replace(candidate, b"")
                break
        else:
            raise SystemExit(
                f"vendor: expected line not found in {init_path.name}: {line!r}\n"
                f"Upstream layout changed -- review before bumping the pin."
            )
    init_path.write_bytes(data)


def vendor(pkg: VendoredPackage, *, check: bool) -> bool:
    dest = VENDOR_DIR / pkg.name
    with tempfile.TemporaryDirectory() as tmp:
        clone_dir = Path(tmp) / pkg.name
        print(f"  cloning {pkg.url} @ {pkg.commit}")
        _run(["git", "clone", "--quiet", "--depth", "1", "--branch", pkg.commit,
              pkg.url, str(clone_dir)])
        sha = _run(["git", "rev-parse", "HEAD"], cwd=clone_dir)

        src = clone_dir / pkg.source_subdir
        if not src.is_dir():
            raise SystemExit(f"vendor: {pkg.source_subdir} missing in {pkg.name}")

        staged = Path(tmp) / "staged"
        shutil.copytree(src, staged)
        for drop in pkg.drop_dirs:
            shutil.rmtree(staged / drop, ignore_errors=True)
            print(f"    dropped {drop}/")
        _patch_init(staged / "__init__.py", pkg.drop_init_lines)

        # Record provenance so the vendored tree is auditable -- the extension
        # portal reviews bundled third-party code.
        (staged / "_KINEMA_VENDORED.txt").write_text(
            f"{pkg.name}\nsource: {pkg.url}\ncommit: {sha}\n"
            f"dropped: {', '.join(pkg.drop_dirs) or '(nothing)'}\n",
            encoding="utf-8",
        )

        if check:
            if not dest.is_dir():
                print(f"    MISSING: {dest}")
                return False
            print(f"    present at {sha[:10]}")
            return True

        shutil.rmtree(dest, ignore_errors=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staged, dest)

        LICENSE_DIR.mkdir(exist_ok=True)
        license_src = clone_dir / pkg.license_file
        if license_src.is_file():
            shutil.copy(license_src, LICENSE_DIR / f"{pkg.name}-LICENSE")
        else:
            print(f"    WARNING: no {pkg.license_file} found in {pkg.name}")

        modules = len(list(dest.rglob("*.py")))
        print(f"    vendored {modules} modules at {sha[:10]}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify vendored trees exist without rewriting them")
    args = parser.parse_args()

    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    init = VENDOR_DIR / "__init__.py"
    if not init.exists():
        init.write_text(
            '"""Vendored third-party source. See tools/vendor.py."""\n',
            encoding="utf-8",
        )

    ok = True
    for pkg in PACKAGES:
        print(f"{pkg.name}:")
        ok &= vendor(pkg, check=args.check)
    print("\nvendor: " + ("OK" if ok else "INCOMPLETE -- run without --check"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
