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

The vendored copies stay as close to upstream as they can, and every divergence
is declared on the :class:`VendoredPackage` rather than applied by hand:

**Dropped directories.** ``pyroki/viewer`` depends on ``viser``, a web-based 3D
visualiser -- useless inside Blender and ~40 MB. ``jaxls/_py310`` is the
fallback tree for Python 3.10 and 3.11; ``jaxls/__init__.py`` selects it with
``sys.version_info``, and Blender 5.2 embeds 3.13, so it can never be reached.

**Rewritten imports.** This is the load-bearing one. Blender refuses to let an
extension put anything on ``sys.path`` or register a top-level module from
inside its own directory -- ``addon_utils.py`` walks ``sys.modules`` and flags
every module whose file lives in the extension but whose name is not under
``bl_ext.<repo>.<addon>``. Vendoring is still allowed; the packages just have to
import as ``kinema.vendor.jaxls`` and ``kinema.vendor.pyroki``.

Both trees are almost entirely relative already, so this is seven lines. They
are listed in ``rewrite_imports`` and applied on every vendor run, because the
failure mode is a pin bump silently restoring 28 policy violations. A rewrite
whose target line has vanished upstream is a hard error here, not a surprise at
someone's first solve.

The list is checked rather than trusted: after rewriting, the staged tree is
scanned and any surviving absolute self-import fails the run. A hand-written
list is precisely the thing that misses a file in a subdirectory, and the
symptom -- ``ModuleNotFoundError: No module named 'jaxls'`` -- shows up far
from its cause.

Usage::

    uv run python tools/vendor.py           # vendor at the pinned commits
    uv run python tools/vendor.py --check   # verify tree matches upstream
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = REPO_ROOT / "src" / "kinema" / "vendor"
LICENSE_DIR = REPO_ROOT / "LICENSES"


#: Provenance written into each vendored tree; also how a build checks itself.
STAMP_NAME = "_KINEMA_VENDORED.txt"


@dataclass(frozen=True)
class VendoredPackage:
    name: str
    url: str
    #: A full 40-character SHA, never a branch name. ``main`` moves, and a
    #: moving vendor pin means two builds of the same tag can ship different
    #: solver code -- with no diff anywhere to show it.
    commit: str
    #: Path of the importable package *inside* the cloned repo.
    source_subdir: str
    license_file: str
    #: Sub-directories to delete after copying (heavy or irrelevant deps).
    drop_dirs: tuple[str, ...] = ()
    #: Exact lines to remove from ``__init__.py`` (newline-agnostic).
    drop_init_lines: tuple[str, ...] = ()
    #: ``(file relative to the package, exact old line, new line)``. Turns the
    #: package's absolute self-imports into relative ones so it can be imported
    #: as ``kinema.vendor.<name>`` -- see the module docstring. Newline-agnostic
    #: and order-independent; a line that is not found is a hard error.
    rewrite_imports: tuple[tuple[str, str, str], ...] = ()


PACKAGES = (
    VendoredPackage(
        name="pyroki",
        url="https://github.com/chungmin99/pyroki.git",
        # Pinned. Bump deliberately and re-run the integration tests: PyRoki's
        # cost API is not stable across commits.
        commit="388e43e1fc0d0ee382968d3dd72970fd62a0450c",
        source_subdir="src/pyroki",
        license_file="LICENSE",
        # viewer/ is the only viser-dependent part of the package.
        drop_dirs=("viewer",),
        drop_init_lines=("from . import viewer as viewer",),
        # pyroki reaches its sibling by name, from three different depths.
        # Both packages live under vendor/, so the number of dots is one per
        # level back up to it: ".." from pyroki/, "..." from pyroki/_residuals/.
        # _robot.py keeps the module itself bound rather than a name from it,
        # because it uses `jaxls.Var` inside an annotation.
        rewrite_imports=(
            ("_robot.py", "import jaxls", "from .. import jaxls"),
            ("costs.py", "from jaxls import Cost", "from ..jaxls import Cost"),
            (
                "collision/_robot_collision.py",
                "from pyroki._robot import Robot",
                "from .._robot import Robot",
            ),
            (
                "_residuals/_pose_residual_analytic_jac.py",
                "import jaxls",
                "from ... import jaxls",
            ),
            (
                "_residuals/_pose_residual_numerical_jac.py",
                "import jaxls",
                "from ... import jaxls",
            ),
            (
                "_residuals/_residuals.py",
                "from jaxls import Var, VarValues",
                "from ...jaxls import Var, VarValues",
            ),
        ),
    ),
    VendoredPackage(
        name="jaxls",
        url="https://github.com/brentyi/jaxls.git",
        commit="50a58be88c5ef74532f09e3f55268b4f02c490e3",
        source_subdir="src/jaxls",
        license_file="LICENSE",
        # The Python 3.10/3.11 fallback tree. __init__.py picks it with
        # sys.version_info, and Blender 5.2 embeds 3.13, so it is unreachable.
        drop_dirs=("_py310",),
        # The only absolute self-import in the whole tree.
        rewrite_imports=(
            (
                "_solvers.py",
                "from jaxls._preconditioning import (",
                "from ._preconditioning import (",
            ),
        ),
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


def _rewrite_imports(
    package_root: Path, rewrites: tuple[tuple[str, str, str], ...]
) -> None:
    """Turn absolute self-imports into relative ones, in place.

    Same newline handling and same loud failure as :func:`_patch_init`, for the
    same reason: silence here means the add-on ships with the very policy
    violations this exists to prevent, and nothing notices until Blender puts a
    warning triangle on the add-on.

    Matched with the trailing newline attached so a line cannot match a longer
    line it happens to prefix -- ``import jaxls`` must not hit
    ``import jaxls_extras``.
    """
    for relative_path, old, new in rewrites:
        target = package_root / relative_path
        if not target.is_file():
            raise SystemExit(
                f"vendor: {relative_path} not found in {package_root.name}\n"
                f"Upstream layout changed -- review before bumping the pin."
            )
        data = target.read_bytes()
        for newline in (b"\r\n", b"\n"):
            candidate = old.encode() + newline
            if candidate in data:
                data = data.replace(candidate, new.encode() + newline)
                break
        else:
            raise SystemExit(
                f"vendor: expected line not found in {relative_path}: {old!r}\n"
                f"Upstream layout changed -- review before bumping the pin."
            )
        target.write_bytes(data)
        print(f"    rewrote {relative_path}: {old!r} -> {new!r}")


#: Package names that must never appear in an absolute import inside a
#: vendored tree, because both are vendored and neither is installed.
_VENDORED_NAMES = frozenset({"jaxls", "pyroki"})


def _absolute_self_imports(path: Path) -> list[str]:
    """Every absolute import of a vendored package in one file.

    Parsed rather than pattern-matched. A regex over source text gets this
    wrong in both directions: it misses ``import os, jaxls`` and
    ``x(); import pyroki``, and it falsely matches ``from jaxls_extra import
    ...``. Since the whole point of this check is to be trustworthy when the
    hand-written rewrite list is not, a false negative defeats it entirely.

    ``ast.walk`` also reaches imports nested inside functions and ``try``
    blocks, which upstream uses for optional dependencies.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in _VENDORED_NAMES:
                    found.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            # level > 0 is already relative, which is the goal state.
            if node.level == 0 and node.module:
                if node.module.split(".")[0] in _VENDORED_NAMES:
                    names = ", ".join(a.name for a in node.names)
                    found.append(
                        f"line {node.lineno}: from {node.module} import {names}"
                    )
    return found


def _assert_no_absolute_self_imports(package_root: Path) -> None:
    """Fail if any absolute import of a vendored package survives.

    ``rewrite_imports`` is an explicit list so the divergence from upstream is
    declared and reviewable, but a hand-written list is exactly the thing that
    misses a file. This is the post-condition: whatever the list says, the tree
    that ships must contain no absolute self-import at all.

    Worth having because the failure is remote from its cause -- a missed line
    surfaces as ``ModuleNotFoundError: No module named 'jaxls'`` at a user's
    first solve, long after the vendor run that caused it.
    """
    offenders = []
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(package_root).as_posix()
        for entry in _absolute_self_imports(path):
            offenders.append(f"  {relative}: {entry}")
    if offenders:
        raise SystemExit(
            f"vendor: {package_root.name} still has absolute self-imports:\n"
            + "\n".join(offenders)
            + "\n\nAdd them to rewrite_imports. Left in, they import as "
            "top-level modules, which Blender reports as a policy violation "
            "and which fail outright once vendor/ is off sys.path."
        )


def _clone_at(url: str, commit: str, dest: Path) -> str:
    """Clone ``url`` at exactly ``commit``.

    ``git clone --branch`` resolves refs only, so it cannot take a SHA. Fetching
    the single wanted commit is the cheap path and GitHub serves it; a host that
    refuses falls back to full history, which still lands on the same tree.
    """
    _run(["git", "init", "--quiet", str(dest)])
    _run(["git", "remote", "add", "origin", url], cwd=dest)
    try:
        _run(["git", "fetch", "--quiet", "--depth", "1", "origin", commit], cwd=dest)
    except subprocess.CalledProcessError:
        print("    host refused a single-commit fetch; fetching full history")
        _run(["git", "fetch", "--quiet", "origin"], cwd=dest)
    _run(["git", "checkout", "--quiet", commit], cwd=dest)
    return _run(["git", "rev-parse", "HEAD"], cwd=dest)


def recipe_fingerprint(pkg: VendoredPackage) -> str:
    """A short hash of everything this script does to the tree after cloning.

    The commit alone is not enough to say a vendored tree is current. The trees
    are gitignored, so they persist across branches and pulls, and *this file*
    changes independently of the pins: adding a rewrite or a dropped directory
    leaves an existing tree stale in a way the SHA cannot see. Without this, a
    tree vendored before the import rewrites passes ``--check`` and builds a
    release with the old absolute imports in it.
    """
    recipe = repr((pkg.drop_dirs, pkg.drop_init_lines, pkg.rewrite_imports))
    return hashlib.sha256(recipe.encode()).hexdigest()[:12]


def recorded_state(dest: Path) -> tuple[str, str] | None:
    """``(commit, recipe)`` a vendored tree was built from, or None if absent.

    A tree stamped before recipes were fingerprinted has no ``recipe:`` line;
    it reads back as empty, which never matches and correctly forces a
    re-vendor.
    """
    stamp = dest / STAMP_NAME
    if not (dest / "__init__.py").is_file() or not stamp.is_file():
        return None
    commit = recipe = ""
    for line in stamp.read_text(encoding="utf-8").splitlines():
        if line.startswith("commit:"):
            commit = line.split(":", 1)[1].strip()
        elif line.startswith("recipe:"):
            recipe = line.split(":", 1)[1].strip()
    return (commit, recipe) if commit else None


def vendor(pkg: VendoredPackage, *, check: bool) -> bool:
    dest = VENDOR_DIR / pkg.name

    if check:
        # Now that the pin is a SHA, the stamp answers this without a clone.
        state = recorded_state(dest)
        if state is None:
            print(f"    MISSING: {dest}")
            return False
        recorded, recipe = state
        if recorded != pkg.commit:
            print(f"    STALE: vendored at {recorded[:10]}, "
                  f"pinned at {pkg.commit[:10]}")
            return False
        expected = recipe_fingerprint(pkg)
        if recipe != expected:
            print(f"    STALE: vendored with recipe {recipe or '(none)'}, "
                  f"tools/vendor.py now specifies {expected}")
            print("           re-run: uv run python tools/vendor.py")
            return False
        print(f"    present at {recorded[:10]} (recipe {recipe})")
        return True

    with tempfile.TemporaryDirectory() as tmp:
        clone_dir = Path(tmp) / pkg.name
        print(f"  cloning {pkg.url} @ {pkg.commit[:10]}")
        sha = _clone_at(pkg.url, pkg.commit, clone_dir)

        src = clone_dir / pkg.source_subdir
        if not src.is_dir():
            raise SystemExit(f"vendor: {pkg.source_subdir} missing in {pkg.name}")

        staged = Path(tmp) / "staged"
        shutil.copytree(src, staged)
        for drop in pkg.drop_dirs:
            shutil.rmtree(staged / drop, ignore_errors=True)
            print(f"    dropped {drop}/")
        _patch_init(staged / "__init__.py", pkg.drop_init_lines)
        _rewrite_imports(staged, pkg.rewrite_imports)
        _assert_no_absolute_self_imports(staged)

        # Record provenance so the vendored tree is auditable -- the extension
        # portal reviews bundled third-party code.
        rewrites = "\n".join(
            f"  {path}: {old!r} -> {new!r}"
            for path, old, new in pkg.rewrite_imports
        )
        (staged / STAMP_NAME).write_text(
            f"{pkg.name}\nsource: {pkg.url}\ncommit: {sha}\n"
            f"recipe: {recipe_fingerprint(pkg)}\n"
            f"dropped: {', '.join(pkg.drop_dirs) or '(nothing)'}\n"
            f"rewritten imports:\n{rewrites or '  (none)'}\n",
            encoding="utf-8",
        )

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
                        help="verify vendored trees match the pinned commits")
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


