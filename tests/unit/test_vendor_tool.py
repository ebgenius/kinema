"""The vendoring tool's two safety nets.

Neither is exercised by an ordinary run, which is the problem: both only matter
when something has already gone wrong, and the failure they prevent -- a
release built from a stale or half-rewritten vendored tree -- is invisible
until Blender refuses the add-on or the solver fails to import.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def vendor_tool():
    """tools/vendor.py, imported by path -- tools/ is not a package."""
    spec = importlib.util.spec_from_file_location(
        "kinema_vendor_tool", REPO_ROOT / "tools" / "vendor.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


class TestAbsoluteSelfImportScan:
    """The post-condition that makes the hand-written rewrite list trustworthy.

    It replaced a regex that got this wrong in both directions, so the cases
    below are the ones the regex actually failed.
    """

    def scan(self, vendor_tool, tmp_path, source: str) -> list[str]:
        path = tmp_path / "sample.py"
        path.write_text(source, encoding="utf-8")
        return vendor_tool._absolute_self_imports(path)

    @pytest.mark.parametrize(
        "source",
        [
            "import jaxls",
            "import pyroki",
            "from jaxls import Var",
            "from jaxls._preconditioning import make",
            "import os, jaxls",                      # regex missed this
            "import sys\nimport jaxls as j",
            "def f():\n    import pyroki\n",         # nested in a function
            "try:\n    import jaxls\nexcept ImportError:\n    pass\n",
        ],
    )
    def test_absolute_self_imports_are_found(self, vendor_tool, tmp_path, source):
        assert self.scan(vendor_tool, tmp_path, source), source

    @pytest.mark.parametrize(
        "source",
        [
            "from . import jaxls",
            "from .. import jaxls",
            "from ...jaxls import Var",
            "from .._robot import Robot",
            "import jaxls_extra",                    # regex falsely matched this
            "from jaxls_extra import thing",         # and this
            "import numpy as np",
            "jaxls = None  # not an import at all",
            "'''import jaxls, in a docstring'''",
        ],
    )
    def test_these_are_not_flagged(self, vendor_tool, tmp_path, source):
        assert self.scan(vendor_tool, tmp_path, source) == [], source

    def test_the_shipped_trees_are_clean(self, vendor_tool):
        """The real assertion: what is on disk right now would pass."""
        vendor_dir = REPO_ROOT / "src" / "kinema" / "vendor"
        for package in ("jaxls", "pyroki"):
            root = vendor_dir / package
            if not root.is_dir():
                pytest.skip(f"{package} is not vendored here")
            vendor_tool._assert_no_absolute_self_imports(root)


class TestRecipeFingerprint:
    """Why the commit SHA alone cannot say a vendored tree is current."""

    def test_changing_a_rewrite_changes_the_fingerprint(self, vendor_tool):
        pkg = vendor_tool.PACKAGES[0]
        before = vendor_tool.recipe_fingerprint(pkg)
        altered = type(pkg)(
            **{
                **pkg.__dict__,
                "rewrite_imports": pkg.rewrite_imports[:-1],
            }
        )
        assert vendor_tool.recipe_fingerprint(altered) != before

    def test_changing_a_dropped_directory_changes_the_fingerprint(self, vendor_tool):
        pkg = vendor_tool.PACKAGES[0]
        before = vendor_tool.recipe_fingerprint(pkg)
        altered = type(pkg)(**{**pkg.__dict__, "drop_dirs": ()})
        assert vendor_tool.recipe_fingerprint(altered) != before

    def test_the_same_recipe_is_stable(self, vendor_tool):
        """It goes in a stamp that is compared on every build, so it must not
        drift between runs of the same code."""
        pkg = vendor_tool.PACKAGES[0]
        assert vendor_tool.recipe_fingerprint(pkg) == vendor_tool.recipe_fingerprint(pkg)

    def test_the_shipped_trees_match_the_current_recipe(self, vendor_tool):
        """A tree vendored before a recipe change must read as stale -- this is
        the check that stops a release being built from one."""
        for pkg in vendor_tool.PACKAGES:
            dest = vendor_tool.VENDOR_DIR / pkg.name
            state = vendor_tool.recorded_state(dest)
            if state is None:
                pytest.skip(f"{pkg.name} is not vendored here")
            _, recipe = state
            assert recipe == vendor_tool.recipe_fingerprint(pkg), (
                f"{pkg.name} was vendored with a different recipe; "
                "re-run tools/vendor.py"
            )
