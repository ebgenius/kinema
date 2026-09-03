"""Blender's extension policy, as assertions.

Blender 5.2 puts a warning triangle on an add-on that breaks either of two
rules, and lists every offence in Preferences. ``addon_utils`` computes them by
walking ``sys.modules``:

* a module whose ``__file__`` is inside the extension but whose name is **not**
  under ``bl_ext.<repo>.<addon>`` is a violation;
* a ``sys.path`` entry inside the extension is a violation.

Kinema broke both from 0.2.0 to 0.3.0 -- 28 warnings, one per vendored jaxls
and pyroki module plus the ``sys.path`` entry that put them there. Vendoring is
allowed; the packages just have to import under the add-on's own name.

These run against the *dev checkout*, which is linked into Blender's extensions
directory by ``dev.py link``, so they catch a regression at the source. The
shipped zip is checked separately by the scratchpad's ``policy_check.py``, which
calls Blender's own reporter.
"""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("bpy")

from ..conftest import load_addon_module  # noqa: E402

runtime = load_addon_module("runtime")


@pytest.fixture(scope="module")
def loaded(addon):
    """The solver stack, actually imported.

    Not a no-op fixture: an unimported module cannot appear in ``sys.modules``,
    so every assertion below would pass vacuously on a lazy add-on.
    """
    stack = runtime.load_solver_stack()
    if not stack:
        pytest.skip(f"solver stack unavailable: {runtime.solver_error()}")
    return stack


def test_the_vendored_packages_are_not_top_level(loaded):
    """The 28-violation bug. `import jaxls` must resolve to nothing."""
    for name in ("jaxls", "pyroki"):
        assert name not in sys.modules, (
            f"'{name}' is a top-level module; it must import as "
            f"{runtime.__package__}.vendor.{name}"
        )


def test_the_vendored_packages_are_under_the_addon(loaded):
    for name in ("jaxls", "pyroki"):
        dotted = f"{runtime.__package__}.vendor.{name}"
        assert dotted in sys.modules, f"{dotted} was never imported"
        assert loaded[name] is sys.modules[dotted], (
            f"the stack's '{name}' is a different object from {dotted} -- "
            "the source is being imported twice under two names"
        )


def test_every_vendored_submodule_is_namespaced(loaded):
    """Not just the package roots: Blender flags each submodule separately, so
    a stray absolute import inside the tree costs one warning per module."""
    addon_root = f"{runtime.__package__}."
    strays = [
        name
        for name, module in list(sys.modules.items())
        if getattr(module, "__file__", None)
        and "vendor" in str(module.__file__).replace("\\", "/").split("/")
        and str(module.__file__).replace("\\", "/").endswith(".py")
        and not name.startswith(addon_root)
        and name.split(".")[0] in ("jaxls", "pyroki")
    ]
    assert not strays, f"vendored modules outside the add-on namespace: {strays}"


def test_nothing_was_added_to_sys_path(loaded):
    """The second half of the policy, and the one that used to be deliberate."""
    addon_dir = str(runtime.__file__).replace("\\", "/").rsplit("/", 1)[0]
    inside = [p for p in sys.path if p and addon_dir in str(p).replace("\\", "/")]
    assert not inside, f"sys.path entries inside the add-on: {inside}"
