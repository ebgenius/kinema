"""The vendored jaxls orders its cost groups without looking at memory addresses.

jaxls sorts cost groups by ``str()`` of their tree structure. That text starts
with the cost's residual function -- which ``Cost.factory`` makes as a new lambda
every time a cost is built, printed with its address -- and only ends with the
cost's name::

    PyTreeDef(CustomNode(Cost[(<function Cost.factory.<locals>...<lambda> at 0x...>,
                               'l2_squared', 'auto', None, None, None, 'pose_residual')], ...

So the address decided the order of groups, and with it the program JAX traces
and the key it files the compiled program under in its persistent cache: all of
them followed wherever Python happened to allocate those lambdas. A solver
compiled in one Blender session missed the cache in the next. ``tools/vendor.py``
patches the sort to strip the addresses, which leaves the name to decide.
"""

from __future__ import annotations

import pytest

from ..conftest import load_addon_module

jax = pytest.importorskip("jax")
jaxls = load_addon_module("vendor.jaxls")


class Joint(jaxls.Var[jax.Array], default_factory=lambda: jax.numpy.zeros(1)):
    """One joint's value, standing in for a robot's."""


def residual_a(vals, var):
    return vals[var]


def residual_b(vals, var):
    return 2.0 * vals[var]


def _costs_addressed_against_their_names():
    """Two costs whose lambdas sit in the opposite order to their names.

    Without the patch the addresses win, so this is the arrangement where the
    order goes wrong. Built both ways round until the allocator obliges.
    """
    for _ in range(20):
        a = jaxls.Cost.factory(residual_a)(Joint(0))
        b = jaxls.Cost.factory(residual_b)(Joint(0))
        if id(b.compute_residual) < id(a.compute_residual):
            return a, b
        b = jaxls.Cost.factory(residual_b)(Joint(0))
        a = jaxls.Cost.factory(residual_a)(Joint(0))
        if id(b.compute_residual) < id(a.compute_residual):
            return a, b
    pytest.skip("could not get the two lambdas allocated against their names")


def test_cost_groups_are_ordered_by_name_not_by_address():
    a, b = _costs_addressed_against_their_names()
    for given in ([a, b], [b, a]):
        # Elimination off: with one variable it would eliminate everything and
        # build a zero-size array, which Blender's jaxlib refuses outright. The
        # groups are ordered before elimination, so this changes nothing tested.
        analyzed = jaxls.LeastSquaresProblem(given, [Joint(0)]).analyze(
            schur_elimination="off"
        )
        order = [stacked.name for stacked in analyzed._stacked_costs]
        assert order == ["residual_a", "residual_b"], f"given {[c.name for c in given]}"
