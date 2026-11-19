"""What a PyRoki solve reports about itself. Needs a real ``bpy`` and the JAX stack.

The claim: the solve readout shows how many iterations PyRoki took, as the
NumPy fallback's does, rather than a constant ``0 it`` that reads as though
the solver did nothing (#57). Pays one PyRoki compile.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from ..conftest import requires_bpy

pytestmark = requires_bpy


@pytest.fixture
def builder(addon):
    return importlib.import_module(f"{addon.__name__}.rig.builder")


@pytest.fixture
def manager(addon):
    return importlib.import_module(f"{addon.__name__}.solver.manager")


@pytest.fixture
def handlers(addon):
    return importlib.import_module(f"{addon.__name__}.handlers")


@pytest.fixture(autouse=True)
def _free_compiled_solvers(addon, manager):
    """Drop the compiled kernels, so the next module starts with none of them."""
    yield
    import gc

    manager.invalidate()
    gc.collect()


@pytest.fixture
def arm6(addon, fixture_dir, clean_scene, builder):
    """arm6 in a working pose, with an IK target and PyRoki as its solver."""
    import bpy

    assert "FINISHED" in bpy.ops.kinema.build_robot(
        filepath=str(fixture_dir / "arm6.urdf")
    )
    rig = next(o for o in bpy.data.objects if builder.is_kinema_rig(o))
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    for name, value in {"joint2": -0.6, "joint3": 1.0, "joint5": 0.4}.items():
        rig.pose.bones[name].rotation_euler[1] = value
    bpy.context.view_layer.update()
    bpy.ops.kinema.add_ik()
    rig.kinema_ik_enabled = False
    rig.kinema_solver_mode = "PYROKI"
    return rig


def test_a_pyroki_solve_reports_the_iterations_it_took(arm6, manager, handlers):
    """The target 50 mm from the tool: PyRoki has to iterate to reach it, and says
    how many times it did."""
    import bpy

    solver = manager.get_solver(arm6)
    tip = arm6.pose.bones[solver.tip_bone]
    with handlers.suspended():
        goal = tip.matrix.copy()
        goal.translation.z -= 0.05
        arm6.pose.bones[solver.ik_bone].matrix = goal
        bpy.context.view_layer.update()
        result = solver.solve(arm6, manager.MODE_PYROKI)
        bpy.context.view_layer.update()

    assert result.backend == "PyRoki", "PyRoki did not answer, so this proves nothing"
    assert result.converged
    assert np.linalg.norm(np.array(tip.matrix)[:3, 3] - np.array(goal)[:3, 3]) < 1e-4
    assert result.iterations > 0
    assert f"{result.iterations} it" in result.summary
