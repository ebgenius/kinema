"""IK from a pose where a joint is pinned at a limit while the arm is singular (#56).

The Franka's elbow is limited to [-3.07, -0.07]: at rest it is held just short of
straight. Live IK from there used to stop 75 mm short of a target it can reach,
and stay there; the elbow only came unstuck once moved by hand. Needs a real
``bpy`` and PyRoki.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from ..conftest import requires_bpy

pytestmark = requires_bpy


@pytest.fixture
def modules(addon):
    def get(name):
        return importlib.import_module(f"{addon.__name__}.{name}")

    return get


@pytest.fixture
def fer(modules, fixture_dir, clean_scene):
    """The Franka at rest with an IK target moved as #56's report moves it."""
    import bpy
    from mathutils import Vector

    builder = modules("rig.builder")
    assert "FINISHED" in bpy.ops.kinema.build_robot(
        filepath=str(fixture_dir / "fer_kinematics.urdf")
    )
    rig = bpy.context.view_layer.objects.active
    rig.kinema_solver_mode = "PYROKI"
    assert "FINISHED" in bpy.ops.kinema.add_ik()
    rig.kinema_ik_enabled = False
    goal = rig.pose.bones[rig[builder.PROP_IK_BONE]]
    matrix = goal.matrix.copy()
    matrix.translation += Vector((0.0, 0.05, -0.08))
    goal.matrix = matrix
    bpy.context.view_layer.update()
    return rig


def _pyroki_or_skip(modules, rig):
    manager = modules("solver.manager")
    solver = manager.get_solver(rig)
    if solver is None or solver.pyroki(rig) is None:
        pytest.skip("PyRoki isn't available here")
    return manager, solver


class TestFromAPinnedElbow:
    def test_the_target_is_reached(self, fer, modules):
        import bpy

        manager, solver = _pyroki_or_skip(modules, fer)
        chain_mod = modules("solver.chain")
        # The preconditions: the elbow starts pinned, and the target is reachable.
        elbow = solver.chain.bone_names.index("fer_joint4")
        assert solver.chain.upper[elbow] == pytest.approx(-0.0698)
        assert manager.find_solutions(fer, solver, seeds=40), "reachable from other seeds"

        modules("handlers").solve_rig(fer, force=True)
        bpy.context.view_layer.update()
        result = manager.get_solver(fer).last_result
        assert result.converged, result.summary
        q = chain_mod.read_configuration(fer, solver.chain)
        assert q[elbow] < -0.5, "the elbow bent off its limit"

    def test_a_solve_that_reaches_its_goal_is_not_retried(self, fer, modules, monkeypatch):
        """The retry is for a miss only: a reached goal costs one solve."""
        import bpy

        manager, solver = _pyroki_or_skip(modules, fer)
        modules("handlers").solve_rig(fer, force=True)
        bpy.context.view_layer.update()
        assert manager.get_solver(fer).last_result.converged, "precondition"
        calls = []
        real = manager.RigSolver._solve_pyroki

        def counted(self, *args, **kwargs):
            calls.append(1)
            return real(self, *args, **kwargs)

        monkeypatch.setattr(manager.RigSolver, "_solve_pyroki", counted)
        modules("handlers").solve_rig(fer, force=True)
        assert len(calls) == 1
