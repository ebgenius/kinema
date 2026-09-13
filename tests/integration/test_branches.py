"""Choosing among the configurations that reach a pose. Needs a real ``bpy``.

The claims:

* a search finds more than one way to reach a pose, and every one of them
  actually reaches it,
* applying one leaves the tool where it was,
* and live IK stays on the configuration that was chosen.

Steering a redundant arm's elbow lives in ``test_nullspace.py``.
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
def ik_ops(addon):
    return importlib.import_module(f"{addon.__name__}.ops.ik")


@pytest.fixture
def chain_mod(addon):
    return importlib.import_module(f"{addon.__name__}.solver.chain")


@pytest.fixture
def handlers(addon):
    return importlib.import_module(f"{addon.__name__}.handlers")


def _rig_from(fixture_dir, builder, name: str):
    import bpy

    assert "FINISHED" in bpy.ops.kinema.build_robot(
        filepath=str(fixture_dir / name)
    )
    rig = next(o for o in bpy.data.objects if builder.is_kinema_rig(o))
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.kinema.add_ik()
    return rig


@pytest.fixture
def arm6(addon, fixture_dir, clean_scene, builder):
    """Six joints: enough to reach a full pose, with nothing left over."""
    rig = _rig_from(fixture_dir, builder, "arm6.urdf")
    rig.kinema_ik_enabled = False
    return rig


def _q(builder, rig):
    return np.array([pb.rotation_euler[1] for pb in builder.joint_bones(rig)])


def _set_q(builder, rig, q):
    import bpy

    for pose_bone, value in zip(builder.joint_bones(rig), q, strict=True):
        pose_bone.rotation_euler[1] = value
    bpy.context.view_layer.update()


def _tool(builder, rig):
    import bpy

    name = rig.get(builder.PROP_TCP_BONE) or builder.TCP_BONE
    bpy.context.view_layer.update()
    return np.array(rig.pose.bones[name].matrix.translation)


REACHABLE = [0.4, -0.7, 1.1, 0.2, 0.5, 0.0]


class TestFindingSolutions:
    def test_more_than_one_way_to_reach_a_pose(self, arm6, builder, ik_ops):
        """The whole point: the solver was only ever offering one of them."""
        import bpy

        _set_q(builder, arm6, REACHABLE)
        bpy.ops.kinema.snap_ik()

        assert "FINISHED" in bpy.ops.kinema.find_solutions(seeds=30)
        assert ik_ops.solution_count(arm6) > 1

    def test_every_solution_actually_reaches_the_goal(
        self, arm6, builder, ik_ops, manager, addon
    ):
        """A configuration that misses is not an alternative, it is a failure.

        Orientation as well as position. Checking the point alone would pass a
        configuration that arrives there facing somewhere else, which a
        least-squares solve will happily return when the orientation is out of
        reach -- and this test would have called it a solution.
        """
        import bpy

        branches = importlib.import_module(f"{addon.__name__}.solver.branches")

        _set_q(builder, arm6, REACHABLE)
        bpy.ops.kinema.snap_ik()
        bpy.context.view_layer.update()
        solver = manager.get_solver(arm6)
        goal = solver.chain.forward(_q(builder, arm6))
        assert "FINISHED" in bpy.ops.kinema.find_solutions(seeds=30)
        solutions = ik_ops._read_solutions(arm6)
        # Without this, a search that found nothing skips the loop and the test
        # passes having checked no residual at all.
        assert solutions, "the search found nothing, so nothing below was checked"

        for values in solutions:
            position, orientation = branches.reach_error(
                solver.chain, np.array(values), goal
            )
            assert position < branches.REACH_TOLERANCE, values
            assert orientation < branches.ORIENTATION_TOLERANCE, values

    def test_the_search_leaves_the_pose_alone(self, arm6, builder):
        """It runs dozens of solves. None of them may be what you are left with."""
        import bpy

        _set_q(builder, arm6, REACHABLE)
        bpy.ops.kinema.snap_ik()
        before = _q(builder, arm6)

        bpy.ops.kinema.find_solutions(seeds=20)
        assert np.allclose(_q(builder, arm6), before, atol=1e-6)

    def test_applying_a_solution_keeps_the_tool_put(self, arm6, builder, ik_ops):
        """Different arm, same tool. That is what makes it an alternative."""
        import bpy

        _set_q(builder, arm6, REACHABLE)
        bpy.ops.kinema.snap_ik()
        goal = _tool(builder, arm6).copy()
        bpy.ops.kinema.find_solutions(seeds=30)

        moved = False
        for index in range(ik_ops.solution_count(arm6)):
            assert "FINISHED" in bpy.ops.kinema.apply_solution(index=index)
            assert np.allclose(_tool(builder, arm6), goal, atol=1e-3)
            moved = moved or not np.allclose(_q(builder, arm6), REACHABLE, atol=0.1)
        assert moved, "every solution was the configuration we started in"

    def test_the_arrows_cycle_and_wrap(self, arm6, builder, ik_ops):
        import bpy

        _set_q(builder, arm6, REACHABLE)
        bpy.ops.kinema.snap_ik()
        bpy.ops.kinema.find_solutions(seeds=30)
        count = ik_ops.solution_count(arm6)

        bpy.ops.kinema.apply_solution(index=0)
        bpy.ops.kinema.apply_solution(step=-1)
        assert arm6.kinema_active_solution == count - 1, "stepping back did not wrap"
        bpy.ops.kinema.apply_solution(step=1)
        assert arm6.kinema_active_solution == 0

    def test_an_index_past_the_end_is_refused(self, arm6, builder, ik_ops):
        """It used to raise IndexError out of execute rather than report."""
        import bpy

        _set_q(builder, arm6, REACHABLE)
        bpy.ops.kinema.snap_ik()
        bpy.ops.kinema.find_solutions(seeds=20)

        with pytest.raises(RuntimeError, match="no solution"):
            bpy.ops.kinema.apply_solution(index=ik_ops.solution_count(arm6) + 5)

    def test_reading_them_back_needs_no_solver(self, arm6, builder, ik_ops, manager):
        """The panel reads these on every redraw.

        Deriving the width from a solver meant building one on a cache miss --
        chain extraction on the UI thread, on the first redraw after anything
        invalidated the cache, which reopening a file does.
        """
        import bpy

        _set_q(builder, arm6, REACHABLE)
        bpy.ops.kinema.snap_ik()
        bpy.ops.kinema.find_solutions(seeds=20)
        expected = ik_ops.solution_count(arm6)
        assert expected > 0

        manager.invalidate()
        assert not manager._cache, "the cache did not actually clear"
        assert ik_ops.solution_count(arm6) == expected

    def test_a_moved_target_makes_them_stale(self, arm6, builder, ik_ops):
        """They describe one pose. Move it and they describe nothing."""
        import bpy
        from mathutils import Vector

        _set_q(builder, arm6, REACHABLE)
        bpy.ops.kinema.snap_ik()
        bpy.ops.kinema.find_solutions(seeds=20)
        assert not ik_ops.solutions_are_stale(arm6)

        goal = arm6.pose.bones[arm6.get(builder.PROP_IK_BONE)]
        goal.matrix.translation = goal.matrix.translation + Vector((0.0, 0.05, 0.0))
        bpy.context.view_layer.update()
        assert ik_ops.solutions_are_stale(arm6)

    def test_live_ik_stays_on_the_chosen_configuration(self, arm6, builder, ik_ops):
        """Otherwise picking one would be undone by the next viewport update."""
        import bpy

        _set_q(builder, arm6, REACHABLE)
        bpy.ops.kinema.snap_ik()
        bpy.ops.kinema.find_solutions(seeds=30)
        assert ik_ops.solution_count(arm6) > 1

        # The one furthest from where we are, so a slide back would show.
        start = _q(builder, arm6)
        stored = ik_ops._read_solutions(arm6)
        far = max(
            range(len(stored)),
            key=lambda i: float(np.max(np.abs(np.array(stored[i]) - start))),
        )
        bpy.ops.kinema.apply_solution(index=far)
        chosen = _q(builder, arm6)

        arm6.kinema_ik_enabled = True
        for _ in range(3):
            bpy.context.view_layer.update()
        assert np.allclose(_q(builder, arm6), chosen, atol=0.05), (
            "live IK slid off the configuration that was chosen"
        )


