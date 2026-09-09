"""Choosing among the configurations that reach a pose. Needs a real ``bpy``.

The claims:

* a search finds more than one way to reach a pose, and every one of them
  actually reaches it,
* applying one leaves the tool where it was,
* live IK stays on the configuration that was chosen,
* the elbow target steers a redundant arm without giving up the tool,
* and it is not offered on an arm with no freedom to steer.
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
        bpy.ops.kinema.find_solutions(seeds=30)

        for values in ik_ops._read_solutions(arm6):
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


@pytest.fixture
def arm7(addon, fixture_dir, clean_scene, builder):
    """Seven joints: one degree of freedom left over after the tool pose."""
    rig = _rig_from(fixture_dir, builder, "arm7.urdf")
    rig.kinema_ik_enabled = False
    return rig


REDUNDANT = [0.3, -0.8, 0.5, 1.0, 0.2, 0.6, 0.0]


class TestElbowTarget:
    def test_a_six_axis_arm_is_refused(self, arm6):
        """Six joints against a six-DoF pose leaves nothing to steer."""
        import bpy

        with pytest.raises(RuntimeError, match="freedom left over"):
            bpy.ops.kinema.add_elbow_target()

    def test_it_is_not_offered_on_a_six_axis_arm(self, arm6, builder):
        """The panel hides it rather than offering a control that does nothing."""
        assert not arm6.get(builder.PROP_ELBOW_BONE)

    def test_a_redundant_arm_gets_one(self, arm7, builder, manager):
        import bpy

        solver = manager.get_solver(arm7)
        assert solver.chain.dof == 7

        assert "FINISHED" in bpy.ops.kinema.add_elbow_target()
        name = arm7.get(builder.PROP_ELBOW_BONE)
        assert name and name in arm7.pose.bones
        # Parented to Root, never into the chain: a control that moved with the
        # arm it steers would chase itself.
        assert arm7.data.bones[name].parent.name == builder.ROOT_BONE

    def test_dragging_it_moves_the_elbow_and_not_the_tool(
        self, arm7, builder, manager
    ):
        """The claim the whole control rests on.

        A redundant arm can reconfigure around a tool that stands still, and
        the elbow goal is weighted far below the tool precisely so it can only
        move where that freedom already exists.
        """
        import bpy
        from mathutils import Vector

        _set_q(builder, arm7, REDUNDANT)
        bpy.ops.kinema.snap_ik()
        assert "FINISHED" in bpy.ops.kinema.add_elbow_target()
        arm7.kinema_ik_enabled = True
        bpy.context.view_layer.update()

        joint = arm7.get(builder.PROP_ELBOW_JOINT)
        tool_before = _tool(builder, arm7).copy()
        elbow_before = np.array(arm7.pose.bones[joint].matrix.translation)

        elbow = arm7.pose.bones[arm7.get(builder.PROP_ELBOW_BONE)]
        elbow.matrix.translation = elbow.matrix.translation + Vector((0.0, 0.6, 0.3))
        for _ in range(3):
            bpy.context.view_layer.update()

        tool_after = _tool(builder, arm7)
        elbow_after = np.array(arm7.pose.bones[joint].matrix.translation)

        moved = float(np.linalg.norm(elbow_after - elbow_before))
        drift = float(np.linalg.norm(tool_after - tool_before))
        assert moved > 0.02, f"the elbow did not follow ({moved * 1000:.2f} mm)"
        assert drift < 1e-3, f"the tool gave way ({drift * 1000:.3f} mm)"

    def test_zero_strength_stops_it_steering(self, arm7, builder):
        """The slider has to actually mean something at both ends."""
        import bpy
        from mathutils import Vector

        _set_q(builder, arm7, REDUNDANT)
        bpy.ops.kinema.snap_ik()
        bpy.ops.kinema.add_elbow_target()
        arm7.kinema_elbow_strength = 0.0
        arm7.kinema_ik_enabled = True
        bpy.context.view_layer.update()

        joint = arm7.get(builder.PROP_ELBOW_JOINT)
        before = np.array(arm7.pose.bones[joint].matrix.translation)

        elbow = arm7.pose.bones[arm7.get(builder.PROP_ELBOW_BONE)]
        elbow.matrix.translation = elbow.matrix.translation + Vector((0.0, 0.6, 0.3))
        for _ in range(3):
            bpy.context.view_layer.update()

        after = np.array(arm7.pose.bones[joint].matrix.translation)
        assert np.allclose(after, before, atol=1e-3)

    def test_removing_it_leaves_plain_ik(self, arm7, builder):
        import bpy

        bpy.ops.kinema.add_elbow_target()
        name = arm7.get(builder.PROP_ELBOW_BONE)

        assert "FINISHED" in bpy.ops.kinema.remove_elbow_target()
        assert not arm7.get(builder.PROP_ELBOW_BONE)
        assert name not in arm7.pose.bones
