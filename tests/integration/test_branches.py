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

    def test_a_search_selects_the_configuration_the_arm_stands_in(
        self, arm6, builder, ik_ops
    ):
        """Searching again isn't a way to be moved on to another one."""
        import bpy

        _set_q(builder, arm6, REACHABLE)
        bpy.ops.kinema.snap_ik()
        assert "FINISHED" in bpy.ops.kinema.find_solutions(seeds=30)
        stored = ik_ops._read_solutions(arm6)
        assert len(stored) > 1, "precondition: more than one to choose from"
        chosen = np.array(stored[arm6.kinema_active_solution])
        np.testing.assert_allclose(chosen, REACHABLE, atol=1e-3)
        np.testing.assert_allclose(_q(builder, arm6), REACHABLE, atol=1e-6)

    def _moved(self, arm6, builder):
        """The target moved 5 cm off the tool, the arm left where it was."""
        import bpy
        from mathutils import Vector

        goal = arm6.pose.bones[arm6.get(builder.PROP_IK_BONE)]
        goal.matrix.translation = goal.matrix.translation + Vector((0.0, 0.05, 0.0))
        bpy.context.view_layer.update()

    def test_searching_again_once_the_target_moved_leaves_the_arm(
        self, arm6, builder, ik_ops
    ):
        import bpy

        _set_q(builder, arm6, REACHABLE)
        bpy.ops.kinema.snap_ik()
        bpy.ops.kinema.find_solutions(seeds=30)
        self._moved(arm6, builder)
        assert ik_ops.solutions_are_stale(arm6), "precondition: stale"
        before = _q(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.find_solutions(seeds=30)
        assert not ik_ops.solutions_are_stale(arm6)
        np.testing.assert_allclose(_q(builder, arm6), before, atol=1e-6)

    def test_an_arrow_once_the_target_moved_searches_again_then_steps(
        self, arm6, builder, ik_ops
    ):
        """Found again for the goal as it is, from where the arm stands: the step is
        from the configuration nearest it."""
        import bpy

        _set_q(builder, arm6, REACHABLE)
        bpy.ops.kinema.snap_ik()
        bpy.ops.kinema.find_solutions(seeds=30)
        self._moved(arm6, builder)
        before = _q(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.apply_solution(step=1)
        assert not ik_ops.solutions_are_stale(arm6), "found again for the moved target"
        stored = [np.array(q) for q in ik_ops._read_solutions(arm6)]
        assert len(stored) > 1, "precondition: more than one to step between"
        index = arm6.kinema_active_solution
        nearest = int(np.argmin([np.max(np.abs(q - before)) for q in stored]))
        assert index == (nearest + 1) % len(stored)
        np.testing.assert_allclose(_q(builder, arm6), stored[index], atol=1e-5)

    def test_live_ik_doesnt_move_the_arm_during_a_search(self, arm6, builder, ik_ops):
        """The target moved, live IK not yet run: the search leaves the arm where it
        stands, rather than letting an update in it solve for the moved target. Called
        as the operator calls it, to read the arm before the update an operator's end
        brings, after which live IK rightly follows the target."""
        import bpy
        from mathutils import Vector

        _set_q(builder, arm6, REACHABLE)
        bpy.ops.kinema.snap_ik()
        arm6.kinema_ik_enabled = True
        bpy.context.view_layer.update()
        goal = arm6.pose.bones[arm6.get(builder.PROP_IK_BONE)]
        goal.location += Vector((0.0, 0.0, 0.03))
        assert ik_ops.search_solutions(bpy.context, arm6, seeds=20).done
        np.testing.assert_allclose(_q(builder, arm6), REACHABLE, atol=1e-6)

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




#: Away from REACHABLE, reachable: where the target goes for a configuration switch.
ELSEWHERE = [-1.2, -0.3, 1.6, -0.8, 1.1, 0.6]


class TestAutoConfiguration:
    """Live IK that can't reach the target from the configuration it is in finds
    another that does; a target out of reach is left as it is (#103)."""

    @pytest.fixture
    def handlers(self, addon):
        return importlib.import_module(f"{addon.__name__}.handlers")

    @pytest.fixture
    def search(self, manager, monkeypatch):
        """manager.find_solutions, counted."""
        calls = []
        real = manager.find_solutions

        def counted(*args, **kwargs):
            calls.append(kwargs.get("seeds"))
            return real(*args, **kwargs)

        monkeypatch.setattr(manager, "find_solutions", counted)
        return calls

    def _miss_once(self, manager, monkeypatch):
        """The next live solve misses and writes nothing, as tracking lost leaves the
        arm: stuck in the configuration it was in."""
        from importlib import import_module

        base = import_module(manager.__name__.rsplit(".", 1)[0] + ".base")
        real = manager.RigSolver.solve
        state = {"missed": False}

        def solve(self, rig, mode=manager.MODE_PYROKI, live=False):
            if not state["missed"]:
                state["missed"] = True
                q = np.zeros(self.chain.dof)
                return base.SolveResult(q, False, 0.03, 0.8, 100, "test")
            return real(self, rig, mode, live)

        monkeypatch.setattr(manager.RigSolver, "solve", solve)

    def _target_at(self, builder, rig, q):
        """The IK target where the tool is with the joints at ``q``; the arm left
        where it was."""
        import bpy

        before = _q(builder, rig)
        _set_q(builder, rig, q)
        tip = rig.pose.bones[rig.get(builder.PROP_TCP_BONE) or builder.TCP_BONE]
        goal = tip.matrix.copy()
        _set_q(builder, rig, before)
        rig.pose.bones[rig.get(builder.PROP_IK_BONE)].matrix = goal
        bpy.context.view_layer.update()

    def test_a_miss_the_arm_could_reach_another_way_switches_configuration(
        self, arm6, builder, manager, ik_ops, handlers, search, monkeypatch
    ):
        import bpy

        _set_q(builder, arm6, REACHABLE)
        self._target_at(builder, arm6, ELSEWHERE)
        assert ik_ops.goal_off_tool(arm6), "precondition: the target elsewhere"
        arm6.kinema_auto_configuration = True
        self._miss_once(manager, monkeypatch)
        handlers.solve_rig(arm6, force=True, live=True)
        bpy.context.view_layer.update()
        assert search == [ik_ops.AUTO_SEEDS]
        assert not ik_ops.goal_off_tool(arm6), "the tool on the target"
        assert ik_ops.switched_since_recorded(arm6), "for the quick bar to say so"
        assert ik_ops.solution_count(arm6) > 0, "what was found, to step through"
        stored = ik_ops._read_solutions(arm6)
        np.testing.assert_allclose(
            stored[arm6.kinema_active_solution], _q(builder, arm6), atol=1e-3
        )

    def test_off_it_leaves_the_miss_alone(
        self, arm6, builder, manager, ik_ops, handlers, search, monkeypatch
    ):
        import bpy

        _set_q(builder, arm6, REACHABLE)
        self._target_at(builder, arm6, ELSEWHERE)
        self._miss_once(manager, monkeypatch)
        handlers.solve_rig(arm6, force=True, live=True)
        bpy.context.view_layer.update()
        assert search == []
        assert ik_ops.goal_off_tool(arm6)
        np.testing.assert_allclose(_q(builder, arm6), REACHABLE, atol=1e-6)

    def test_not_while_playing_or_rendering(
        self, arm6, builder, manager, ik_ops, handlers, search, monkeypatch
    ):
        """Frames play as they were keyed: a search is for a drag."""
        _set_q(builder, arm6, REACHABLE)
        self._target_at(builder, arm6, ELSEWHERE)
        arm6.kinema_auto_configuration = True
        self._miss_once(manager, monkeypatch)
        handlers.solve_rig(arm6, force=True, live=False)
        assert search == []

    def test_out_of_reach_nothing_changes_and_it_waits_to_search_again(
        self, arm6, builder, manager, ik_ops, handlers, search, monkeypatch
    ):
        """No configuration reaches it: the arm stays as live IK left it, and a
        target that hasn't moved on isn't searched for again."""
        import bpy
        from mathutils import Vector

        _set_q(builder, arm6, REACHABLE)
        bpy.ops.kinema.snap_ik()
        arm6.kinema_auto_configuration = True
        arm6.kinema_solver_mode = manager.MODE_NUMPY
        goal = arm6.pose.bones[arm6.get(builder.PROP_IK_BONE)]
        goal.location += Vector((0.0, 0.0, 5.0))
        bpy.context.view_layer.update()
        handlers.solve_rig(arm6, force=True, live=True)
        assert search == [ik_ops.AUTO_SEEDS]
        assert ik_ops.auto_found_nothing(arm6), "for the quick bar to say it's out of reach"
        stood = _q(builder, arm6)
        handlers.solve_rig(arm6, force=True, live=True)
        assert search == [ik_ops.AUTO_SEEDS], "the same miss isn't searched twice"
        # Live IK edges on toward it each solve; no other configuration was taken.
        np.testing.assert_allclose(_q(builder, arm6), stood, atol=0.01)

        monkeypatch.setattr(ik_ops, "AUTO_INTERVAL", 0.0)
        handlers.solve_rig(arm6, force=True, live=True)
        assert len(search) == 1, "nor once the time is up, while it hasn't moved"
        goal.location += Vector((0.0, 0.02, 0.0))
        bpy.context.view_layer.update()
        handlers.solve_rig(arm6, force=True, live=True)
        assert len(search) == 2, "moved on 2 cm: searched again"

    def test_a_turn_counts_on_a_scaled_robot_too(self, ik_ops):
        """The goal's matrix carries the robot's scale; a turn past AUTO_TURN is one
        at any scale, and a turn short of it isn't."""
        from mathutils import Matrix

        def turned(degrees: float, scale: float) -> np.ndarray:
            return np.array(Matrix.Rotation(np.radians(degrees), 4, "Z")) @ np.diag(
                [scale, scale, scale, 1.0]
            )

        for scale in (0.5, 1.0, 2.0):
            last = turned(0.0, scale)
            assert ik_ops._moved_on(last, turned(10.0, scale)), scale
            assert not ik_ops._moved_on(last, turned(3.0, scale)), scale
