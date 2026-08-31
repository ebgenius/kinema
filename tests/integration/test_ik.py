"""IK operators, live solving and baking. Run via ``dev.py test``."""

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
def handlers(addon):
    return importlib.import_module(f"{addon.__name__}.handlers")


@pytest.fixture
def manager(addon):
    return importlib.import_module(f"{addon.__name__}.solver.manager")


@pytest.fixture
def rig(addon, builder, fixture_dir, clean_scene):
    """A 6-DoF rig imported through the real operator, posed off its rest pose.

    Six joints are the minimum for an arbitrary position *and* orientation, so
    a full pose goal is actually achievable here -- on the 3-DoF arm3 fixture
    the solver can only ever trade position against orientation.
    """
    import bpy

    bpy.ops.kinema.import_urdf(filepath=str(fixture_dir / "arm6.urdf"))
    armature = next(o for o in bpy.data.objects if builder.is_kinema_rig(o))
    bpy.context.view_layer.objects.active = armature
    # Fold the arm off its rest pose: it starts stretched out at full reach,
    # where almost any target is unreachable, and "did adding IK disturb the
    # pose?" is only meaningful if there is a pose to disturb.
    armature.pose.bones["joint2"].rotation_euler[1] = -0.8
    armature.pose.bones["joint3"].rotation_euler[1] = 1.2
    armature.pose.bones["joint5"].rotation_euler[1] = 0.5
    bpy.context.view_layer.update()
    return armature


def _np4(matrix) -> np.ndarray:
    return np.array([[matrix[r][c] for c in range(4)] for r in range(4)])


def _tcp(rig, builder) -> np.ndarray:
    return _np4(rig.pose.bones[builder.TCP_BONE].matrix)


class TestAddIkTarget:
    def test_creates_a_control_bone(self, rig, builder):
        import bpy

        assert bpy.ops.kinema.add_ik() == {"FINISHED"}
        ik_name = rig.get(builder.PROP_IK_BONE)
        assert ik_name and ik_name in rig.pose.bones
        assert rig.kinema_ik_enabled

    def test_does_not_disturb_the_current_pose(self, rig, builder):
        """Regression: a new bone starts at its *rest* position, so enabling the
        solver before snapping the goal onto the tool dragged the whole arm back
        to the URDF rest pose and destroyed whatever the user had posed."""
        import bpy

        before = _tcp(rig, builder)
        bpy.ops.kinema.add_ik()
        bpy.context.view_layer.update()
        assert np.allclose(_tcp(rig, builder), before, atol=1e-6)

    def test_control_is_not_parented_into_the_chain(self, rig, builder):
        """An IK goal that moved with the arm it drives would chase its own tail."""
        import bpy

        bpy.ops.kinema.add_ik()
        ik_bone = rig.data.bones[rig.get(builder.PROP_IK_BONE)]
        assert ik_bone.parent is None or ik_bone.parent.name == builder.ROOT_BONE

    def test_control_lands_in_the_ik_collection(self, rig, builder):
        import bpy

        bpy.ops.kinema.add_ik()
        ik_name = rig.get(builder.PROP_IK_BONE)
        collections = [c.name for c in rig.data.bones[ik_name].collections]
        assert builder.COLLECTION_IK in collections


class TestLiveSolving:
    def test_moving_the_target_moves_the_tool(self, rig, builder, handlers):
        import bpy
        from mathutils import Vector

        bpy.ops.kinema.add_ik()
        ik_name = rig.get(builder.PROP_IK_BONE)

        matrix = rig.pose.bones[ik_name].matrix.copy()
        matrix.translation += Vector((0.0, 0.0, -0.08))
        rig.pose.bones[ik_name].matrix = matrix
        bpy.context.view_layer.update()
        handlers.solve_rig(rig, force=True)
        bpy.context.view_layer.update()

        goal = _np4(rig.pose.bones[ik_name].matrix)[:3, 3]
        reached = _tcp(rig, builder)[:3, 3]
        assert np.linalg.norm(goal - reached) < 5e-3

    def test_idle_updates_do_not_drift_or_loop(self, rig, builder, handlers):
        """The re-entrancy guard plus the cached target must make this a no-op."""
        import bpy

        bpy.ops.kinema.add_ik()
        before = _tcp(rig, builder)
        for _ in range(5):
            bpy.context.view_layer.update()
        assert np.allclose(_tcp(rig, builder), before, atol=1e-9)

    def test_solver_mode_off_stops_solving(self, rig, builder, handlers):
        import bpy
        from mathutils import Vector

        bpy.ops.kinema.add_ik()
        ik_name = rig.get(builder.PROP_IK_BONE)
        rig.kinema_solver_mode = "OFF"

        before = [b.rotation_euler[1] for b in builder.joint_bones(rig)]
        matrix = rig.pose.bones[ik_name].matrix.copy()
        matrix.translation += Vector((0.05, 0, 0))
        rig.pose.bones[ik_name].matrix = matrix
        bpy.context.view_layer.update()

        after = [b.rotation_euler[1] for b in builder.joint_bones(rig)]
        assert np.allclose(before, after)

    def test_numpy_backend_solves_too(self, rig, builder, handlers, manager):
        import bpy
        from mathutils import Vector

        bpy.ops.kinema.add_ik()
        ik_name = rig.get(builder.PROP_IK_BONE)
        rig.kinema_solver_mode = "NUMPY"

        matrix = rig.pose.bones[ik_name].matrix.copy()
        matrix.translation += Vector((0.0, 0.0, -0.05))
        rig.pose.bones[ik_name].matrix = matrix
        bpy.context.view_layer.update()
        handlers.solve_rig(rig, force=True)

        result = manager.get_solver(rig, ik_name).last_result
        assert result is not None
        assert result.backend == "NumPy"


    def test_pyroki_backend_is_used_by_default(self, rig, builder, handlers, manager):
        """Regression: loading the URDF with build_scene_graph=False left PyRoki
        unable to find the base link, so every rig silently fell back to NumPy
        while still reporting success."""
        import bpy

        bpy.ops.kinema.add_ik()
        ik_name = rig.get(builder.PROP_IK_BONE)
        solver = manager.get_solver(rig, ik_name)

        handlers.solve_rig(rig, force=True)
        assert solver.pyroki_error is None, (
            f"PyRoki unavailable for a local URDF rig: {solver.pyroki_error}"
        )
        assert solver.last_result.backend == "PyRoki"


class TestSolveBudget:
    def test_first_solve_is_not_timed(self, rig, builder, handlers, manager):
        """Regression: the first PyRoki solve includes JAX's JIT compile -- tens
        of seconds. Recording that as the solve time convinced the budget check
        the rig was hopelessly slow and throttled every later update."""
        import bpy

        bpy.ops.kinema.add_ik()
        handlers.reset(rig.name)
        manager.invalidate(rig.name)

        handlers.solve_rig(rig, force=True)
        assert handlers.last_solve_ms(rig) is None, "first solve must not be timed"

        handlers.solve_rig(rig, force=True)
        assert handlers.last_solve_ms(rig) is not None

    def test_throttling_recovers_instead_of_latching(self, handlers):
        """A slow rig must still solve one update in every _MAX_SKIPS + 1."""
        handlers.reset()
        handlers._last_duration["slowbot"] = 10.0  # way over any budget
        skipped = sum(handlers._over_budget("slowbot", 0.033) for _ in range(20))
        assert skipped < 20, "throttling latched off permanently"
        assert skipped == 20 - (20 // (handlers._MAX_SKIPS + 1))
        handlers.reset()


class TestBake:
    def test_bake_writes_joint_keyframes(self, rig, builder, handlers):
        import bpy
        from mathutils import Vector

        bpy.ops.kinema.add_ik()
        ik_name = rig.get(builder.PROP_IK_BONE)
        scene = bpy.context.scene

        base = rig.pose.bones[ik_name].matrix.copy()
        for frame, offset in ((1, (0, 0, 0)), (10, (0.0, 0.0, -0.08))):
            scene.frame_set(frame)
            matrix = base.copy()
            matrix.translation += Vector(offset)
            rig.pose.bones[ik_name].matrix = matrix
            rig.pose.bones[ik_name].keyframe_insert("location", frame=frame)

        assert bpy.ops.kinema.bake_ik(frame_start=1, frame_end=10, step=1) == {"FINISHED"}

        action = rig.animation_data.action
        paths = {c.data_path for c in _all_fcurves(action)}
        assert any("joint1" in p for p in paths)
        assert not rig.kinema_ik_enabled, "live IK should be off after baking"

    def test_baked_animation_plays_without_the_solver(self, rig, builder):
        """The whole point: a delivered .blend must not need Kinema to animate."""
        import bpy
        from mathutils import Vector

        bpy.ops.kinema.add_ik()
        ik_name = rig.get(builder.PROP_IK_BONE)
        scene = bpy.context.scene

        base = rig.pose.bones[ik_name].matrix.copy()
        for frame, offset in ((1, (0, 0, 0)), (10, (0.0, 0.0, -0.08))):
            scene.frame_set(frame)
            matrix = base.copy()
            matrix.translation += Vector(offset)
            rig.pose.bones[ik_name].matrix = matrix
            rig.pose.bones[ik_name].keyframe_insert("location", frame=frame)
        bpy.ops.kinema.bake_ik(frame_start=1, frame_end=10, step=1)

        rig.kinema_solver_mode = "OFF"
        scene.frame_set(1)
        first = _tcp(rig, builder)[:3, 3].copy()
        scene.frame_set(10)
        last = _tcp(rig, builder)[:3, 3].copy()
        assert np.linalg.norm(last - first) > 1e-3, "baked keys did not move the tool"


def ik_module():
    import importlib

    from ..conftest import EXTENSION_ID

    return importlib.import_module(f"{EXTENSION_ID}.ops.ik")


def manager_of(rig):
    import importlib

    from ..conftest import EXTENSION_ID

    manager = importlib.import_module(f"{EXTENSION_ID}.solver.manager")
    return manager.get_solver(rig)


def _tip_curves(rig) -> list:
    """Every F-curve animating the rig's IK tip."""
    action = rig.animation_data.action if rig.animation_data else None
    if action is None:
        return []
    return [c for c in _all_fcurves(action) if c.data_path == "kinema_ik_tip"]


def _add_ik_uncompiled(rig) -> None:
    """Add an IK target without paying JAX's ~10 s JIT compile.

    ``add_ik`` warms the solver up in whichever backend the rig is set to, so a
    test that only cares which *chain* got built asks for NumPy and skips the
    compile entirely. Twenty of those add up to minutes of wall clock and a lot
    of resident memory, for a code path the test never inspects.
    """
    import bpy

    rig.kinema_solver_mode = "NUMPY"
    bpy.ops.kinema.add_ik()


class TestIkTip:
    """Aiming the solver at a bone other than the TCP, and keyframing which one."""

    def test_default_targets_the_tcp(self, rig, builder, manager):
        _add_ik_uncompiled(rig)
        solver = manager.get_solver(rig)

        assert rig.kinema_ik_tip == -1
        assert solver.tip_bone == builder.TCP_BONE
        assert solver.chain.dof == len(builder.joint_bones(rig))

    def test_a_mid_chain_tip_shortens_the_chain(self, rig, builder, manager):
        """The Panda case from the demos: cut the gripper joints out of the chain."""
        _add_ik_uncompiled(rig)
        full = manager.get_solver(rig).chain.dof

        rig.kinema_ik_tip = 2  # joint3
        solver = manager.get_solver(rig)

        assert solver.tip_bone == "joint3"
        assert solver.chain.dof == 3
        assert solver.chain.dof < full
        assert solver.chain.bone_names == ["joint1", "joint2", "joint3"]

    def test_the_new_tip_is_what_reaches_the_goal(self, rig, builder, handlers):
        """joint6, not the TCP, must be what lands on the goal.

        Deliberately the last joint rather than a mid-chain one: its chain is
        still the full six joints, so an arbitrary position *and* orientation
        is actually achievable and the tolerance can be tight. A shorter chain
        would trade position against orientation and prove nothing about which
        frame the solver aimed at.
        """
        import bpy
        from mathutils import Vector

        bpy.ops.kinema.add_ik()
        bpy.ops.kinema.set_ik_tip(index=5)  # joint6

        ik_name = rig.get(builder.PROP_IK_BONE)
        tcp_name = rig.get(builder.PROP_TCP_BONE)
        separation = np.linalg.norm(
            _np4(rig.pose.bones["joint6"].matrix)[:3, 3]
            - _np4(rig.pose.bones[tcp_name].matrix)[:3, 3]
        )
        assert separation > 1e-2, "tip and TCP coincide, so this proves nothing"

        matrix = rig.pose.bones[ik_name].matrix.copy()
        matrix.translation += Vector((0.0, 0.0, -0.05))
        rig.pose.bones[ik_name].matrix = matrix
        bpy.context.view_layer.update()
        handlers.solve_rig(rig, force=True)
        bpy.context.view_layer.update()

        goal = _np4(rig.pose.bones[ik_name].matrix)[:3, 3]
        assert np.linalg.norm(goal - _np4(rig.pose.bones["joint6"].matrix)[:3, 3]) < 5e-3
        assert np.linalg.norm(goal - _np4(rig.pose.bones[tcp_name].matrix)[:3, 3]) > 1e-2

    def test_a_short_chain_still_drives_its_own_tip(self, rig, builder, handlers):
        """A 4-DoF chain cannot hit a full pose goal, but it must still try.

        The claim here is only that the solver moved the *new* tip toward the
        goal rather than the TCP -- position and orientation are being traded
        off against each other, so no tight tolerance is meaningful.
        """
        import bpy
        from mathutils import Vector

        _add_ik_uncompiled(rig)
        bpy.ops.kinema.set_ik_tip(index=3)  # joint4
        # Off, so the only solve is the one this test asks for. Left on, the
        # depsgraph handler solves inside view_layer.update() and "before" is
        # measured after the fact -- the same trap tools/demo/README.md records.
        rig.kinema_ik_enabled = False

        ik_name = rig.get(builder.PROP_IK_BONE)
        tcp_name = rig.get(builder.PROP_TCP_BONE)
        matrix = rig.pose.bones[ik_name].matrix.copy()
        matrix.translation += Vector((0.0, 0.0, -0.05))
        rig.pose.bones[ik_name].matrix = matrix
        bpy.context.view_layer.update()

        def error(bone_name):
            goal = _np4(rig.pose.bones[ik_name].matrix)[:3, 3]
            return np.linalg.norm(goal - _np4(rig.pose.bones[bone_name].matrix)[:3, 3])

        before = error("joint4")
        handlers.solve_rig(rig, force=True)
        bpy.context.view_layer.update()

        # No absolute tolerance: four joints against a six-dimensional goal are
        # trading position off against orientation, so what "close enough"
        # means is a property of the residual weights, not of this feature.
        assert error("joint4") < before, "the solver did not move the tip toward the goal"
        assert error("joint4") < error(tcp_name), "the solver was still chasing the TCP"

    def test_out_of_range_falls_back_to_the_tcp(self, rig, builder, manager):
        """A rig rebuilt with fewer joints must not leave the solver aimed at nothing."""
        rig.kinema_ik_tip = 99
        assert manager.tip_bone(rig) == builder.TCP_BONE

    def test_the_tip_is_keyframable(self, rig, builder, manager):
        """The whole reason the tip is an index and not a bone name.

        Blender animates integers and does not animate strings, so this is what
        lets a shot hand the goal from the wrist to the elbow part-way through.
        """
        import bpy

        _add_ik_uncompiled(rig)
        scene = bpy.context.scene

        scene.frame_set(1)
        rig.kinema_ik_tip = -1
        rig.keyframe_insert(data_path="kinema_ik_tip", frame=1)
        scene.frame_set(10)
        rig.kinema_ik_tip = 2
        rig.keyframe_insert(data_path="kinema_ik_tip", frame=10)

        scene.frame_set(1)
        assert rig.kinema_ik_tip == -1
        assert manager.get_solver(rig).tip_bone == builder.TCP_BONE

        scene.frame_set(10)
        assert rig.kinema_ik_tip == 2
        assert manager.get_solver(rig).tip_bone == "joint3"

    def test_changing_the_tip_alone_re_solves(self, rig, builder, handlers, manager):
        """Regression: the handler skips when the goal matrix has not moved.

        Moving the tip leaves the goal exactly where it was, so comparing the
        matrix alone would call that "nothing happened" and go on driving the
        old chain.
        """
        _add_ik_uncompiled(rig)
        handlers.solve_rig(rig, force=True)

        rig.kinema_ik_tip = 2
        assert handlers.solve_rig(rig) is True, "a tip change did not re-solve"

    def test_set_ik_tip_snaps_the_control(self, rig, builder):
        """Switching the tip must not jerk the arm, same rule as snap_ik."""
        import bpy

        _add_ik_uncompiled(rig)
        ik_name = rig.get(builder.PROP_IK_BONE)

        assert bpy.ops.kinema.set_ik_tip(index=2) == {"FINISHED"}
        bpy.context.view_layer.update()

        goal = _np4(rig.pose.bones[ik_name].matrix)[:3, 3]
        tip = _np4(rig.pose.bones["joint3"].matrix)[:3, 3]
        assert np.linalg.norm(goal - tip) < 1e-6

    def test_set_ik_tip_snaps_with_live_ik_running(self, rig, builder):
        """Regression: the snapshot was taken after the handler had moved the arm.

        Writing kinema_ik_tip drops the cached goal, so the next depsgraph
        update solves the *new* chain against the *old* goal and swings the
        arm. Reading the tip's pose after that captured the disturbed pose and
        baked the jerk in -- the opposite of what this operator promises.
        """
        import bpy

        _add_ik_uncompiled(rig)
        assert rig.kinema_ik_enabled, "this test is only meaningful with live IK on"

        bpy.context.view_layer.update()
        before = _np4(rig.pose.bones["joint3"].matrix)[:3, 3].copy()

        assert bpy.ops.kinema.set_ik_tip(index=2) == {"FINISHED"}
        bpy.context.view_layer.update()

        ik_name = rig.get(builder.PROP_IK_BONE)
        goal = _np4(rig.pose.bones[ik_name].matrix)[:3, 3]
        assert np.linalg.norm(goal - before) < 1e-6, (
            "the control snapped to a pose the solver had already disturbed"
        )

    def test_keyed_tips_step_rather_than_ramp(self, rig, builder, manager):
        """Regression: an interpolated index is a chain nobody asked for.

        Keyed with the default Bezier interpolation, a hand-off from the TCP
        (-1) to joint3 (2) passes through 0 and 1 on the frames between, so the
        solver drives two other chains on the way -- and each unseen link pays
        its own JAX compile.
        """
        import bpy

        _add_ik_uncompiled(rig)
        scene = bpy.context.scene
        bpy.context.view_layer.objects.active = rig

        scene.frame_set(1)
        rig.kinema_ik_tip = -1
        assert bpy.ops.kinema.key_ik_tip() == {"FINISHED"}
        scene.frame_set(40)
        rig.kinema_ik_tip = 2
        assert bpy.ops.kinema.key_ik_tip() == {"FINISHED"}

        seen = set()
        for frame in range(1, 41):
            scene.frame_set(frame)
            seen.add(rig.kinema_ik_tip)

        assert seen == {-1, 2}, f"the tip ramped through intermediate chains: {sorted(seen)}"

    def test_keying_forces_constant_interpolation(self, rig):
        import bpy

        _add_ik_uncompiled(rig)
        bpy.context.view_layer.objects.active = rig
        assert bpy.ops.kinema.key_ik_tip() == {"FINISHED"}

        curves = _tip_curves(rig)
        assert curves, "no F-curve was created for the tip"
        assert all(
            point.interpolation == "CONSTANT"
            for curve in curves
            for point in curve.keyframe_points
        )

    def test_set_ik_tip_normalises_a_hand_keyed_curve(self, rig):
        """A key made with the decorator dot gets fixed the next time we touch it."""
        import bpy

        _add_ik_uncompiled(rig)
        bpy.context.view_layer.objects.active = rig

        bpy.context.scene.frame_set(1)
        rig.keyframe_insert(data_path="kinema_ik_tip", frame=1)
        curve = _tip_curves(rig)[0]
        for point in curve.keyframe_points:
            point.interpolation = "BEZIER"

        bpy.ops.kinema.set_ik_tip(index=2)
        assert all(p.interpolation == "CONSTANT" for p in curve.keyframe_points)

    def test_set_ik_tip_keys_a_change_on_an_animated_tip(self, rig):
        """Regression: on an animated tip, a plain write does not survive.

        The operator ends with a view_layer.update(), which re-evaluates the
        animation and puts the curve's value straight back -- so the radio
        button in the Bones list appeared to do nothing at all once the tip had
        been keyed even once. Changing the target on a keyed channel has to key
        it there.
        """
        import bpy

        _add_ik_uncompiled(rig)
        scene = bpy.context.scene
        bpy.context.view_layer.objects.active = rig

        scene.frame_set(1)
        bpy.ops.kinema.set_ik_tip(index=2)
        bpy.ops.kinema.key_ik_tip()

        scene.frame_set(6)
        assert bpy.ops.kinema.set_ik_tip(index=-1) == {"FINISHED"}
        assert rig.kinema_ik_tip == -1, "the animation reverted the change"

        # And it stuck, rather than merely surviving until the next evaluation.
        scene.frame_set(1)
        assert rig.kinema_ik_tip == 2
        scene.frame_set(6)
        assert rig.kinema_ik_tip == -1

    def test_set_ik_tip_does_not_start_animating_an_unkeyed_tip(self, rig):
        """Only a channel that is *already* animated gets keyed."""
        import bpy

        _add_ik_uncompiled(rig)
        bpy.context.view_layer.objects.active = rig

        bpy.ops.kinema.set_ik_tip(index=2)
        assert _tip_curves(rig) == [], "changing the tip created animation unasked"
        assert rig.kinema_ik_tip == 2

    def test_set_ik_tip_rejects_a_bone_outside_the_chain(self, rig):
        """An ERROR report from bpy.ops surfaces in Python as a RuntimeError."""
        import bpy

        _add_ik_uncompiled(rig)
        with pytest.raises(RuntimeError, match="not part of this rig's chain"):
            bpy.ops.kinema.set_ik_tip(index=99)
        assert rig.kinema_ik_tip == -1, "a rejected tip must not be written"


class TestSetTcpInvalidates:
    def test_moving_the_tcp_rebuilds_the_chain(self, rig, builder, manager):
        """Regression: set_tcp re-roots the chain but the bone keeps its name,
        so get_solver's staleness check saw nothing wrong and kept driving the
        old chain. The demos only escaped it by calling set_tcp before add_ik."""
        import bpy

        _add_ik_uncompiled(rig)
        before = manager.get_solver(rig).chain.dof

        bpy.context.view_layer.objects.active = rig
        bpy.ops.object.mode_set(mode="POSE")
        rig.data.bones.active = rig.data.bones["joint3"]
        bpy.ops.kinema.set_tcp()
        bpy.ops.object.mode_set(mode="OBJECT")

        after = manager.get_solver(rig).chain.dof
        assert after == 3
        assert after < before


class TestBakeWithAKeyedTip:
    """Baking has to follow the tip, not the chain that was live when it started."""

    def _bake_with_tip_handoff(self, rig, builder):
        """Key the tip from joint3 (frames 1-5) to the TCP (frames 6-10), then bake."""
        import bpy
        from mathutils import Vector

        _add_ik_uncompiled(rig)
        scene = bpy.context.scene
        bpy.context.view_layer.objects.active = rig
        ik_name = rig.get(builder.PROP_IK_BONE)

        scene.frame_set(1)
        # Through the operator, so the control snaps onto each new tip and the
        # goal stays something the live chain can actually be asked for.
        bpy.ops.kinema.set_ik_tip(index=2)  # joint3
        bpy.ops.kinema.key_ik_tip()
        rig.pose.bones[ik_name].keyframe_insert("location", frame=1)

        scene.frame_set(6)
        bpy.ops.kinema.set_ik_tip(index=-1)  # back to the TCP
        bpy.ops.kinema.key_ik_tip()
        matrix = rig.pose.bones[ik_name].matrix.copy()
        matrix.translation += Vector((0.0, 0.0, -0.04))
        rig.pose.bones[ik_name].matrix = matrix
        rig.pose.bones[ik_name].keyframe_insert("location", frame=6)

        # A third key, so the goal keeps moving through the TCP window too.
        # Without it the goal is constant after frame 6 and every joint holds
        # still there, which would make "did the chain change?" unfalsifiable.
        scene.frame_set(10)
        matrix = rig.pose.bones[ik_name].matrix.copy()
        matrix.translation += Vector((0.0, -0.05, -0.02))
        rig.pose.bones[ik_name].matrix = matrix
        rig.pose.bones[ik_name].keyframe_insert("location", frame=10)

        assert bpy.ops.kinema.bake_ik(frame_start=1, frame_end=10, step=1) == {"FINISHED"}
        return {c.data_path for c in _all_fcurves(rig.animation_data.action)}

    def test_bakes_the_union_of_every_chain_in_the_range(self, rig, builder):
        """Regression: the bake captured one solver and one joint list up front.

        With a keyed tip the chain changes mid-bake, so joints that only become
        active later were never keyed -- and the stale solver overwrote the
        frame-change handler's correct solve on the way past.
        """
        paths = self._bake_with_tip_handoff(rig, builder)

        # joint3's chain is joints 1-3; the TCP's is all six. Every one of them
        # has to come out of the bake with a curve.
        for name in ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"):
            assert any(f'"{name}"' in path for path in paths), f"{name} was never keyed"

    def test_joints_beyond_the_tip_are_not_driven(self, rig, builder):
        """While the tip is joint3, the solver owns joints 1-3 and nothing else.

        This is the signature of the bug: the stale solver held the TCP's
        six-joint chain, so it drove joint4-6 through the joint3 window too.
        Checked by playing the baked curves back with the solver off, which is
        what a delivered .blend does.
        """
        import bpy

        self._bake_with_tip_handoff(rig, builder)
        scene = bpy.context.scene
        rig.kinema_solver_mode = "OFF"
        rig.kinema_ik_enabled = False

        def spread(bone_name, frames):
            values = []
            for frame in frames:
                scene.frame_set(frame)
                values.append(rig.pose.bones[bone_name].rotation_euler[1])
            return max(values) - min(values)

        # Frames 1-5 are the joint3 window; 6-10 are the TCP's.
        assert spread("joint5", range(1, 6)) < 1e-6, "a joint beyond the tip was driven"
        assert spread("joint5", range(6, 11)) > 1e-4, (
            "joint5 never moves at all, so the check above proves nothing"
        )

    def test_a_static_tip_still_bakes_its_own_chain(self, rig, builder):
        """The scan is skipped when the tip is not animated; that path still works."""
        import bpy
        from mathutils import Vector

        _add_ik_uncompiled(rig)
        bpy.ops.kinema.set_ik_tip(index=2)  # joint3, never keyed
        ik_name = rig.get(builder.PROP_IK_BONE)
        scene = bpy.context.scene

        base = rig.pose.bones[ik_name].matrix.copy()
        for frame, offset in ((1, (0, 0, 0)), (10, (0.0, 0.0, -0.03))):
            scene.frame_set(frame)
            matrix = base.copy()
            matrix.translation += Vector(offset)
            rig.pose.bones[ik_name].matrix = matrix
            rig.pose.bones[ik_name].keyframe_insert("location", frame=frame)

        assert bpy.ops.kinema.bake_ik(frame_start=1, frame_end=10, step=1) == {"FINISHED"}
        paths = {c.data_path for c in _all_fcurves(rig.animation_data.action)}
        assert any('"joint3"' in path for path in paths)

    def test_the_union_ignores_the_frame_the_user_is_sitting_on(self, rig, builder):
        """Regression: the union was seeded from scene.frame_current.

        That frame need not be inside the bake range. Sitting on a frame whose
        tip is the TCP and baking a range that only ever targets joint3 pulled
        joints 4-6 into the union, cleared their existing curves, and wrote keys
        for joints never active in the range.
        """
        import bpy

        _add_ik_uncompiled(rig)
        scene = bpy.context.scene
        bpy.context.view_layer.objects.active = rig

        # joint3 for frames 1-10; the TCP only from frame 50 on.
        scene.frame_set(1)
        bpy.ops.kinema.set_ik_tip(index=2)
        bpy.ops.kinema.key_ik_tip()
        scene.frame_set(50)
        bpy.ops.kinema.set_ik_tip(index=-1)
        bpy.ops.kinema.key_ik_tip()

        # Park on frame 50, outside the range about to be baked.
        scene.frame_set(50)
        assert manager_of(rig).tip_bone == builder.TCP_BONE

        bones = ik_module().KINEMA_OT_bake_ik._bones_over_range(
            rig, scene, range(1, 11), rig.get(builder.PROP_IK_BONE)
        )
        assert bones == ["joint1", "joint2", "joint3"], (
            f"the current frame leaked into the union: {bones}"
        )

    def test_a_shared_action_does_not_cross_contaminate(self, rig, builder, fixture_dir):
        """One Action can hold several slots, one per rig.

        Matching on data path alone would find the *other* rig's tip curve --
        enough to auto-key this rig off someone else's animation and rewrite
        the other slot's interpolation.
        """
        import bpy

        _add_ik_uncompiled(rig)
        bpy.context.view_layer.objects.active = rig
        bpy.ops.kinema.set_ik_tip(index=2)
        bpy.ops.kinema.key_ik_tip()
        action = rig.animation_data.action
        if not hasattr(action, "layers"):
            pytest.skip("this Blender stores actions without slots")

        # A second rig sharing the very same action, in a slot of its own.
        bpy.ops.kinema.import_urdf(filepath=str(fixture_dir / "arm6.urdf"))
        other = next(
            o for o in bpy.data.objects if builder.is_kinema_rig(o) and o is not rig
        )
        other.animation_data_create()
        other.animation_data.action = action
        other.animation_data.action_slot = action.slots.new("OBJECT", "other")
        assert (
            other.animation_data.action_slot.handle
            != rig.animation_data.action_slot.handle
        ), "both rigs ended up in one slot, so this proves nothing"

        # The add-on's own lookup, not this file's helper: slot isolation is
        # exactly the thing under test.
        tip_curves = ik_module()._tip_curves
        assert tip_curves(other) == [], (
            "the other rig's slot picked up this rig's tip curve"
        )
        assert tip_curves(rig), "this rig lost sight of its own tip curve"

    def test_bake_restores_live_ik_it_turned_off(self, rig, builder):
        import bpy

        _add_ik_uncompiled(rig)
        assert rig.kinema_ik_enabled
        bpy.ops.kinema.bake_ik(
            frame_start=1, frame_end=3, step=1, disable_live_ik=False
        )
        assert rig.kinema_ik_enabled, "the bake left live IK switched off"


class TestSolverIdentity:
    """Blender reuses a deleted object's name for the next one to claim it."""

    def test_a_new_rig_reusing_the_name_gets_its_own_solver(
        self, addon, builder, manager, fixture_dir, clean_scene
    ):
        """Regression: the caches were keyed on the object name alone.

        Delete a rig and import another, and Blender hands the new object the
        old name -- which handed it the old robot's cached chain and, if the
        link name and joint count matched, its compiled JAX solver too.
        """
        import bpy

        bpy.ops.kinema.import_urdf(filepath=str(fixture_dir / "arm6.urdf"))
        first = next(o for o in bpy.data.objects if builder.is_kinema_rig(o))
        bpy.context.view_layer.objects.active = first
        _add_ik_uncompiled(first)
        name, first_identity = first.name, manager.rig_identity(first)
        assert manager.get_solver(first) is not None

        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)

        # A different robot, which Blender will name the same thing.
        bpy.ops.kinema.import_urdf(filepath=str(fixture_dir / "arm3.urdf"))
        second = next(o for o in bpy.data.objects if builder.is_kinema_rig(o))
        bpy.context.view_layer.objects.active = second
        second.name = name
        _add_ik_uncompiled(second)

        assert manager.rig_identity(second) != first_identity, (
            "session_uid was reused, so this test cannot detect the bug"
        )
        solver = manager.get_solver(second)
        assert solver.identity == manager.rig_identity(second)
        assert solver.chain.dof == 3, "inherited the deleted rig's six-joint chain"


class TestHandlerSuspension:
    def test_suspended_stops_the_live_solve(self, rig, builder, handlers):
        """kinema_ik_enabled cannot do this job: it is animatable itself, so a
        curve on it puts live IK back on at every frame_set."""
        import bpy
        from mathutils import Vector

        _add_ik_uncompiled(rig)
        ik_name = rig.get(builder.PROP_IK_BONE)
        before = [b.rotation_euler[1] for b in builder.joint_bones(rig)]

        with handlers.suspended():
            matrix = rig.pose.bones[ik_name].matrix.copy()
            matrix.translation += Vector((0.0, 0.0, -0.05))
            rig.pose.bones[ik_name].matrix = matrix
            bpy.context.view_layer.update()
            after = [b.rotation_euler[1] for b in builder.joint_bones(rig)]

        assert np.allclose(before, after), "the handler solved while suspended"
        assert handlers._solving is False, "suspension leaked past the block"

        # ...and live IK works again: a fresh move of the target, not a bare
        # update(), because an unchanged depsgraph fires no handler at all.
        matrix = rig.pose.bones[ik_name].matrix.copy()
        matrix.translation += Vector((0.0, 0.0, -0.03))
        rig.pose.bones[ik_name].matrix = matrix
        bpy.context.view_layer.update()
        assert not np.allclose(
            before, [b.rotation_euler[1] for b in builder.joint_bones(rig)]
        ), "the handler stayed suspended after the block"

    def test_bake_survives_an_animated_live_ik_flag(self, rig, builder):
        """The property being keyed on must not reintroduce the double solve."""
        import bpy

        _add_ik_uncompiled(rig)
        scene = bpy.context.scene
        bpy.context.view_layer.objects.active = rig
        rig.kinema_ik_enabled = True
        rig.keyframe_insert(data_path="kinema_ik_enabled", frame=1)

        assert bpy.ops.kinema.bake_ik(
            frame_start=1, frame_end=5, step=1, disable_live_ik=False
        ) == {"FINISHED"}
        scene.frame_set(1)


class TestRemoveIk:
    def test_removes_the_bone_and_flags(self, rig, builder):
        import bpy

        bpy.ops.kinema.add_ik()
        ik_name = rig.get(builder.PROP_IK_BONE)
        assert bpy.ops.kinema.remove_ik() == {"FINISHED"}
        assert ik_name not in rig.data.bones
        assert not rig.kinema_ik_enabled


def _all_fcurves(action):
    legacy = getattr(action, "fcurves", None)
    if legacy is not None:
        return list(legacy)
    curves = []
    for layer in getattr(action, "layers", ()):
        for strip in getattr(layer, "strips", ()):
            for channelbag in getattr(strip, "channelbags", ()):
                curves.extend(channelbag.fcurves)
    return curves
