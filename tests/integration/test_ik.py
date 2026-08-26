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
