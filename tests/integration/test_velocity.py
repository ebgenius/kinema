"""Joint velocity limits: read at import, checked at every frame. Needs a real ``bpy``.

The claims worth holding onto:

* the description's ``<limit velocity>`` lands on the joint bone,
* a keyed joint is measured from its own curve, exactly, however the frame was
  reached,
* a joint live IK drives is measured against the frame before, as the handler
  saw it *after* solving,
* a joint pressed against its stop is not racing, whatever its curve does,
* Ignore Velocity Limits silences the panel and the viewport, not the numbers,
* a joint_limits.yaml overrides the description joint by joint.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from ..conftest import EXTENSION_ID, requires_bpy

pytestmark = requires_bpy

FPS = 24


@pytest.fixture
def builder(addon):
    return importlib.import_module(f"{addon.__name__}.rig.builder")


@pytest.fixture
def velocity(addon):
    return importlib.import_module(f"{addon.__name__}.ops.velocity")


@pytest.fixture
def overlay(addon):
    return importlib.import_module(f"{addon.__name__}.ui.overlay")


@pytest.fixture
def handlers(addon):
    return importlib.import_module(f"{addon.__name__}.handlers")


@pytest.fixture
def arm6(addon, builder, fixture_dir, clean_scene):
    """arm6: joints 1-3 limited to 3.15 rad/s, joints 4-6 to 3.2."""
    import bpy

    scene = bpy.context.scene
    scene.render.fps, scene.render.fps_base = FPS, 1.0
    scene.frame_set(1)
    assert "FINISHED" in bpy.ops.kinema.build_robot(
        filepath=str(fixture_dir / "arm6.urdf")
    )
    rig = next(o for o in bpy.data.objects if builder.is_kinema_rig(o))
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    yield rig
    if rig.animation_data is not None:
        rig.animation_data_clear()
    scene.frame_set(1)


def _key(rig, bone: str, frames_values) -> None:
    """Key one joint's rotation at each (frame, value), with straight lines between."""
    pose_bone = rig.pose.bones[bone]
    for frame, value in frames_values:
        pose_bone.rotation_euler[1] = value
        pose_bone.keyframe_insert("rotation_euler", index=1, frame=frame)
    _linear(rig)


def _curves(rig):
    ik = importlib.import_module(f"{EXTENSION_ID}.ops.ik")
    return [curve for container in ik.own_fcurve_containers(rig) for curve in container]


def _linear(rig) -> None:
    for curve in _curves(rig):
        for point in curve.keyframe_points:
            point.interpolation = "LINEAR"
        curve.update()


def _by_joint(velocity, rig):
    import bpy

    return {r.joint: r for r in velocity.readings(rig, bpy.context.scene)}


class TestImport:
    def test_bones_carry_the_description_velocity(self, arm6, builder):
        bones = arm6.data.bones
        assert bones["joint1"][builder.PROP_VELOCITY] == pytest.approx(3.15)
        assert bones["joint6"][builder.PROP_VELOCITY] == pytest.approx(3.2)

    def test_every_limited_joint_is_read(self, arm6, velocity):
        assert set(_by_joint(velocity, arm6)) == {f"joint{i}" for i in range(1, 7)}

    def test_mjcf_has_no_velocity_limits(self, addon, builder, fixture_dir, clean_scene, velocity):
        """MJCF has no such attribute: the rig says so rather than inventing one."""
        import bpy

        assert "FINISHED" in bpy.ops.kinema.build_robot(
            filepath=str(fixture_dir / "mjcf_arm.xml")
        )
        rig = next(o for o in bpy.data.objects if builder.is_kinema_rig(o))
        assert builder.joint_bones(rig), "precondition: the MJCF rig has joints"
        assert velocity.readings(rig, bpy.context.scene) == []


class TestKeyedJoints:
    def test_a_fast_move_is_flagged(self, arm6, velocity):
        import bpy

        _key(arm6, "joint1", [(1, 0.0), (2, 0.5)])
        bpy.context.scene.frame_set(2)

        readings = _by_joint(velocity, arm6)
        assert readings["joint1"].speed == pytest.approx(0.5 * FPS, rel=1e-5)
        assert readings["joint1"].over
        assert readings["joint2"].speed == pytest.approx(0.0)
        assert [r.joint for r in velocity.warnings(arm6, bpy.context.scene)] == ["joint1"]

    def test_a_slow_move_is_not(self, arm6, velocity):
        import bpy

        _key(arm6, "joint1", [(1, 0.0), (25, 0.5)])
        bpy.context.scene.frame_set(13)

        reading = _by_joint(velocity, arm6)["joint1"]
        assert reading.speed == pytest.approx(0.5, rel=1e-4)
        assert not reading.over

    def test_the_frame_rate_sets_the_speed(self, arm6, velocity):
        """The same keys at 12 fps take twice as long, so move half as fast."""
        import bpy

        _key(arm6, "joint1", [(1, 0.0), (2, 0.5)])
        bpy.context.scene.render.fps = 12
        bpy.context.scene.frame_set(2)
        assert _by_joint(velocity, arm6)["joint1"].speed == pytest.approx(6.0, rel=1e-5)

    def test_a_jump_still_measures_a_keyed_joint(self, arm6, velocity):
        """The curve says where the joint was a frame ago; no need to have been there."""
        import bpy

        scene = bpy.context.scene
        _key(arm6, "joint1", [(1, 0.0), (40, 0.0), (41, 0.5)])
        scene.frame_set(1)
        scene.frame_set(41)

        assert _by_joint(velocity, arm6)["joint1"].over

    def test_a_joint_against_its_stop_is_not_racing(self, arm6, velocity):
        """joint2 stops at 2.5 rad; a curve that runs on past it moves nothing."""
        import bpy

        _key(arm6, "joint2", [(1, 2.5), (2, 20.0)])
        bpy.context.scene.frame_set(2)
        assert arm6.pose.bones["joint2"].rotation_euler[1] == pytest.approx(20.0), (
            "precondition: the channel itself moved, only the constraint held it"
        )

        reading = _by_joint(velocity, arm6)["joint2"]
        assert reading.speed == pytest.approx(0.0, abs=1e-5)
        assert not reading.over


class TestIgnore:
    def test_ignoring_silences_the_warnings(self, arm6, velocity, overlay):
        import bpy

        _key(arm6, "joint1", [(1, 0.0), (2, 0.5)])
        bpy.context.scene.frame_set(2)
        assert velocity.warnings(arm6, bpy.context.scene), "precondition: over the limit"
        assert [rig for rig, _ in overlay.flagged(bpy.context)] == [arm6]

        arm6.kinema_ignore_velocity = True
        assert velocity.warnings(arm6, bpy.context.scene) == []
        assert overlay.flagged(bpy.context) == []

    def test_ignoring_keeps_the_numbers(self, arm6, velocity):
        """The panel still lists every joint, greyed; only the alarm is off."""
        import bpy

        _key(arm6, "joint1", [(1, 0.0), (2, 0.5)])
        bpy.context.scene.frame_set(2)
        arm6.kinema_ignore_velocity = True
        assert _by_joint(velocity, arm6)["joint1"].over


class TestLiveIk:
    @pytest.fixture
    def live(self, arm6, builder):
        """IK on, the target keyed through a sudden jump: frame 1, 2, then still."""
        import bpy
        from mathutils import Vector

        arm6.pose.bones["joint2"].rotation_euler[1] = -0.8
        arm6.pose.bones["joint3"].rotation_euler[1] = 1.2
        arm6.pose.bones["joint5"].rotation_euler[1] = 0.5
        bpy.context.view_layer.update()
        # NumPy: this is about when the pose is read, not how it is solved,
        # and the JAX compile would cost ten seconds a test for nothing.
        arm6.kinema_solver_mode = "NUMPY"
        bpy.ops.kinema.add_ik()
        ik_name = arm6.get(builder.PROP_IK_BONE)
        goal = arm6.pose.bones[ik_name]
        base = goal.matrix.copy()
        for frame, offset in ((1, 0.0), (2, 0.12), (3, 0.12), (20, 0.12)):
            matrix = base.copy()
            matrix.translation += Vector((0.0, offset, 0.0))
            goal.matrix = matrix
            goal.keyframe_insert("location", frame=frame)
        arm6.kinema_ik_enabled = True
        return arm6

    @staticmethod
    def _q(builder, rig):
        return {pb.name: pb.rotation_euler[1] for pb in builder.joint_bones(rig)}

    def test_the_joints_carry_no_curves(self, live, builder):
        """Precondition for this class: nothing but the solver moves the joints."""
        paths = {curve.data_path for curve in _curves(live)}
        assert not any("joint" in path for path in paths)

    def test_a_driven_joint_is_measured_against_the_frame_before(
        self, live, builder, velocity
    ):
        import bpy

        scene = bpy.context.scene
        scene.frame_set(1)
        before = self._q(builder, live)
        scene.frame_set(2)
        after = self._q(builder, live)
        moved = max(abs(after[n] - before[n]) for n in after)
        assert moved > 0.01, "precondition: the solver moved the arm"

        readings = _by_joint(velocity, live)
        for name, reading in readings.items():
            assert reading.speed == pytest.approx(
                abs(after[name] - before[name]) * FPS, abs=1e-4
            ), name
        assert any(r.over for r in readings.values())

    def test_a_still_target_reads_as_still_after_a_move(self, live, velocity):
        """The pose is recorded after the solve. Recorded before it, frame 2
        would hold frame 1's joints, and frame 3 would show frame 2's move."""
        import bpy

        scene = bpy.context.scene
        for frame in (1, 2, 3):
            scene.frame_set(frame)
        readings = _by_joint(velocity, live)
        assert max(r.speed for r in readings.values()) < 0.05

    def test_after_a_jump_a_driven_joint_is_unknown(self, live, velocity):
        import bpy

        scene = bpy.context.scene
        scene.frame_set(1)
        scene.frame_set(20)
        readings = _by_joint(velocity, live)
        assert all(r.speed is None for r in readings.values())
        assert velocity.warnings(live, scene) == []

    def test_a_held_keyed_joint_follows_its_curve(self, live, velocity):
        """Held means the solver leaves it alone, so its curve is the truth."""
        import bpy

        live.pose.bones["joint1"].kinema_ik_hold = True
        _key(live, "joint1", [(10, 0.0), (11, 0.5)])
        scene = bpy.context.scene
        scene.frame_set(1)
        scene.frame_set(11)

        readings = _by_joint(velocity, live)
        assert readings["joint1"].speed == pytest.approx(0.5 * FPS, rel=1e-5)
        assert readings["joint2"].speed is None, "precondition: the others are driven"


class TestOverlay:
    def test_the_lines_trace_the_joint_dial(self, arm6, velocity, overlay):
        """The revolute widget's ring: 0.35 bone lengths round the axis, half way up."""
        import bpy
        from mathutils import Euler, Matrix

        arm6.matrix_world = (
            Matrix.Translation((1.0, -2.0, 0.5))
            @ Euler((0.3, 0.0, 1.1)).to_matrix().to_4x4()
        )
        _key(arm6, "joint2", [(1, 0.0), (2, 0.5)])
        bpy.context.scene.frame_set(2)
        over = velocity.warnings(arm6, bpy.context.scene)
        assert [r.joint for r in over] == ["joint2"]

        pose_bone = arm6.pose.bones["joint2"]
        length = pose_bone.bone.length
        to_bone = (arm6.matrix_world @ pose_bone.matrix).inverted()
        lines = overlay.warning_lines(arm6, over)
        assert len(lines) == len(pose_bone.custom_shape.data.edges)

        ring = [to_bone @ point for segment in lines[:24] for point in segment]
        for point in ring:
            assert point.y == pytest.approx(0.5 * length, abs=1e-5)
            assert np.hypot(point.x, point.z) == pytest.approx(0.35 * length, abs=1e-5)

    def test_the_draw_handlers_are_registered(self, addon, overlay):
        assert len(overlay._handles) == 2


class TestLoadJointLimits:
    YAML = """
joint_limits:
  joint1:
    has_velocity_limits: true
    max_velocity: 0.5
  joint2:
    has_velocity_limits: false
  some_other_joint:
    has_velocity_limits: true
    max_velocity: 9.0
"""

    def test_the_file_overrides_joint_by_joint(self, arm6, builder, tmp_path):
        import bpy

        path = tmp_path / "joint_limits.yaml"
        path.write_text(self.YAML, encoding="utf-8")
        assert "FINISHED" in bpy.ops.kinema.load_joint_limits(filepath=str(path))

        bones = arm6.data.bones
        assert bones["joint1"][builder.PROP_VELOCITY] == pytest.approx(0.5)
        assert builder.PROP_VELOCITY not in bones["joint2"], "turned off by the file"
        assert bones["joint3"][builder.PROP_VELOCITY] == pytest.approx(3.15)

    def test_a_loaded_limit_changes_the_verdict(self, arm6, velocity, tmp_path):
        import bpy

        _key(arm6, "joint1", [(1, 0.0), (25, 1.0)])
        bpy.context.scene.frame_set(13)
        assert not _by_joint(velocity, arm6)["joint1"].over, "1 rad/s against 3.15"

        path = tmp_path / "joint_limits.yaml"
        path.write_text(self.YAML, encoding="utf-8")
        bpy.ops.kinema.load_joint_limits(filepath=str(path))
        assert _by_joint(velocity, arm6)["joint1"].over, "1 rad/s against 0.5"

    def test_another_robots_file_is_refused(self, arm6, tmp_path):
        import bpy

        path = tmp_path / "joint_limits.yaml"
        path.write_text(
            "joint_limits:\n  panda_joint1:\n    has_velocity_limits: true\n"
            "    max_velocity: 2.0\n",
            encoding="utf-8",
        )
        with pytest.raises(RuntimeError, match="panda_joint1"):
            bpy.ops.kinema.load_joint_limits(filepath=str(path))

    def test_a_file_without_limits_is_refused(self, arm6, tmp_path):
        import bpy

        path = tmp_path / "joint_limits.yaml"
        path.write_text("controller_manager:\n  update_rate: 100\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="joint_limits"):
            bpy.ops.kinema.load_joint_limits(filepath=str(path))
