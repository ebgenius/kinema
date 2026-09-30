"""Joint motion limits: read at import, checked at every frame. Needs a real ``bpy``.

The claims worth holding onto:

* the description's ``<limit velocity>`` and ``<limit effort>`` land on the
  joint bone,
* a keyed joint is measured from its own curve, exactly, however the frame was
  reached -- its speed from the frame before, its acceleration from the two,
* a joint live IK drives is measured against the frames before, as the handler
  saw them *after* solving,
* a joint pressed against its stop is not racing, whatever its curve does,
* Ignore Motion Limits silences the panel and the viewport, and measures and
  records nothing while it is on,
* a joint_limits.yaml overrides the description joint by joint and kind by kind.
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
def manager(addon):
    return importlib.import_module(f"{addon.__name__}.solver.manager")


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


def _accelerations(velocity, rig):
    import bpy

    return {r.joint: r for r in velocity.acceleration_readings(rig, bpy.context.scene)}


def _limit_acceleration(builder, rig, value: float) -> None:
    """The same acceleration limit on every joint: URDF gives none."""
    for pose_bone in builder.joint_bones(rig):
        pose_bone.bone[builder.PROP_ACCELERATION] = value


class _Layout:
    """Enough of a UILayout to draw a panel into headless: it keeps the labels."""

    def __init__(self, texts=None):
        self.texts = [] if texts is None else texts
        self.active = True
        self.alert = False
        self.alignment = "EXPAND"

    def label(self, *, text="", icon="NONE"):
        self.texts.append(text)

    def row(self, **_):
        return _Layout(self.texts)

    column = row

    def prop(self, *_, **__):
        pass

    def operator(self, *_, **__):
        pass


class TestImport:
    def test_bones_carry_the_description_velocity(self, arm6, builder):
        bones = arm6.data.bones
        assert bones["joint1"][builder.PROP_VELOCITY] == pytest.approx(3.15)
        assert bones["joint6"][builder.PROP_VELOCITY] == pytest.approx(3.2)

    def test_every_limited_joint_is_read(self, arm6, velocity):
        assert set(_by_joint(velocity, arm6)) == {f"joint{i}" for i in range(1, 7)}

    def test_bones_carry_the_description_effort(self, arm6, builder):
        bones = arm6.data.bones
        assert bones["joint1"][builder.PROP_EFFORT] == pytest.approx(150.0)
        assert bones["joint6"][builder.PROP_EFFORT] == pytest.approx(28.0)

    def test_the_effort_is_listed_under_not_checked(self, arm6):
        import types

        import bpy

        panel = importlib.import_module(f"{EXTENSION_ID}.ui.panel")
        layout = _Layout()
        panel.KINEMA_PT_motion_unchecked.draw(types.SimpleNamespace(layout=layout), bpy.context)
        assert [layout.texts.count(text) for text in ("150 N·m", "28 N·m")] == [3, 3]

    def test_urdf_brings_no_acceleration_limits(self, arm6, builder, velocity):
        """URDF has no field for one: nothing is checked until a file adds them."""
        assert not any(builder.PROP_ACCELERATION in bone for bone in arm6.data.bones)
        assert _accelerations(velocity, arm6) == {}

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


class TestKeyedAcceleration:
    """joint1 still to frame 10, then 0.05 rad a frame to frame 20, then still."""

    @pytest.fixture
    def ramp(self, arm6, builder):
        _limit_acceleration(builder, arm6, 10.0)
        _key(arm6, "joint1", [(1, 0.0), (10, 0.0), (20, 0.5)])
        return arm6

    def test_setting_off_is_measured(self, ramp, velocity):
        """From still to 0.05 rad a frame: 0.05 times 24 squared is 28.8 rad/s²."""
        import bpy

        bpy.context.scene.frame_set(11)
        reading = _accelerations(velocity, ramp)["joint1"]
        assert reading.speed == pytest.approx(0.05 * FPS * FPS, rel=1e-4)
        assert reading.over
        assert any(
            r.joint == "joint1" and r.kind == "acceleration"
            for r in velocity.warnings(ramp, bpy.context.scene)
        )

    def test_stopping_is_measured(self, ramp, velocity):
        import bpy

        bpy.context.scene.frame_set(21)
        assert _accelerations(velocity, ramp)["joint1"].speed == pytest.approx(
            0.05 * FPS * FPS, rel=1e-4
        )

    @pytest.mark.parametrize("frame", [5, 15], ids=["still", "steady"])
    def test_a_steady_speed_is_not_accelerating(self, ramp, velocity, frame):
        import bpy

        bpy.context.scene.frame_set(frame)
        reading = _accelerations(velocity, ramp)["joint1"]
        assert reading.speed == pytest.approx(0.0, abs=0.01)
        assert not reading.over

    def test_the_frame_rate_squared_sets_it(self, ramp, velocity):
        """The same keys at twice the frame rate change speed four times as hard."""
        import bpy

        bpy.context.scene.render.fps = 2 * FPS
        bpy.context.scene.frame_set(11)
        assert _accelerations(velocity, ramp)["joint1"].speed == pytest.approx(
            0.05 * (2 * FPS) ** 2, rel=1e-4
        )


class TestShared:
    """A redraw asks for the readings several times over; they are worked out once."""

    def test_the_readings_are_worked_out_once_a_frame(self, arm6, velocity, monkeypatch):
        import bpy

        _key(arm6, "joint1", [(1, 0.0), (2, 0.5), (3, 0.5)])
        scene = bpy.context.scene
        scene.frame_set(2)
        frames = []
        measure = velocity._speeds

        def counted(rig, scene):
            frames.append(scene.frame_current)
            return measure(rig, scene)

        monkeypatch.setattr(velocity, "_speeds", counted)
        for _ in range(3):
            velocity.readings(arm6, scene)
            velocity.warnings(arm6, scene)
        assert frames == [2]

        scene.frame_set(3)
        velocity.warnings(arm6, scene)
        assert frames == [2, 3]

    def test_an_edit_on_the_same_frame_is_measured_again(self, arm6, velocity):
        """Re-keying the frame on screen updates the depsgraph, which clears what was shared."""
        import bpy

        _key(arm6, "joint1", [(1, 0.0), (2, 0.5)])
        bpy.context.scene.frame_set(2)
        assert _by_joint(velocity, arm6)["joint1"].over, "precondition: 12 rad/s"

        _key(arm6, "joint1", [(2, 0.05)])
        bpy.context.view_layer.update()
        assert not _by_joint(velocity, arm6)["joint1"].over, "still 12 rad/s: read before the edit"


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

    def test_a_checked_rig_lists_what_it_measures(self, arm6):
        """12 rad/s is 688°/s, against arm6's 3.15 rad/s: 180°/s."""
        import bpy

        panel = importlib.import_module(f"{EXTENSION_ID}.ui.panel")
        _key(arm6, "joint1", [(1, 0.0), (2, 0.5)])
        bpy.context.scene.frame_set(2)

        layout = _Layout()
        panel._draw_checked(layout, bpy.context, "speed", missing=("none",), unknown="?")
        assert "688°/s  of  180°/s" in layout.texts

    def test_an_ignored_rig_lists_its_limits_and_measures_nothing(
        self, arm6, velocity, monkeypatch
    ):
        import bpy

        panel = importlib.import_module(f"{EXTENSION_ID}.ui.panel")
        _key(arm6, "joint1", [(1, 0.0), (2, 0.5)])
        bpy.context.scene.frame_set(2)
        arm6.kinema_ignore_velocity = True

        def measure(*_):
            raise AssertionError("measured a rig whose limits are ignored")

        monkeypatch.setattr(velocity, "readings", measure)
        monkeypatch.setattr(velocity, "acceleration_readings", measure)
        layout = _Layout()
        panel._draw_checked(layout, bpy.context, "speed", missing=("none",), unknown="?")
        assert "limit 180°/s" in layout.texts

    def test_an_ignored_rig_is_not_observed(self, arm6, velocity):
        import bpy

        scene = bpy.context.scene
        scene.frame_set(1)
        scene.frame_set(2)
        assert velocity._history.before(arm6.session_uid, 2) is not None, (
            "precondition: a checked rig is"
        )

        arm6.kinema_ignore_velocity = True
        scene.frame_set(10)
        scene.frame_set(11)
        assert velocity._history.before(arm6.session_uid, 11) is None

    def test_turning_the_check_back_on_forgets_the_frames_before(self, arm6, velocity):
        """They were seen before it was ignored; the animation may have changed since."""
        import bpy

        scene = bpy.context.scene
        scene.frame_set(1)
        scene.frame_set(2)
        arm6.kinema_ignore_velocity = True
        arm6.kinema_ignore_velocity = False
        assert velocity._history.before(arm6.session_uid, 3) is None


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

    def test_turning_the_check_back_on_reads_a_driven_joint_afresh(self, live, velocity):
        """Unticking Ignore starts the history again, and the readings shared
        from before go with it: a driven joint is unknown until a frame is stepped."""
        import bpy

        scene = bpy.context.scene
        scene.frame_set(1)
        scene.frame_set(2)
        assert _by_joint(velocity, live)["joint2"].speed is not None, "precondition: measured"

        live.kinema_ignore_velocity = True
        live.kinema_ignore_velocity = False
        assert _by_joint(velocity, live)["joint2"].speed is None

    def test_after_a_jump_a_driven_joint_is_unknown(self, live, velocity):
        import bpy

        scene = bpy.context.scene
        scene.frame_set(1)
        scene.frame_set(20)
        readings = _by_joint(velocity, live)
        assert all(r.speed is None for r in readings.values())
        assert velocity.warnings(live, scene) == []

    def test_a_driven_joint_accelerates_against_the_two_frames_before(
        self, live, builder, velocity
    ):
        """Moved at frame 2 and stopped at 3: that stop is the acceleration."""
        import bpy

        _limit_acceleration(builder, live, 10.0)
        scene = bpy.context.scene
        seen = []
        for frame in (1, 2, 3):
            scene.frame_set(frame)
            seen.append(self._q(builder, live))

        readings = _accelerations(velocity, live)
        for name, reading in readings.items():
            expected = abs(seen[2][name] - 2.0 * seen[1][name] + seen[0][name]) * FPS**2
            assert reading.speed == pytest.approx(expected, rel=1e-3, abs=1e-2), name
        assert any(r.over for r in readings.values())

    def test_a_joint_limited_only_in_acceleration_is_still_recorded(
        self, live, builder, velocity
    ):
        """What was seen is kept for either check, not only for the speed one."""
        import bpy

        _limit_acceleration(builder, live, 10.0)
        for pose_bone in builder.joint_bones(live):
            del pose_bone.bone[builder.PROP_VELOCITY]
        scene = bpy.context.scene
        for frame in (1, 2, 3):
            scene.frame_set(frame)

        readings = _accelerations(velocity, live)
        assert readings, "precondition: the joints are checked"
        assert all(r.speed is not None for r in readings.values())

    def test_one_frame_stepped_leaves_a_driven_acceleration_unknown(
        self, live, builder, velocity
    ):
        import bpy

        _limit_acceleration(builder, live, 10.0)
        scene = bpy.context.scene
        scene.frame_set(1)
        scene.frame_set(2)
        assert _by_joint(velocity, live)["joint2"].speed is not None, (
            "precondition: one frame is enough for a speed"
        )
        assert all(r.speed is None for r in _accelerations(velocity, live).values())

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

    @pytest.mark.parametrize("cached", [True, False], ids=["solver-cached", "no-solver-yet"])
    def test_a_joint_past_the_tip_follows_its_curve(self, live, velocity, manager, cached):
        """Aimed at joint3, the solver never writes joints 4 to 6, so their curves hold.

        Without a cached solver -- a file just opened, nothing solved yet -- this
        has to come out the same: the chain is read off the tip, not the cache.
        """
        import bpy

        live.kinema_ik_tip = 2
        _key(live, "joint5", [(10, 0.0), (11, 0.5)])
        scene = bpy.context.scene
        scene.frame_set(1)
        scene.frame_set(11)
        if not cached:
            manager.invalidate()
        assert (manager._cache.get(live.name) is not None) == cached, "precondition"

        readings = _by_joint(velocity, live)
        assert readings["joint5"].speed == pytest.approx(0.5 * FPS, rel=1e-5)
        assert readings["joint5"].over
        assert readings["joint2"].speed is None, "precondition: the chain is driven"


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

    def test_a_joint_over_both_limits_is_traced_and_labelled_once(
        self, arm6, builder, velocity, overlay
    ):
        """12 rad/s against 3.15, and 288 rad/s² -- half a radian in a frame
        from still -- against 10."""
        import bpy

        _limit_acceleration(builder, arm6, 10.0)
        _key(arm6, "joint2", [(1, 0.0), (2, 0.5)])
        bpy.context.scene.frame_set(2)
        over = velocity.warnings(arm6, bpy.context.scene)
        assert [(r.joint, r.kind) for r in over] == [
            ("joint2", "speed"), ("joint2", "acceleration")
        ], "precondition: over both"

        edges = len(arm6.pose.bones["joint2"].custom_shape.data.edges)
        assert len(overlay.warning_lines(arm6, over)) == edges
        grouped = overlay.by_joint(over)
        assert list(grouped) == ["joint2"]
        assert overlay.label_text("joint2", grouped["joint2"]) == (
            "joint2  speed 381%  accel 2880%"
        )

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

    EVERY_KIND = """
joint_limits:
  joint1:
    has_velocity_limits: true
    max_velocity: 2.0
    has_acceleration_limits: true
    max_acceleration: 4.0
    has_jerk_limits: true
    max_jerk: 80.0
    has_effort_limits: true
    max_effort: 90.0
  joint2:
    has_effort_limits: false
  joint3:
    has_acceleration_limits: true
    max_acceleration: 6.0
"""

    def _load_every_kind(self, tmp_path):
        import bpy

        path = tmp_path / "joint_limits.yaml"
        path.write_text(self.EVERY_KIND, encoding="utf-8")
        assert "FINISHED" in bpy.ops.kinema.load_joint_limits(filepath=str(path))

    def test_every_kind_is_loaded(self, arm6, builder, tmp_path):
        self._load_every_kind(tmp_path)
        bone = arm6.data.bones["joint1"]
        assert [bone[prop] for prop in builder.LIMIT_PROPS.values()] == pytest.approx(
            [2.0, 4.0, 80.0, 90.0]
        )

    def test_each_kind_is_overridden_on_its_own(self, arm6, builder, tmp_path):
        """joint2 turns its effort off and keeps its speed; joint3 gains an
        acceleration and keeps its speed and effort."""
        self._load_every_kind(tmp_path)
        bones = arm6.data.bones
        assert builder.PROP_EFFORT not in bones["joint2"]
        assert bones["joint2"][builder.PROP_VELOCITY] == pytest.approx(3.15)
        assert bones["joint3"][builder.PROP_ACCELERATION] == pytest.approx(6.0)
        assert bones["joint3"][builder.PROP_VELOCITY] == pytest.approx(3.15)
        assert bones["joint3"][builder.PROP_EFFORT] == pytest.approx(150.0)

    def test_a_loaded_acceleration_limit_is_checked(self, arm6, velocity, tmp_path):
        """0.05 rad a frame from still: 28.8 rad/s², against 4."""
        import bpy

        _key(arm6, "joint1", [(1, 0.0), (10, 0.0), (20, 0.5)])
        bpy.context.scene.frame_set(11)
        assert _accelerations(velocity, arm6) == {}, "precondition: none before loading"

        self._load_every_kind(tmp_path)
        assert _accelerations(velocity, arm6)["joint1"].over

    def test_the_rig_describes_them_to_the_solver(self, addon, arm6, tmp_path):
        """What a rig with external axes is solved from: the bones, not the file."""
        rig_model = importlib.import_module(f"{addon.__name__}.solver.rig_model")
        self._load_every_kind(tmp_path)
        joint = next(
            j for j in rig_model.model_from_rig(arm6).joints if j.name == "joint1"
        )
        assert (joint.velocity, joint.acceleration, joint.jerk, joint.effort) == (
            pytest.approx(2.0), pytest.approx(4.0), pytest.approx(80.0), pytest.approx(90.0)
        )

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
