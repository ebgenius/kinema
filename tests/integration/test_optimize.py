"""Optimize Motion. Needs a real ``bpy``, and JAX for the solve.

The claims worth holding onto:
- a job over its limits comes back within them, keyed on every frame with live
  IK off, its waypoints where they were and its linear move on its line;
- a move without the frames to keep within its limits says about how many it
  needs, and the move before it, with time to spare, doesn't;
- a long job eases into a linear move and out of it within its limits;
- a job generated before the waypoints changed, or never generated, is refused,
  with nothing written;
- Check Again on an optimized job still says how many frames a move needs;
- held joints keep their keys;
- a solve that lands after the robot changed (a hold, a limit, the TCP, a held
  joint, Root) or the frame rate did writes nothing, though scrubbing the timeline meanwhile doesn't
  count; one that fails in an unexpected way still ends, and once it has begun
  writing keys it ends in an undo step;
- Generate Motion afterwards replaces it, as it replaces its own keys;
- a second run compiles nothing.

Solving runs modal from the panel, which needs a window; these run the operator
as a script does, which waits for the result.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from ..conftest import requires_bpy

pytestmark = requires_bpy

HOME = np.array([0.0, -0.6, 1.0, 0.0, 0.6, 0.0])
OFFSETS = (
    [0.0] * 6,
    [0.9, 0.3, -0.3, 0.3, 0.2, 0.6],
    [1.2, 0.45, -0.45, 0.3, 0.0, 0.6],
    [-0.5, 0.1, 0.1, -0.2, 0.2, -0.4],
)
MOVES = ("JOINT", "JOINT", "LINEAR", "JOINT")
#: Every joint's top speed (rad/s) and acceleration limit (rad/s²).
SPEED, ACCELERATION = 1.5, 6.0
#: Time enough for every move, with the right profile.
ROOMY = (1, 61, 121, 181)
#: Time enough, but only just: the last move needs joint 1 near its top speed,
#: which smoothing alone takes it past.
TIGHT = (1, 61, 121, 166)
#: The last move, joint 1 turning 1.7 rad, in 30 frames: too few.
SHORT = (1, 61, 121, 151)
#: 150-frame moves at 24 fps, under acceleration limits of LONG_ACCELERATION:
#: the joints are only over where a joint move meets the line, which arrives
#: and leaves at full speed.
LONG = (1, 151, 301, 451)
LONG_ACCELERATION = 0.5


@pytest.fixture
def builder(addon):
    return importlib.import_module(f"{addon.__name__}.rig.builder")


@pytest.fixture
def modules(addon):
    def module(name):
        return importlib.import_module(f"{addon.__name__}.{name}")

    return module


@pytest.fixture
def arm6(addon, fixture_dir, clean_scene, builder, modules):
    """arm6 with live IK on NumPy, and every joint limited as SPEED and ACCELERATION say."""
    import bpy

    scene = bpy.context.scene
    scene.render.fps, scene.render.fps_base = 30, 1.0
    assert "FINISHED" in bpy.ops.kinema.build_robot(filepath=str(fixture_dir / "arm6.urdf"))
    rig = next(o for o in bpy.data.objects if builder.is_kinema_rig(o))
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    _set_q(builder, rig, HOME)
    rig.kinema_solver_mode = "NUMPY"
    assert "FINISHED" in bpy.ops.kinema.add_ik()
    rig.kinema_ik_enabled = False
    for pose_bone in builder.joint_bones(rig):
        pose_bone.bone[builder.PROP_VELOCITY] = SPEED
        pose_bone.bone[builder.PROP_ACCELERATION] = ACCELERATION
    modules("solver.manager").refresh_limits(rig)
    return rig


def _set_q(builder, rig, q):
    import bpy

    for pose_bone, value in zip(builder.joint_bones(rig), q, strict=True):
        pose_bone.rotation_euler[1] = float(value)
    bpy.context.view_layer.update()


def _teach(builder, rig, frames):
    """Four waypoints at ``frames``, the third arrived at in a line."""
    import bpy

    ik_name = rig[builder.PROP_IK_BONE]
    tcp = rig.get(builder.PROP_TCP_BONE) or builder.TCP_BONE
    for index, (frame, offset, move) in enumerate(zip(frames, OFFSETS, MOVES, strict=True)):
        bpy.context.scene.frame_set(frame)
        _set_q(builder, rig, HOME + np.array(offset))
        rig.pose.bones[ik_name].matrix = rig.pose.bones[tcp].matrix.copy()
        bpy.context.view_layer.update()
        assert "FINISHED" in bpy.ops.kinema.add_waypoint(name="ABCD"[index])
        rig.kinema_waypoints[-1].move = move


def _job(builder, rig, frames):
    import bpy

    _teach(builder, rig, frames)
    assert "FINISHED" in bpy.ops.kinema.generate_motion()


def _problems(modules, rig):
    check = modules("rig.motion_check")
    return {row.name: check.problems(row) for row in rig.kinema_motion_check}


def _keys(modules, rig):
    """Joint name -> the frames its channel is keyed on."""
    ik = modules("ops.ik")
    return {
        name: sorted(int(point.co[0]) for point in curve.keyframe_points)
        for name, curve in ik.joint_curves(rig).items()
    }


def _waypoint_error_mm(modules, rig):
    """The worst distance of the tool from a waypoint's pose, at its frame, as it plays."""
    import bpy

    wp = modules("ops.waypoints")
    worst = 0.0
    for waypoint in rig.kinema_waypoints:
        bpy.context.scene.frame_set(int(waypoint.frame))
        tool = np.array(rig.pose.bones[wp._tool_bone(rig)].matrix)
        goal = np.array(wp.pose_of(waypoint))
        worst = max(worst, 1000.0 * float(np.linalg.norm(tool[:3, 3] - goal[:3, 3])))
    return worst


class TestOptimizing:
    def test_a_job_over_its_limits_comes_back_within_them(self, arm6, builder, modules):
        """As generated, joint 1 runs at 153% of its acceleration limit, and 113% of its speed."""
        import bpy

        _job(builder, arm6, TIGHT)
        assert all(_problems(modules, arm6).values()), "precondition: every move flagged"

        assert "FINISHED" in bpy.ops.kinema.optimize_motion()
        assert _problems(modules, arm6) == {"B": [], "C": [], "D": []}
        rows = {row.name: row for row in arm6.kinema_motion_check}
        assert 0.9 < rows["D"].speed_ratio <= 1.0, "precondition: near its top speed"
        for name, frames in _keys(modules, arm6).items():
            assert frames == list(range(TIGHT[0], TIGHT[-1] + 1)), name
        for frame in (TIGHT[0], 90, 140, TIGHT[-1]):
            bpy.context.scene.frame_set(frame)
            assert not arm6.kinema_ik_enabled, frame
        assert _waypoint_error_mm(modules, arm6) < 0.01
        line = next(row for row in arm6.kinema_motion_check if row.move == "LINEAR")
        assert 0.0 <= line.line_error < 1e-5

    def test_a_move_short_of_frames_says_how_many_it_needs(self, arm6, builder, modules):
        """The last move asks joint 1 for 1.7 rad in a second: too much at 1.5 rad/s."""
        import bpy

        _job(builder, arm6, SHORT)
        assert "FINISHED" in bpy.ops.kinema.optimize_motion()

        rows = {row.name: row for row in arm6.kinema_motion_check}
        assert rows["D"].frames_needed > SHORT[3] - SHORT[2]
        assert any("needs about" in found for found in _problems(modules, arm6)["D"])
        # The move before it meets it at a waypoint the check reports in both,
        # but has time to spare itself.
        assert rows["C"].frames_needed == 0
        assert rows["B"].frames_needed == 0
        assert _waypoint_error_mm(modules, arm6) < 0.05

    def test_a_long_job_eases_into_its_line_and_out_of_it(self, arm6, builder, modules):
        """Easing in is spread thinly over a move's 150 frames, which an inexact step never took."""
        import bpy

        bpy.context.scene.render.fps = 24
        for pose_bone in builder.joint_bones(arm6):
            pose_bone.bone[builder.PROP_ACCELERATION] = LONG_ACCELERATION
        modules("solver.manager").refresh_limits(arm6)
        _job(builder, arm6, LONG)
        assert _problems(modules, arm6)["B"], "precondition: over where B meets the line"

        assert "FINISHED" in bpy.ops.kinema.optimize_motion()
        problems = _problems(modules, arm6)
        assert problems == {"B": [], "C": [], "D": []}
        assert _waypoint_error_mm(modules, arm6) < 0.01

    def test_check_again_keeps_saying_how_many_frames_it_needs(self, arm6, builder, modules):
        """On the job Optimize Motion wrote; not on one generated afterwards."""
        import bpy

        _job(builder, arm6, SHORT)
        assert "FINISHED" in bpy.ops.kinema.optimize_motion()
        needed = {row.name: row.frames_needed for row in arm6.kinema_motion_check}
        assert needed["D"] > SHORT[3] - SHORT[2], "precondition: D short of frames"

        assert "FINISHED" in bpy.ops.kinema.check_motion()
        assert {row.name: row.frames_needed for row in arm6.kinema_motion_check} == needed
        assert any("needs about" in found for found in _problems(modules, arm6)["D"])

        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        assert "FINISHED" in bpy.ops.kinema.check_motion()
        assert all(row.frames_needed == 0 for row in arm6.kinema_motion_check)

    def test_held_joints_keep_their_keys(self, arm6, builder, modules):
        import bpy

        _job(builder, arm6, ROOMY)
        held = builder.joint_bones(arm6)[0]
        held.kinema_ik_hold = True
        before = _keys(modules, arm6)

        assert "FINISHED" in bpy.ops.kinema.optimize_motion()
        after = _keys(modules, arm6)
        assert after[held.name] == before[held.name]
        assert after[builder.joint_bones(arm6)[1].name] != before[builder.joint_bones(arm6)[1].name]

    def test_generating_again_replaces_it(self, arm6, builder, modules):
        import bpy

        _job(builder, arm6, ROOMY)
        generated = _keys(modules, arm6)
        assert "FINISHED" in bpy.ops.kinema.optimize_motion()
        assert _keys(modules, arm6) != generated, "precondition: optimizing keyed every frame"

        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        assert _keys(modules, arm6) == generated

    def test_a_second_run_compiles_nothing(self, arm6, builder, modules):
        import bpy

        trajectory = modules("solver.trajectory")
        _job(builder, arm6, ROOMY)
        assert "FINISHED" in bpy.ops.kinema.optimize_motion()
        assert len(trajectory._optimizers) == 1
        optimizer, _ = next(iter(trajectory._optimizers.values()))
        solve = optimizer._fn[0]
        compiled = solve._cache_size()

        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        assert "FINISHED" in bpy.ops.kinema.optimize_motion()
        assert next(iter(trajectory._optimizers.values()))[0] is optimizer
        assert solve._cache_size() == compiled == 1


class TestWhileSolving:
    """Blender stays live while a solve runs, so the rig can change before it lands."""

    @pytest.mark.parametrize(
        "change",
        [
            "hold",
            "limit",
            "tcp",
            "held joint dragged",
            "held joint's keys edited",
            "root posed",
            "frame rate",
        ],
    )
    def test_a_robot_changed_meanwhile_gets_nothing_written(self, arm6, builder, modules, change):
        import bpy

        optimize = modules("ops.optimize")
        _job(builder, arm6, ROOMY)
        held = builder.joint_bones(arm6)[0]
        if change.startswith("held"):
            held.kinema_ik_hold = True
        if change == "held joint dragged":
            # Placed by hand, as a rail often is: nothing keys it.
            ik = modules("ops.ik")
            curve = ik.joint_curves(arm6)[held.name]
            for container in ik.own_fcurve_containers(arm6):
                if curve in list(container):
                    container.remove(curve)
            assert held.name not in ik.joint_curves(arm6), "precondition: unkeyed"
        before = _keys(modules, arm6)
        job = optimize.prepare(bpy.context, arm6)
        optimize.dispatch(job)

        joint = builder.joint_bones(arm6)[2]
        if change == "hold":
            joint.kinema_ik_hold = True
        elif change == "limit":
            joint.bone[builder.PROP_VELOCITY] = 0.5
        elif change == "tcp":
            tcp = arm6.data.bones[arm6.get(builder.PROP_TCP_BONE) or builder.TCP_BONE]
            bpy.context.view_layer.objects.active = arm6
            bpy.ops.object.mode_set(mode="EDIT")
            arm6.data.edit_bones[tcp.name].head.z += 0.05
            bpy.ops.object.mode_set(mode="OBJECT")
        elif change == "held joint dragged":
            held.rotation_euler[1] += 0.1
        elif change == "held joint's keys edited":
            modules("ops.ik").joint_curves(arm6)[held.name].keyframe_points[1].co.y += 0.1
        elif change == "root posed":
            arm6.pose.bones[builder.ROOT_BONE].location.x += 0.05
        else:
            bpy.context.scene.render.fps = 24

        written, kind, message = optimize.finish(bpy.context, job)
        assert (written, kind) == (False, "WARNING")
        assert "nothing was written" in message
        assert _keys(modules, arm6) == before

    def test_scrubbing_meanwhile_still_gets_it_written(self, arm6, builder, modules):
        """A keyed held joint moves with the frame; the keys the solve read haven't changed."""
        import bpy

        optimize = modules("ops.optimize")
        _job(builder, arm6, ROOMY)
        builder.joint_bones(arm6)[0].kinema_ik_hold = True
        job = optimize.prepare(bpy.context, arm6)
        optimize.dispatch(job)
        bpy.context.scene.frame_set(90)

        written, _, message = optimize.finish(bpy.context, job)
        assert written, message

    def test_an_unexpected_error_ends_the_solve(self, arm6, builder, modules, monkeypatch):
        """Not only a refusal: anything left running keeps the rig marked as optimizing."""
        import bpy

        optimize = modules("ops.optimize")
        _job(builder, arm6, ROOMY)
        reports = []
        running = _operator(optimize, optimize.prepare(bpy.context, arm6), reports)
        # Restored afterwards, so a failure here can't leave other tests' rig busy.
        monkeypatch.setattr(optimize, "_running", {arm6.name})

        def broken(job):
            raise ValueError("a shape JAX didn't expect")

        monkeypatch.setattr(optimize, "dispatch", broken)
        timer = type("Event", (), {"type": "TIMER", "value": "NOTHING"})()
        operator = optimize.KINEMA_OT_optimize_motion
        assert operator.modal(running, bpy.context, timer) == {"CANCELLED"}
        assert not optimize.optimizing(arm6)
        assert reports == [({"ERROR"}, "Optimize Motion failed: a shape JAX didn't expect")]

    def test_an_unexpected_error_from_a_script_is_reported(
        self, arm6, builder, modules, monkeypatch
    ):
        """As from the panel: reported as Optimize Motion failing, with nothing written."""
        import bpy

        optimize = modules("ops.optimize")
        _job(builder, arm6, ROOMY)
        before = _keys(modules, arm6)

        def broken(job):
            raise ValueError("a shape JAX didn't expect")

        monkeypatch.setattr(optimize, "dispatch", broken)
        with pytest.raises(RuntimeError, match="Optimize Motion failed: a shape JAX didn't expect"):
            bpy.ops.kinema.optimize_motion()
        assert _keys(modules, arm6) == before

    @pytest.mark.parametrize("fails", ["before writing", "after writing began"])
    def test_a_failure_once_keys_are_written_still_finishes(
        self, arm6, builder, modules, monkeypatch, fails
    ):
        """Only a finished operator gets an undo step, which is what puts the job back."""
        import bpy

        optimize = modules("ops.optimize")
        _job(builder, arm6, ROOMY)
        before = _keys(modules, arm6)
        job = optimize.prepare(bpy.context, arm6)
        optimize.dispatch(job)
        reports = []
        running = _operator(optimize, job, reports)

        def broken(*args, **kwargs):
            raise ValueError("broken")

        if fails == "before writing":
            monkeypatch.setattr(optimize, "model_signature", broken)
        else:
            monkeypatch.setattr(modules("ops.waypoints"), "check_job", broken)
        result = optimize.KINEMA_OT_optimize_motion._finish(running, bpy.context, job)

        assert reports == [({"ERROR"}, "Optimize Motion failed: broken")]
        if fails == "before writing":
            assert result == {"CANCELLED"}
            assert _keys(modules, arm6) == before
        else:
            assert result == {"FINISHED"}
            assert _keys(modules, arm6) != before, "precondition: keys were written"


def _operator(optimize, job, reports):
    """A stand-in for the running operator: Blender makes the real one only in a window."""
    operator = optimize.KINEMA_OT_optimize_motion
    return type(
        "Running",
        (),
        {
            "_job": job,
            "_timer": None,
            "_end": operator._end,
            "_finish": operator._finish,
            "report": lambda self, kind, message: reports.append((kind, message)),
        },
    )()


class TestRefusing:
    def test_a_job_never_generated_is_refused(self, arm6, builder, modules):
        import bpy

        _teach(builder, arm6, ROOMY)
        before = _keys(modules, arm6)
        with pytest.raises(RuntimeError, match="Generate Motion first"):
            bpy.ops.kinema.optimize_motion()
        assert _keys(modules, arm6) == before

    def test_a_job_generated_before_the_waypoints_changed_is_refused(self, arm6, builder, modules):
        import bpy

        _job(builder, arm6, ROOMY)
        before = _keys(modules, arm6)
        arm6.kinema_waypoints[1].frame = 70
        with pytest.raises(RuntimeError, match="Generate Motion again"):
            bpy.ops.kinema.optimize_motion()
        assert _keys(modules, arm6) == before
