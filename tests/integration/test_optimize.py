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
import itertools

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
#: How far above the forearm's middle a post clips it, part-way into it.
CLIPPED = 0.06
#: Half the size of a crate on the arm's outermost capsule, 160 mm into it: big enough
#: that going round it takes the whole move.
CRATE = 0.12
SMALL_CRATE = 0.08
#: The most a joint may wind back before a move or overshoot its end, in radians.
LUG_BOUND = 0.02


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


def _in_the_way(builder, modules, rig, frame=31, up=None, size=0.03, shape="BOX"):
    """An obstacle by the arm's outermost capsule on ``frame``, mid-way through move B.

    ``up`` above the capsule's middle: at :data:`CLIPPED`, the default, it
    reaches 33 mm into the arm; at 0 it is buried in it, 72 mm deep.
    """
    import bpy

    from .test_collision import _fit, _obstacle

    collision = modules("ops.collision")
    _fit(rig)
    bpy.context.scene.frame_set(frame)
    capsules = collision.capsules_of(rig)
    order = [pose_bone.name for pose_bone in builder.joint_bones(rig)]
    index = max(range(len(capsules)), key=lambda i: order.index(capsules[i].bone))
    a, b, _ = collision.capsule_ends(rig, capsules)
    rise = CLIPPED if up is None else up
    middle = (a[index] + b[index]) / 2.0 + np.array([0.0, 0.0, rise])
    obstacle = _obstacle("Post", tuple(middle), size=size, shape=shape)
    bpy.context.view_layer.objects.active = rig
    return obstacle


def _closest(rig):
    return min(
        (row for row in rig.kinema_motion_check if row.clearance_obstacle),
        key=lambda row: row.clearance,
    )


def _lug(builder, rig, radius=0.2, height=0.1, size=0.03):
    """A cube riding on joint 1, ``radius`` out from its axis and ``height`` up."""
    import bpy
    from mathutils import Matrix

    from .test_collision import CUBE_FACES

    mesh = bpy.data.meshes.new("Lug")
    side = (-size, size)
    mesh.from_pydata([(x, y, z) for x in side for y in side for z in side], [], CUBE_FACES)
    lug = bpy.data.objects.new("Lug", mesh)
    bpy.context.scene.collection.objects.link(lug)
    lug.parent, lug.parent_type = rig, "BONE"
    lug.parent_bone = builder.joint_bones(rig)[0].name
    bpy.context.view_layer.update()
    lug.matrix_world = Matrix.Translation((radius, 0.0, height))
    bpy.context.view_layer.update()
    return lug


def _joint_path(modules, builder, rig, frames):
    """(frames, joints): every joint's value on each frame, as it plays."""
    import bpy

    velocity = modules("ops.velocity")
    path = []
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        path.append([velocity.joint_value(bone) for bone in builder.joint_bones(rig)])
    return np.array(path)


def _least_between_waypoints(modules, rig):
    """How close the robot comes to an obstacle on any frame but a waypoint's, in metres."""
    import bpy

    collision = modules("ops.collision")
    geometry = modules("rig.collision")
    obstacles = collision.obstacles(bpy.context)
    held = {int(waypoint.frame) for waypoint in rig.kinema_waypoints}
    first, last = min(held), max(held)
    least = np.inf
    for frame in range(first, last + 1):
        if frame in held:
            continue
        bpy.context.scene.frame_set(frame)
        capsules = collision.capsules_of(rig)
        a, b, radii = collision.capsule_ends(rig, capsules)
        least = min(least, float(geometry.clearances(a, b, radii, obstacles).min()))
    return least


class HurriedPast(AssertionError):
    """The lug went past its bound: the one failure the lug test's mark expects.

    Its own setup and preconditions fail as plain assertions, and so as failures.
    """


def _wind_back_and_overshoot(path):
    """How far a joint's ``path`` over one move goes back before it starts, and past its end."""
    sign = np.sign(path[-1] - path[0])
    return (
        max(0.0, float(-(sign * (path - path[0])).min())),
        max(0.0, float((sign * (path - path[-1])).max())),
    )


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_wind_back_and_overshoot_are_measured_along_the_move(sign):
    """Either way the joint turns: back past its start, and on past its end."""
    path = sign * np.array([0.0, -0.2, 1.0, 2.3, 2.0])
    assert _wind_back_and_overshoot(path) == pytest.approx((0.2, 0.3))
    assert _wind_back_and_overshoot(sign * np.array([0.0, 1.0, 2.0])) == (0.0, 0.0)


class TestKeepingClear:
    @pytest.mark.parametrize(
        "robot", ["arm6", "arm6 with a tool", "arm6 on a track", "rail6", "mjcf_arm"]
    )
    def test_the_capsules_ride_the_solver_s_links(
        self, addon, fixture_dir, clean_scene, builder, modules, robot
    ):
        """Placed by the solver's own kinematics, each is where Blender shows it, in any pose."""
        import bpy

        from .test_collision import TestWhatALinkIs, _fit, _obstacle

        optimize, collision = modules("ops.optimize"), modules("ops.collision")
        name = robot.split()[0]
        path = fixture_dir / (f"{name}.xml" if name.startswith("mjcf") else f"{name}.urdf")
        assert "FINISHED" in bpy.ops.kinema.build_robot(filepath=str(path))
        rig = next(o for o in bpy.data.objects if builder.is_kinema_rig(o))
        bpy.context.view_layer.objects.active = rig
        rig.select_set(True)
        rig.kinema_solver_mode = "NUMPY"
        assert "FINISHED" in bpy.ops.kinema.add_ik()
        if robot.endswith("track"):
            assert "FINISHED" in bpy.ops.kinema.add_external_axis(axis_name="track")
        if robot.endswith("tool"):
            tcp = rig.get(builder.PROP_TCP_BONE) or builder.TCP_BONE
            TestWhatALinkIs._on_bone(rig, tcp, "Gripper", size=0.04, offset=(0.0, 0.0, 0.05))
        # Moved and turned, so the robot's own placement is in it too.
        rig.location, rig.rotation_euler = (0.3, -0.2, 0.1), (0.0, 0.0, 0.7)
        _fit(rig)
        _obstacle("FarAway", (5.0, 5.0, 5.0), size=0.1)
        bpy.context.view_layer.objects.active = rig

        solver, pyroki = optimize._model(rig)
        which, links, a, b = optimize.capsule_links(rig, pyroki)
        capsules = collision.capsules_of(rig)
        assert len(which) == len(capsules), "precondition: every capsule moves with a joint"
        if robot.endswith("tool"):
            assert any(capsules[i].bone == tcp for i in which), "precondition: one on the TCP"
        jax = pytest.importorskip("jax")
        import jaxlie

        column = {joint: index for index, joint in enumerate(pyroki.actuated_names)}
        manager = modules("solver.manager")
        random = np.random.default_rng(0)
        for _ in range(3):
            for pose_bone in builder.joint_bones(rig):
                low = pose_bone.bone.get(builder.PROP_LOWER, -1.0)
                high = pose_bone.bone.get(builder.PROP_UPPER, 1.0)
                prismatic = pose_bone.bone.get(builder.PROP_JOINT_TYPE) == "prismatic"
                value = float(random.uniform(max(low, -1.0), min(high, 1.0)))
                if prismatic:
                    pose_bone.location[1] = value
                else:
                    pose_bone.rotation_euler[1] = value
            bpy.context.view_layer.update()
            q = np.zeros(len(column))
            for pose_bone in builder.joint_bones(rig):
                if pose_bone.name in column:
                    q[column[pose_bone.name]] = modules("ops.velocity").joint_value(pose_bone)
            poses = jaxlie.SE3(pyroki.robot.forward_kinematics(jax.numpy.asarray(q))[links])
            placement = np.array(rig.matrix_world) @ manager.root_pose(rig)
            solved = [
                np.asarray(poses.apply(jax.numpy.asarray(ends))) @ placement[:3, :3].T
                + placement[:3, 3]
                for ends in (a, b)
            ]
            shown = collision.capsule_ends(rig, [capsules[i] for i in which])[:2]
            for got, expected in zip(solved, shown, strict=True):
                assert np.abs(got - expected).max() < 1e-4

    def test_a_job_through_an_obstacle_comes_back_clear_of_it(self, arm6, builder, modules):
        """The forearm swings through a post on move B; optimized, it goes round it."""
        import bpy

        optimize = modules("ops.optimize")
        _job(builder, arm6, ROOMY)
        _in_the_way(builder, modules, arm6)
        assert "FINISHED" in bpy.ops.kinema.check_motion()
        assert _closest(arm6).clearance < -0.02, "precondition: it hits the post"
        assert any("hits Post" in found for found in _problems(modules, arm6)["B"])

        job = optimize.prepare(bpy.context, arm6)
        assert job.problem.collision is not None
        optimize.dispatch(job)
        written, kind, message = optimize.finish(bpy.context, job)
        assert (written, kind) == (True, "INFO")
        assert "within every limit" in message and "mm clear at the closest" in message
        assert _problems(modules, arm6) == {"B": [], "C": [], "D": []}
        assert _closest(arm6).clearance >= arm6.kinema_keep_clear
        assert _waypoint_error_mm(modules, arm6) < 0.01
        line = next(row for row in arm6.kinema_motion_check if row.move == "LINEAR")
        assert 0.0 <= line.line_error < 1e-5

    def test_it_keeps_as_far_away_as_asked(self, arm6, builder, modules):
        import bpy

        _job(builder, arm6, ROOMY)
        _in_the_way(builder, modules, arm6)
        arm6.kinema_keep_clear = 0.04
        assert "FINISHED" in bpy.ops.kinema.optimize_motion()
        assert 0.04 <= _closest(arm6).clearance < 0.045

    def test_a_post_buried_in_the_arm_is_gone_round_within_the_limits(
        self, arm6, builder, modules
    ):
        """Deep in, a weak pull still takes the arm all the way out: nothing holds it."""
        import bpy

        _job(builder, arm6, ROOMY)
        _in_the_way(builder, modules, arm6, up=0.0)
        assert "FINISHED" in bpy.ops.kinema.check_motion()
        assert _closest(arm6).clearance < -0.07, "precondition: buried"

        assert "FINISHED" in bpy.ops.kinema.optimize_motion()
        assert _closest(arm6).clearance >= arm6.kinema_keep_clear
        assert _problems(modules, arm6) == {"B": [], "C": [], "D": []}
        assert _waypoint_error_mm(modules, arm6) < 0.01

    @pytest.mark.parametrize("frame", [1, 121])
    def test_where_it_can_t_keep_clear_the_job_stays_as_taught(
        self, arm6, builder, modules, frame
    ):
        """A post on the job's pinned first frame, or on waypoint C: no amount of pulling
        clears it there, and pulling hard would wreck the job round it instead."""
        import bpy

        _job(builder, arm6, ROOMY)
        _in_the_way(builder, modules, arm6, frame=frame, up=0.0)
        assert "FINISHED" in bpy.ops.kinema.check_motion()
        before = {row.name: row.clearance for row in arm6.kinema_motion_check}
        assert _closest(arm6).clearance_frame == frame, "precondition: hit there"
        if frame == 1:
            assert before["D"] < 0.0, "precondition: D clips it too, elsewhere"

        assert "FINISHED" in bpy.ops.kinema.optimize_motion()
        assert _waypoint_error_mm(modules, arm6) < 0.01
        closest = _closest(arm6)
        assert (closest.clearance_frame, closest.clearance_obstacle) == (frame, "Post")
        assert closest.clearance < -0.07
        for row in arm6.kinema_motion_check:
            assert row.speed_ratio < 1.05 and row.accel_ratio < 1.05, row.name
        if frame == 1:
            row = next(row for row in arm6.kinema_motion_check if row.name == "D")
            assert row.clearance >= arm6.kinema_keep_clear

    @pytest.mark.parametrize("keep", [0.01, 0.02, 0.03, 0.05])
    def test_a_sweep_of_keep_clear_by_never_ends_inside_the_crate(
        self, arm6, builder, modules, keep
    ):
        """A crate the arm swings 160 mm into, on move B and back on D.

        Every solve has to go round it. Before a stalled solve was solved again
        and the hinges rounded, it ended 63 mm in at 20 mm: jaxls gave up while
        the cost was still falling.

        Move D starts and ends far from the crate, so it keeps all that's asked,
        within the half millimetre the report allows. Move B starts on the job's
        pinned first frame, 7.5 mm from it, and can only be clear.
        """
        import bpy

        _job(builder, arm6, ROOMY)
        _in_the_way(builder, modules, arm6, up=0.0, size=CRATE)
        assert "FINISHED" in bpy.ops.kinema.check_motion()
        assert _closest(arm6).clearance < -0.15, "precondition: deep in"
        before = {row.name: row.clearance for row in arm6.kinema_motion_check}
        assert before["D"] < -0.1, "precondition: deep in on the way back too"

        arm6.kinema_keep_clear = keep
        assert "FINISHED" in bpy.ops.kinema.optimize_motion()
        assert _closest(arm6).clearance > 0.0
        rows = {row.name: row for row in arm6.kinema_motion_check}
        assert rows["D"].clearance >= keep - 0.0005
        assert _waypoint_error_mm(modules, arm6) < 0.01

    def test_a_solve_that_stops_short_is_solved_again(self, arm6, builder, modules):
        """A smaller crate, and a last move with barely the frames for its swing.

        jaxls's first pass ends after 10 iterations on the size of its last
        step, with the cost still falling and, left there, the arm 115 mm in.
        Solved again from there, with its damping reset, it goes round.
        """
        import bpy

        optimize = modules("ops.optimize")
        _job(builder, arm6, TIGHT)
        _in_the_way(builder, modules, arm6, up=0.0, size=SMALL_CRATE)
        arm6.kinema_keep_clear = 0.03
        job = optimize.prepare(bpy.context, arm6)
        optimize.dispatch(job)
        written, _, _ = optimize.finish(bpy.context, job)
        assert written
        # The start at the one saturation throughout: the restarts, unmixed.
        passes = job.pending.passes(start=0)
        assert not passes[0].settled, "precondition: the first pass stopped short"
        assert len(passes) > 1
        for before, after in itertools.pairwise(passes):
            assert after.before == min(before.before, before.after)
        assert _closest(arm6).clearance > 0.025

    def test_more_frames_past_this_crate_is_never_worse(self, arm6, builder, modules):
        """The same job past the same crate, its last move, D, given more frames each time.

        Room to spare should only help. Before, at 30 mm, 161 frames ended 3 mm
        clear and 166 ended 90 mm in. Now move D comes out 6.8 mm in at 161
        frames, then 26.9, 29.5, 30.4, 30.6 and 30.9 mm clear.

        This holds for this scene, not every scene: on the 80 mm crate with a
        short last move, at Keep Clear By 50 mm, 161 frames end 39.7 mm clear
        and 166 end 65.1 mm in. Steps 3 and 4 of #98 are for that.
        """
        import bpy

        _job(builder, arm6, ROOMY)
        _in_the_way(builder, modules, arm6, up=0.0, size=CRATE)
        arm6.kinema_keep_clear = 0.03
        last = max(arm6.kinema_waypoints, key=lambda waypoint: waypoint.frame)
        reached = []
        for end in (161, 166, 171, 176, 181, 211):
            last.frame = end
            assert "FINISHED" in bpy.ops.kinema.generate_motion()
            assert "FINISHED" in bpy.ops.kinema.optimize_motion()
            rows = {row.name: row for row in arm6.kinema_motion_check}
            reached.append(rows["D"].clearance)
        for fewer, more in itertools.pairwise(reached):
            assert more >= fewer - 0.001, [round(1000 * r, 1) for r in reached]

    @pytest.mark.parametrize("keep", [0.01, 0.02, 0.03])
    def test_where_one_solve_settles_inside_the_other_goes_round(
        self, arm6, builder, modules, keep
    ):
        """The crate 60 mm higher: move B can go round it, but a solve settles inside.

        Solved one way only, from the job as it plays, B ended 42 to 47 mm in
        at each of these, the same before and after the solve was solved two
        ways: no flip, but somewhere a solve stops, the pull having levelled
        off. The second start, its pull stronger for its first passes, goes
        round, and costs less, so it is kept.
        """
        import bpy

        _job(builder, arm6, ROOMY)
        _in_the_way(builder, modules, arm6, up=0.06, size=CRATE)
        assert "FINISHED" in bpy.ops.kinema.check_motion()
        assert _closest(arm6).clearance < -0.1, "precondition: deep in"

        arm6.kinema_keep_clear = keep
        assert "FINISHED" in bpy.ops.kinema.optimize_motion()
        assert _least_between_waypoints(modules, arm6) > 0.0
        assert _waypoint_error_mm(modules, arm6) < 0.01

    def test_the_second_start_doesn_t_end_settled_on_its_schedule(self, arm6, builder, modules):
        """The crate 60 mm higher at 20 mm, where the second start's first pass,
        at the schedule's 0.2, ends on jaxls's cost or gradient rule.

        That settles the problem at 0.2, not the one each pass is judged by, so
        the start goes on, as it would after any pass that stopped short and
        gained. It used to end there. And the iterations reported are the kept
        start's alone, where they were both starts' together.
        """
        import bpy

        optimize = modules("ops.optimize")
        trajectory = modules("solver.trajectory")
        _job(builder, arm6, ROOMY)
        _in_the_way(builder, modules, arm6, up=0.06, size=CRATE)
        arm6.kinema_keep_clear = 0.02
        job = optimize.prepare(bpy.context, arm6)
        optimize.dispatch(job)
        written, _, _ = optimize.finish(bpy.context, job)
        assert written
        passes = job.pending.passes(start=1)
        assert passes[0].saturation > trajectory.SATURATION
        assert passes[0].settled, "precondition: the first scheduled pass settles"
        gained = passes[0].after < passes[0].before * (1.0 - trajectory.RESTART_GAIN)
        assert gained, "precondition: settling is the only reason to stop"
        assert len(passes) > 1

        _, iterations = job.pending.result()
        assert iterations == sum(each.iterations for each in job.pending.passes())
        assert len(job.pending.ended()) == 2

    @pytest.mark.parametrize("keep, change", [(0.01, "one frame more"), (0.03, "B 1 mm along x")])
    def test_a_small_change_to_the_job_doesn_t_flip_it(self, arm6, builder, modules, keep, change):
        """The crate again, move D ending at 171: a frame more, or waypoint B 1 mm over.

        Solved one way only, from the job as it plays, the job as it was came
        out 8.2 and 9.9 mm clear, and these 90.6 and 109.3 mm in: the solve
        settled inside the crate, where the pull levels off. Solved a second
        way too, a larger saturation first, and the cheaper kept, each comes
        out as clear as before.
        """
        import bpy
        from mathutils import Vector

        _job(builder, arm6, ROOMY)
        _in_the_way(builder, modules, arm6, up=0.0, size=CRATE)
        arm6.kinema_keep_clear = keep
        last = max(arm6.kinema_waypoints, key=lambda waypoint: waypoint.frame)
        last.frame = 171
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        assert "FINISHED" in bpy.ops.kinema.optimize_motion()
        before = _least_between_waypoints(modules, arm6)
        assert before > 0.0, "precondition: the job as it was comes out clear"

        if change == "one frame more":
            last.frame = 172
        else:
            marker = next(w for w in arm6.kinema_waypoints if w.name == "B").marker
            moved = marker.matrix_basis.copy()
            moved.translation = moved.translation + Vector((0.001, 0.0, 0.0))
            marker.matrix_basis = moved
            bpy.context.view_layer.update()
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        assert "FINISHED" in bpy.ops.kinema.optimize_motion()
        after = _least_between_waypoints(modules, arm6)
        assert after > 0.0 and abs(after - before) < 0.002, (before, after)

    @pytest.mark.xfail(
        strict=True,
        raises=HurriedPast,
        reason="#98's steps 3 and 4: what a pair can be asked for, and conceding it",
    )
    def test_a_lug_that_can_t_get_clear_doesn_t_hurry_past(self, arm6, builder, modules):
        """A lug on joint 1, swept through a post on move B and back through it on D.

        Only joint 1 moves the lug, and it can't turn the other way round, so
        the lug can't get clear. Each frame it spends in the post costs, and
        fewer such frames is all a solve can do about it: it winds joint 1
        back, crosses at the speed limit and overshoots. Measured with this
        kernel, at 10 mm: 0.28 rad back and 0.07 over on move B, at 1.03 times
        the speed limit. The prototype of steps 1-4 on #98 kept it to none of
        either, within the limits; steps 1-3 alone, 0.24 rad back.
        """
        import bpy

        from .test_collision import _fit, _obstacle

        lug = _lug(builder, arm6)
        _job(builder, arm6, ROOMY)
        _fit(arm6)
        bpy.context.scene.frame_set(31)
        _obstacle("Post", tuple(lug.matrix_world.translation), size=0.03)
        bpy.context.view_layer.objects.active = arm6
        assert "FINISHED" in bpy.ops.kinema.check_motion()
        assert _closest(arm6).clearance_bone == builder.joint_bones(arm6)[0].name
        assert _closest(arm6).clearance < -0.05, "precondition: the lug goes through it"

        assert "FINISHED" in bpy.ops.kinema.optimize_motion()
        path = _joint_path(modules, builder, arm6, range(ROOMY[0], ROOMY[-1] + 1))[:, 0]
        assert np.isfinite(path).all()
        for start, end in itertools.pairwise(ROOMY):
            wind, over = _wind_back_and_overshoot(path[start - 1 : end])
            if wind > LUG_BOUND or over > LUG_BOUND:
                raise HurriedPast((start, end, wind, over))
        for row in arm6.kinema_motion_check:
            if row.speed_ratio > 1.0:
                raise HurriedPast((row.name, row.speed_ratio))

    def test_unticked_it_goes_through_as_before(self, arm6, builder, modules):
        import bpy

        optimize = modules("ops.optimize")
        _job(builder, arm6, ROOMY)
        _in_the_way(builder, modules, arm6)
        arm6.kinema_avoid_obstacles = False
        job = optimize.prepare(bpy.context, arm6)
        assert job.problem.collision is None and job.optimizer.shape is None
        optimize.dispatch(job)
        written, _, message = optimize.finish(bpy.context, job)
        assert written and "clear" not in message
        assert _closest(arm6).clearance < -0.02

    def test_an_obstacle_that_moves_is_kept_clear_of_where_it_is(self, arm6, builder, modules):
        """Keyed far off on the job's first frame, and into the arm's way by frame 31."""
        import bpy

        _job(builder, arm6, ROOMY)
        post = _in_the_way(builder, modules, arm6)
        where = post.location.copy()
        post.keyframe_insert("location", frame=31)
        post.location = (where.x, where.y, where.z + 2.0)
        post.keyframe_insert("location", frame=1)
        post.keyframe_insert("location", frame=61)
        bpy.context.view_layer.objects.active = arm6
        assert "FINISHED" in bpy.ops.kinema.check_motion()
        assert _closest(arm6).clearance < -0.02, "precondition: it hits the post"

        assert "FINISHED" in bpy.ops.kinema.optimize_motion()
        assert _closest(arm6).clearance >= arm6.kinema_keep_clear

    def test_a_robot_carried_along_keeps_clear_where_it_is_carried(self, arm6, builder, modules):
        """The robot's object keyed to move: the obstacles are measured from where it is."""
        import bpy

        _job(builder, arm6, ROOMY)
        post = _in_the_way(builder, modules, arm6)
        post.location.y += 0.5
        arm6.keyframe_insert("location", frame=1)
        arm6.location.y += 0.5
        arm6.keyframe_insert("location", frame=31)
        arm6.keyframe_insert("location", frame=181)
        bpy.context.view_layer.objects.active = arm6
        assert "FINISHED" in bpy.ops.kinema.check_motion()
        assert _closest(arm6).clearance < -0.02, "precondition: it hits the post"

        assert "FINISHED" in bpy.ops.kinema.optimize_motion()
        assert _closest(arm6).clearance >= arm6.kinema_keep_clear

    def test_capsules_the_meshes_have_outgrown_are_refused(self, arm6, builder, modules):
        import bpy

        collision = modules("ops.collision")
        _job(builder, arm6, ROOMY)
        _in_the_way(builder, modules, arm6)
        next(iter(collision.link_meshes(arm6).values()))[0].location.x += 0.05
        bpy.context.view_layer.update()
        assert collision.capsules_stale(arm6), "precondition: stale"
        before = _keys(modules, arm6)
        with pytest.raises(RuntimeError, match="Fit Again, or untick Avoid Obstacles"):
            bpy.ops.kinema.optimize_motion()
        assert _keys(modules, arm6) == before
        arm6.kinema_avoid_obstacles = False
        assert "FINISHED" in bpy.ops.kinema.optimize_motion()

    def test_a_link_with_no_frame_to_place_its_capsules_by_is_refused(
        self, arm6, builder, modules
    ):
        """Left out of the solve instead, its capsules would go through the post."""
        import bpy

        collision = modules("ops.collision")
        _job(builder, arm6, ROOMY)
        _in_the_way(builder, modules, arm6)
        node = arm6.data.bones[collision.capsules_of(arm6)[-1].bone]
        while builder.PROP_CHILD_LINK not in node:
            node = node.parent
        del node[builder.PROP_LINK_CORRECTION]
        assert not collision.capsules_stale(arm6), "precondition: the capsules still fit"
        before = _keys(modules, arm6)
        with pytest.raises(RuntimeError, match=f"ride {node.name}, which records no link frame"):
            bpy.ops.kinema.optimize_motion()
        assert _keys(modules, arm6) == before
        arm6.kinema_avoid_obstacles = False
        assert "FINISHED" in bpy.ops.kinema.optimize_motion()

    @pytest.mark.parametrize(
        "change",
        [
            "obstacle moved", "keep clear", "unticked", "capsules cleared", "robot moved",
            "capsule changed", "meshes outgrew the capsules",
        ],
    )
    def test_a_cell_changed_meanwhile_gets_nothing_written(self, arm6, builder, modules, change):
        import bpy

        optimize = modules("ops.optimize")
        collision = modules("ops.collision")
        _job(builder, arm6, ROOMY)
        post = _in_the_way(builder, modules, arm6)
        before = _keys(modules, arm6)
        job = optimize.prepare(bpy.context, arm6)
        optimize.dispatch(job)
        source = arm6.kinema_capsules_source

        if change == "capsule changed":
            arm6.kinema_capsules[-1].radius += 0.01
        elif change == "meshes outgrew the capsules":
            next(iter(collision.link_meshes(arm6).values()))[0].location.x += 0.05
        elif change == "obstacle moved":
            post.location.x += 0.1
        elif change == "keep clear":
            arm6.kinema_keep_clear = 0.05
        elif change == "unticked":
            arm6.kinema_avoid_obstacles = False
        elif change == "capsules cleared":
            bpy.context.view_layer.objects.active = arm6
            assert "FINISHED" in bpy.ops.kinema.clear_capsules()
        else:
            arm6.location.x += 0.1
        bpy.context.view_layer.update()
        if change in ("capsule changed", "meshes outgrew the capsules"):
            # Neither refits: the meshes' fingerprint is the one the solve began with.
            assert arm6.kinema_capsules_source == source, "precondition: not fitted again"
            outgrown = change == "meshes outgrew the capsules"
            assert collision.capsules_stale(arm6) == outgrown, "precondition"

        written, kind, message = optimize.finish(bpy.context, job)
        assert (written, kind) == (False, "WARNING")
        assert "nothing was written" in message
        assert _keys(modules, arm6) == before

    def test_another_obstacle_compiles_nothing(self, arm6, builder, modules):
        """Obstacles are padded to a count, so a kernel serves a cell until it outgrows it."""
        import bpy

        from .test_collision import _obstacle

        trajectory = modules("solver.trajectory")
        _job(builder, arm6, ROOMY)
        _in_the_way(builder, modules, arm6)
        assert "FINISHED" in bpy.ops.kinema.optimize_motion()
        ((optimizer, _),) = trajectory._optimizers.values()
        solve = optimizer._fn[0]

        _obstacle("Wall", (2.0, 0.0, 0.5), size=0.2)
        bpy.context.view_layer.objects.active = arm6
        assert "FINISHED" in bpy.ops.kinema.optimize_motion()
        assert next(iter(trajectory._optimizers.values()))[0] is optimizer
        assert solve._cache_size() == 1


    def test_the_collision_panel_offers_it(self, arm6, builder, modules):
        """Drawn into a layout that records it: real properties, greyed out when unticked."""
        import types

        import bpy

        panel = modules("ui.panel")
        log = []

        class Layout:
            use_property_split = enabled = alert = False
            alignment, scale_y = "", 1.0

            def __init__(self):
                self.active = True

            def row(self, **_):
                made = Layout()
                log.append(("layout", made))
                return made

            column = box = split = row

            def separator(self, **_):
                pass

            def label(self, *, text="", icon="NONE"):
                log.append(("label", text))

            def prop(self, data, name, **_):
                log.append(("prop", data, name, self))

            def operator(self, idname, **_):
                return types.SimpleNamespace()

        _job(builder, arm6, ROOMY)
        _in_the_way(builder, modules, arm6)
        for avoid in (True, False):
            arm6.kinema_avoid_obstacles = avoid
            log.clear()
            panel.KINEMA_PT_collision.draw(types.SimpleNamespace(layout=Layout()), bpy.context)
            drawn = {entry[2]: entry[3] for entry in log if entry[0] == "prop" and entry[1] == arm6}
            assert set(drawn) == {"kinema_avoid_obstacles", "kinema_keep_clear"}
            assert all(name in arm6.bl_rna.properties for name in drawn)
            assert drawn["kinema_avoid_obstacles"].active
            assert drawn["kinema_keep_clear"].active is avoid
            assert ("label", "Optimize Motion") in log
