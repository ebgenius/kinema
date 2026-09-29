"""Waypoint recording and motion generation. Needs a real ``bpy``.

The claims worth holding onto:

* a waypoint remembers the *configuration* it was taught in, not just a pose,
* a JOINT span moves the joints and leaves the solver out of it,
* a LINEAR span drives the tool along the straight line between two poses,
  turning the short way, in the configuration its ends were taught in,
* regenerating replaces the previous motion rather than layering on it,
* a generated job bakes to exactly what it plays,
* the job is checked as it plays, move by move.
"""

from __future__ import annotations

import importlib
import math

import numpy as np
import pytest

from ..conftest import requires_bpy

pytestmark = requires_bpy


@pytest.fixture
def builder(addon):
    return importlib.import_module(f"{addon.__name__}.rig.builder")


@pytest.fixture
def wp_ops(addon):
    return importlib.import_module(f"{addon.__name__}.ops.waypoints")


@pytest.fixture
def ik_ops(addon):
    return importlib.import_module(f"{addon.__name__}.ops.ik")


@pytest.fixture
def arm6(addon, fixture_dir, clean_scene, builder):
    """A 6-DoF rig with an IK target, which linear moves need."""
    import bpy

    assert "FINISHED" in bpy.ops.kinema.build_robot(
        filepath=str(fixture_dir / "arm6.urdf")
    )
    rig = next(o for o in bpy.data.objects if builder.is_kinema_rig(o))
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.kinema.add_ik()
    rig.kinema_ik_enabled = False
    return rig


@pytest.fixture
def wide6(addon, fixture_dir, clean_scene, builder):
    """arm6 with a +-350 degree wrist, which can hold one pose a whole turn apart."""
    import bpy

    assert "FINISHED" in bpy.ops.kinema.build_robot(
        filepath=str(fixture_dir / "arm6_wide.urdf")
    )
    rig = next(o for o in bpy.data.objects if builder.is_kinema_rig(o))
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.kinema.add_ik()
    rig.kinema_ik_enabled = False
    return rig


Q0 = [0.0, -0.6, 1.0, 0.0, 0.6, 0.0]
Q1 = [0.8, -0.9, 1.3, 0.2, 0.5, 0.3]

#: Three configurations on one branch, close enough for a line between them.
REACH_A = [0.4, -0.7, 1.1, 0.2, 0.5, 0.0]
REACH_B = [0.6, -0.8, 1.2, 0.3, 0.6, 0.2]
REACH_C = [0.9, -0.5, 0.9, 0.1, 0.4, 0.1]

#: A wrist where the IK target's rotation passes from one quaternion hemisphere
#: to the other as joint 6 goes through about 1.075 rad. Found by sweeping
#: arm6's wrist. Joint 6 is well inside its limits on either side, so the arm
#: can follow a turn across it.
WRIST = [0.0, -0.6, 1.0, 2.0, 0.4]
FLIP = 1.075


def _set_q(builder, rig, q):
    import bpy

    for pose_bone, value in zip(builder.joint_bones(rig), q, strict=True):
        pose_bone.rotation_euler[1] = value
    bpy.context.view_layer.update()


def _q(builder, rig):
    return np.array([pb.rotation_euler[1] for pb in builder.joint_bones(rig)])


def _tool(builder, rig):
    import bpy

    name = rig.get(builder.PROP_TCP_BONE) or builder.TCP_BONE
    bpy.context.view_layer.update()
    return np.array(rig.pose.bones[name].matrix.translation)


def _drag_goal(builder, rig, offset, steps=20):
    """Move the IK goal by ``offset`` the way a hand drags it: a little at a time.

    Live IK solves each small step from the last, so the arm keeps the
    configuration it started in. Moved in one jump the solver is free to land
    in any configuration that reaches the goal, and a linear move taught that
    way would be one that has to change configuration part-way.
    """
    import bpy
    from mathutils import Vector

    goal = rig.pose.bones[rig.get(builder.PROP_IK_BONE)]
    tcp_name = rig.get(builder.PROP_TCP_BONE) or builder.TCP_BONE
    goal.matrix = rig.pose.bones[tcp_name].matrix.copy()
    rig.kinema_ik_enabled = True
    bpy.context.view_layer.update()
    start = goal.matrix.translation.copy()
    for step in range(1, steps + 1):
        matrix = goal.matrix.copy()
        matrix.translation = start + Vector(offset) * (step / steps)
        goal.matrix = matrix
        bpy.context.view_layer.update()


def _teach(builder, rig, frame, q, move, name):
    """Record a waypoint at ``frame`` with the rig posed at ``q``."""
    import bpy

    bpy.context.scene.frame_set(frame)
    _set_q(builder, rig, q)
    assert "FINISHED" in bpy.ops.kinema.add_waypoint(name=name)
    rig.kinema_waypoints[-1].move = move
    return rig.kinema_waypoints[-1]


class TestRecording:
    def test_a_waypoint_stores_the_pose_and_the_configuration(
        self, arm6, builder, wp_ops
    ):
        """A pose has up to eight solutions; only the joint vector says which."""
        point = _teach(builder, arm6, 1, Q1, "JOINT", "pick")

        assert point.dof == 6
        assert np.allclose(list(point.q)[:6], Q1, atol=1e-6)
        stored = wp_ops._unflatten(point.pose)
        assert np.allclose(
            np.array(stored.translation), _tool(builder, arm6), atol=1e-6
        )

    def test_recording_makes_a_marker_at_the_pose(self, arm6, builder, wp_ops):
        """An Empty, so #3's 'snap it to a feature' has something to snap."""
        point = _teach(builder, arm6, 1, Q1, "JOINT", "pick")

        assert point.marker is not None
        assert point.marker.parent is arm6
        assert wp_ops.PROP_WAYPOINT in point.marker
        assert np.allclose(
            np.array(point.marker.matrix_basis.translation),
            _tool(builder, arm6),
            atol=1e-6,
        )

    def test_going_to_a_waypoint_restores_the_taught_configuration(
        self, arm6, builder
    ):
        """Not a re-solve: the solver is free to pick a different branch."""
        import bpy

        _teach(builder, arm6, 1, Q0, "JOINT", "home")
        _teach(builder, arm6, 30, Q1, "JOINT", "pick")

        _set_q(builder, arm6, [0.0] * 6)
        assert "FINISHED" in bpy.ops.kinema.goto_waypoint(index=0)

        assert bpy.context.scene.frame_current == 1
        assert np.allclose(_q(builder, arm6), Q0, atol=1e-6)

    def test_update_re_records_from_the_current_pose(self, arm6, builder):
        import bpy

        point = _teach(builder, arm6, 1, Q0, "JOINT", "home")
        _set_q(builder, arm6, Q1)
        assert "FINISHED" in bpy.ops.kinema.update_waypoint(index=0)

        assert np.allclose(list(point.q)[:6], Q1, atol=1e-6)

    def test_removing_a_row_takes_its_marker(self, arm6, builder):
        import bpy

        point = _teach(builder, arm6, 1, Q0, "JOINT", "home")
        name = point.marker.name

        assert "FINISHED" in bpy.ops.kinema.remove_waypoint(index=0)
        assert len(arm6.kinema_waypoints) == 0
        assert name not in bpy.data.objects


class TestGeneration:
    def test_two_waypoints_are_needed(self, arm6, builder):
        """bpy raises rather than returning when an operator reports ERROR."""
        import bpy

        _teach(builder, arm6, 1, Q0, "JOINT", "home")
        with pytest.raises(RuntimeError, match="At least two waypoints"):
            bpy.ops.kinema.generate_motion()

    def test_a_joint_span_moves_the_joints_and_leaves_ik_off(self, arm6, builder):
        """Joint-space interpolation is what a MoveJ is."""
        import bpy

        _teach(builder, arm6, 1, Q0, "JOINT", "home")
        _teach(builder, arm6, 21, Q1, "JOINT", "pick")
        assert "FINISHED" in bpy.ops.kinema.generate_motion()

        bpy.context.scene.frame_set(1)
        assert not arm6.kinema_ik_enabled
        assert np.allclose(_q(builder, arm6), Q0, atol=1e-6)

        bpy.context.scene.frame_set(21)
        assert np.allclose(_q(builder, arm6), Q1, atol=1e-6)

        # And it actually moved in between, rather than snapping at the end.
        bpy.context.scene.frame_set(11)
        middle = _q(builder, arm6)
        assert not np.allclose(middle, Q0, atol=1e-3)
        assert not np.allclose(middle, Q1, atol=1e-3)

    def test_a_linear_span_drives_the_tool_along_a_straight_line(
        self, arm6, builder
    ):
        """The claim the move type makes, measured on the tool itself."""
        import bpy

        _teach(builder, arm6, 1, Q1, "JOINT", "start")
        start = _tool(builder, arm6).copy()

        # A second waypoint offset in space, taught by moving the IK goal.
        bpy.context.scene.frame_set(21)
        _drag_goal(builder, arm6, (0.0, 0.2, -0.12))
        assert "FINISHED" in bpy.ops.kinema.add_waypoint(name="cut")
        arm6.kinema_waypoints[-1].move = "LINEAR"
        end = _tool(builder, arm6).copy()
        arm6.kinema_ik_enabled = False

        assert "FINISHED" in bpy.ops.kinema.generate_motion()

        segment = end - start
        worst = 0.0
        for frame in range(1, 22):
            bpy.context.scene.frame_set(frame)
            assert arm6.kinema_ik_enabled, f"IK is off on frame {frame}"
            point = _tool(builder, arm6)
            along = float(np.dot(point - start, segment) / np.dot(segment, segment))
            worst = max(
                worst, float(np.linalg.norm((point - start) - along * segment))
            )
        # 0.1 mm over a 230 mm move. Measured at 0.000 mm; the tolerance is for
        # the solver, not for the claim.
        assert worst < 1e-4, f"tool left the line by {worst * 1000:.3f} mm"

    def test_a_joint_span_then_a_linear_one_toggles_ik_per_frame(
        self, arm6, builder
    ):
        """The mixed case, which is the whole point of gating IK by keyframe.

        A lone linear span proves nothing here: the property is simply left on,
        so the flag reads True whether or not it was ever keyed. Only a job that
        changes move type needs the curve, and only this catches its absence.
        """
        import bpy

        _teach(builder, arm6, 1, Q0, "JOINT", "home")
        _teach(builder, arm6, 21, Q1, "JOINT", "approach")

        bpy.context.scene.frame_set(41)
        _set_q(builder, arm6, Q1)
        _drag_goal(builder, arm6, (0.0, 0.18, -0.1))
        assert "FINISHED" in bpy.ops.kinema.add_waypoint(name="cut")
        arm6.kinema_waypoints[-1].move = "LINEAR"
        arm6.kinema_ik_enabled = False

        assert "FINISHED" in bpy.ops.kinema.generate_motion()

        for frame in (1, 11, 20):
            bpy.context.scene.frame_set(frame)
            assert not arm6.kinema_ik_enabled, f"IK is on during a joint move ({frame})"
        for frame in (21, 31, 41):
            bpy.context.scene.frame_set(frame)
            assert arm6.kinema_ik_enabled, f"IK is off during a linear move ({frame})"

    def test_the_goal_curves_come_out_linear(self, arm6, builder):
        """A linear move should arrive linear, not eased.

        Not about straightness -- Bezier between two keys still traces the same
        line, because x, y and z share the shape and only the speed along it
        changes. It is about the default: easing is something to add in the
        graph editor, not something to have to remove.
        """
        import bpy

        _teach(builder, arm6, 1, Q1, "JOINT", "start")
        bpy.context.scene.frame_set(21)
        _drag_goal(builder, arm6, (0.0, 0.2, 0.0))
        bpy.ops.kinema.add_waypoint(name="cut")
        arm6.kinema_waypoints[-1].move = "LINEAR"
        arm6.kinema_ik_enabled = False

        assert "FINISHED" in bpy.ops.kinema.generate_motion()

        ik_name = arm6.get(builder.PROP_IK_BONE)
        goal_curves = [
            curve
            for curve in _curves(arm6)
            if curve.data_path.startswith(f'pose.bones["{ik_name}"]')
        ]
        assert goal_curves, "the linear span keyed no goal curves"
        for curve in goal_curves:
            for key in curve.keyframe_points:
                assert key.interpolation == "LINEAR", curve.data_path

    def test_regenerating_replaces_the_previous_motion(self, arm6, builder):
        """Old keys surviving would make the robot visit frames nobody asked for."""
        import bpy

        _teach(builder, arm6, 1, Q0, "JOINT", "home")
        point = _teach(builder, arm6, 40, Q1, "JOINT", "pick")
        assert "FINISHED" in bpy.ops.kinema.generate_motion()

        point.frame = 20
        assert "FINISHED" in bpy.ops.kinema.generate_motion()

        frames = {
            key.co[0]
            for container in _curves(arm6)
            for key in container.keyframe_points
        }
        assert 40.0 not in frames, "a key from the first generation survived"
        assert {1.0, 20.0} <= frames

    def test_a_moved_marker_re_aims_a_linear_move(self, arm6, builder):
        """The marker is a control, not a read-out.

        Recording an Empty is only worth doing if dragging it moves the
        waypoint -- snapping one to a feature on the part is the whole reason
        issue #3 asked for it. A marker that changed nothing would be
        decoration.
        """
        import bpy
        from mathutils import Vector

        _teach(builder, arm6, 1, Q1, "JOINT", "start")
        cut = _teach(builder, arm6, 21, Q1, "LINEAR", "cut")

        moved = cut.marker.matrix_basis.copy()
        moved.translation = moved.translation + Vector((0.0, 0.22, -0.05))
        cut.marker.matrix_basis = moved
        bpy.context.view_layer.update()

        assert "FINISHED" in bpy.ops.kinema.generate_motion()

        bpy.context.scene.frame_set(21)
        assert np.allclose(
            _tool(builder, arm6), np.array(moved.translation), atol=1e-4
        ), "the linear move ignored the marker"

    def test_a_moved_marker_on_a_joint_move_is_reported(self, arm6, builder):
        """It cannot follow one, so saying nothing would be the worst answer."""
        import bpy
        from mathutils import Vector

        _teach(builder, arm6, 1, Q0, "JOINT", "home")
        pick = _teach(builder, arm6, 21, Q1, "JOINT", "pick")

        moved = pick.marker.matrix_basis.copy()
        moved.translation = moved.translation + Vector((0.0, 0.2, 0.0))
        pick.marker.matrix_basis = moved
        bpy.context.view_layer.update()

        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        # The joint move still replays what it was taught.
        bpy.context.scene.frame_set(21)
        assert np.allclose(_q(builder, arm6), Q1, atol=1e-6)

    def test_a_linear_finish_switches_ik_off_afterwards(self, arm6, builder):
        """Otherwise the solver runs for every frame after the job forever."""
        import bpy

        _teach(builder, arm6, 1, Q1, "JOINT", "start")
        bpy.context.scene.frame_set(21)
        _drag_goal(builder, arm6, (0.0, 0.15, 0.0))
        bpy.ops.kinema.add_waypoint(name="cut")
        arm6.kinema_waypoints[-1].move = "LINEAR"

        assert "FINISHED" in bpy.ops.kinema.generate_motion()

        bpy.context.scene.frame_set(21)
        assert arm6.kinema_ik_enabled
        bpy.context.scene.frame_set(40)
        assert not arm6.kinema_ik_enabled, "IK is still live past the end of the job"

    def test_regenerating_keeps_animation_outside_the_job(self, arm6, builder):
        """These are ordinary channels; an animator may be using them too."""
        import bpy

        _teach(builder, arm6, 20, Q0, "JOINT", "home")
        _teach(builder, arm6, 40, Q1, "JOINT", "pick")
        assert "FINISHED" in bpy.ops.kinema.generate_motion()

        # A hand-keyed pose well after the job.
        bpy.context.scene.frame_set(90)
        _set_q(builder, arm6, [0.3] * 6)
        for pose_bone in builder.joint_bones(arm6):
            pose_bone.keyframe_insert(data_path="rotation_euler", index=1, frame=90)

        assert "FINISHED" in bpy.ops.kinema.generate_motion()

        bpy.context.scene.frame_set(90)
        assert np.allclose(_q(builder, arm6), [0.3] * 6, atol=1e-6), (
            "regeneration ate a key outside its own range"
        )

    def test_a_joint_job_does_not_claim_the_frame_after_it(self, arm6, builder):
        """The first free frame is the user's, unless the job wrote there.

        A linear finish leaves a switch-off key one past the end and so owns
        that frame. A joint finish writes nothing there -- and claiming it
        anyway would have the next regeneration delete a key put on exactly
        the frame an animator would reach for first.
        """
        import bpy

        _teach(builder, arm6, 1, Q0, "JOINT", "home")
        _teach(builder, arm6, 20, Q1, "JOINT", "pick")
        assert "FINISHED" in bpy.ops.kinema.generate_motion()

        bpy.context.scene.frame_set(21)
        _set_q(builder, arm6, [0.25] * 6)
        for pose_bone in builder.joint_bones(arm6):
            pose_bone.keyframe_insert(data_path="rotation_euler", index=1, frame=21)

        assert "FINISHED" in bpy.ops.kinema.generate_motion()

        bpy.context.scene.frame_set(21)
        assert np.allclose(_q(builder, arm6), [0.25] * 6, atol=1e-6), (
            "regeneration claimed the frame after a joint-ending job"
        )

    def test_removing_the_last_row_leaves_a_valid_index(self, arm6, builder):
        import bpy

        _teach(builder, arm6, 1, Q0, "JOINT", "home")
        assert "FINISHED" in bpy.ops.kinema.remove_waypoint(index=0)
        assert arm6.kinema_active_waypoint == 0

    def test_a_linear_span_without_an_ik_target_is_refused(
        self, addon, builder, fixture_dir, clean_scene
    ):
        """Better than generating a move that silently does not happen."""
        import bpy

        assert "FINISHED" in bpy.ops.kinema.build_robot(
            filepath=str(fixture_dir / "arm6.urdf")
        )
        rig = next(o for o in bpy.data.objects if builder.is_kinema_rig(o))
        bpy.context.view_layer.objects.active = rig
        rig.select_set(True)

        _teach(builder, rig, 1, Q0, "JOINT", "home")
        _teach(builder, rig, 21, Q1, "LINEAR", "cut")
        with pytest.raises(RuntimeError, match="need an IK target"):
            bpy.ops.kinema.generate_motion()


def _turn(a, b) -> float:
    """Degrees between two orientations, the short way."""
    angle = a.rotation_difference(b).angle
    return math.degrees(min(angle, 2.0 * math.pi - angle))


def _tool_rotation(builder, rig, frame):
    import bpy

    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()
    name = rig.get(builder.PROP_TCP_BONE) or builder.TCP_BONE
    return rig.pose.bones[name].matrix.to_quaternion()


def _played(builder, rig, frames):
    """The joint values playback shows on each frame, solver and all."""
    import bpy

    seen = {}
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        seen[frame] = _q(builder, rig)
    return seen


def _raw_rotations(builder, rig, wp_ops, *waypoints):
    """How the IK target's rotation channel comes out for each waypoint's pose.

    What keying the pose as it came would store, sign and all.
    """
    goal = rig.pose.bones[rig.get(builder.PROP_IK_BONE)]
    goal.rotation_mode = "QUATERNION"
    rotations = []
    for waypoint in waypoints:
        goal.matrix = wp_ops.pose_of(waypoint)
        rotations.append(goal.rotation_quaternion.copy())
    return rotations


class TestRotation:
    def test_a_linear_move_turns_the_short_way(self, arm6, builder, wp_ops):
        """Keyed in opposite hemispheres, a 20 degree turn went 340 the other way."""
        import bpy

        a = _teach(builder, arm6, 1, WRIST + [FLIP - math.radians(10)], "JOINT", "a")
        b = _teach(builder, arm6, 21, WRIST + [FLIP + math.radians(10)], "LINEAR", "b")
        first, last = _raw_rotations(builder, arm6, wp_ops, a, b)
        assert first.dot(last) < 0.0, (
            "the two poses come out in one hemisphere, so this proves nothing"
        )

        assert "FINISHED" in bpy.ops.kinema.generate_motion()

        # The target first. The tool alone would not show it: seeded from the
        # taught configurations, IK cannot follow a target going the long way
        # round -- joint 6 would run out of travel -- and stays near its seed.
        goal = arm6.pose.bones[arm6.get(builder.PROP_IK_BONE)]
        turned = {}
        for frame in (1, 11, 21):
            _tool_rotation(builder, arm6, frame)
            turned[frame] = goal.matrix.to_quaternion()
        assert _turn(turned[1], turned[11]) == pytest.approx(10.0, abs=0.5)
        assert _turn(turned[1], turned[21]) == pytest.approx(20.0, abs=0.5)

        start = _tool_rotation(builder, arm6, 1)
        assert _turn(start, _tool_rotation(builder, arm6, 11)) == pytest.approx(10.0, abs=0.5)
        assert _turn(start, _tool_rotation(builder, arm6, 21)) == pytest.approx(20.0, abs=0.5)

    def test_a_long_turn_follows_the_shortest_one_on_every_frame(
        self, arm6, builder, wp_ops
    ):
        """Blender interpolates the four channels independently.

        That keeps to the right path but drifts off its pace the longer the
        turn, so a long turn is keyed in pieces.
        """
        import bpy

        a = _teach(builder, arm6, 1, WRIST + [FLIP - 1.0], "JOINT", "a")
        b = _teach(builder, arm6, 21, WRIST + [FLIP + 1.0], "LINEAR", "b")
        assert "FINISHED" in bpy.ops.kinema.generate_motion()

        start = wp_ops.pose_of(a).to_quaternion()
        end = wp_ops.pose_of(b).to_quaternion()
        end.make_compatible(start)
        goal = arm6.pose.bones[arm6.get(builder.PROP_IK_BONE)]
        worst = 0.0
        for frame in range(1, 22):
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()
            expected = start.slerp(end, (frame - 1) / 20)
            worst = max(worst, _turn(goal.matrix.to_quaternion(), expected))
        assert worst < 0.1, f"the target strayed {worst:.3f} degrees off the turn"

        keys = {
            point.co[0]
            for curve in _curves(arm6)
            if curve.data_path.endswith(".rotation_quaternion")
            for point in curve.keyframe_points
        }
        assert len(keys) > 2, "a 115 degree turn was keyed as one piece"


class TestConfiguration:
    def test_a_linear_move_is_seeded_from_its_taught_configurations(
        self, arm6, builder, ik_ops
    ):
        """IK solves the line from the configurations its two ends were taught in.

        Without keys there, it seeded from the next joint move's first key,
        and solved the whole line in that configuration instead.
        """
        import bpy

        _teach(builder, arm6, 1, REACH_A, "JOINT", "a")
        _teach(builder, arm6, 21, REACH_B, "LINEAR", "b")
        _teach(builder, arm6, 41, REACH_C, "JOINT", "c")
        assert "FINISHED" in bpy.ops.kinema.generate_motion()

        curves = ik_ops.joint_curves(arm6)
        for index, pose_bone in enumerate(builder.joint_bones(arm6)):
            curve = curves[pose_bone.name]
            assert curve.evaluate(1) == pytest.approx(REACH_A[index], abs=1e-5)
            assert curve.evaluate(21) == pytest.approx(REACH_B[index], abs=1e-5)

    def test_a_linear_move_that_would_change_configuration_is_refused(
        self, arm6, builder, ik_ops, addon
    ):
        """A straight line cannot take the arm from one configuration to another."""
        import bpy

        branches = importlib.import_module(f"{addon.__name__}.solver.branches")

        _teach(builder, arm6, 1, REACH_A, "JOINT", "a")
        # B at a pose along from A, but re-taught in another configuration.
        bpy.context.scene.frame_set(21)
        _set_q(builder, arm6, REACH_B)
        bpy.ops.kinema.snap_ik()
        assert "FINISHED" in bpy.ops.kinema.find_solutions(seeds=30)
        solutions = [np.array(values) for values in ik_ops._read_solutions(arm6)]
        other = max(
            range(len(solutions)),
            key=lambda i: branches.joint_distance(solutions[i], REACH_B),
        )
        assert "FINISHED" in bpy.ops.kinema.apply_solution(index=other)
        assert branches.joint_distance(_q(builder, arm6), REACH_B) >= (
            branches.DISTINCT_TOLERANCE
        ), "no other configuration was found, so this proves nothing"
        assert "FINISHED" in bpy.ops.kinema.add_waypoint(name="b")
        arm6.kinema_waypoints[-1].move = "LINEAR"
        _teach(builder, arm6, 41, REACH_C, "JOINT", "c")

        with pytest.raises(RuntimeError, match="can't change configuration"):
            bpy.ops.kinema.generate_motion()
        assert _curves(arm6) == [], "the refused job was keyed anyway"

    def test_an_end_taught_a_whole_turn_away_keeps_the_turn_it_arrives_with(
        self, wide6, builder, addon
    ):
        """The same pose with a wrist joint a whole turn round.

        A line cannot change a joint's turns. Seeded from the taught turn, IK
        was pulled over to it part-way along the line, a whole turn in one
        frame. The end takes the turn the line arrives with instead, and the
        motion check says so, for anyone who meant the other one.
        """
        import bpy

        check = importlib.import_module(f"{addon.__name__}.rig.motion_check")
        _teach(builder, wide6, 1, WRIST + [FLIP - 0.2], "JOINT", "a")
        end = _teach(builder, wide6, 21, WRIST + [FLIP + 0.2], "LINEAR", "b")
        arrives = list(end.q)[5]
        wound = list(end.q)
        wound[5] -= 2.0 * math.pi
        end.q = wound

        assert "FINISHED" in bpy.ops.kinema.generate_motion()

        assert list(end.q)[5] == pytest.approx(arrives, abs=1e-3)
        played = _played(builder, wide6, range(1, 22))
        worst = max(
            float(np.max(np.abs(played[frame] - played[frame - 1])))
            for frame in range(2, 22)
        )
        assert worst < 0.2, f"a joint jumped {math.degrees(worst):.0f} degrees in a frame"
        found = check.problems(wide6.kinema_motion_check[0])
        assert any("joint6" in problem and "taught at" in problem for problem in found), (
            found
        )


class TestBakingAJob:
    """What Bake IK makes of a generated job is what the job played."""

    def test_a_joint_move_survives_baking(self, arm6, builder):
        import bpy

        _teach(builder, arm6, 1, Q0, "JOINT", "home")
        _teach(builder, arm6, 21, Q1, "JOINT", "pick")
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        played = _played(builder, arm6, range(1, 22))
        assert not np.allclose(played[11], Q0, atol=1e-3), "the job never moved"

        assert "FINISHED" in bpy.ops.kinema.bake_ik(frame_start=1, frame_end=21)
        arm6.kinema_solver_mode = "OFF"
        baked = _played(builder, arm6, range(1, 22))
        worst = max(float(np.max(np.abs(baked[f] - played[f]))) for f in played)
        assert worst < 1e-5, f"baking moved a joint by {worst:.4f} rad"

    def test_a_mixed_job_bakes_to_what_it_played(self, arm6, builder):
        """Joint moves from their curves, the linear move from IK seeded as playback seeds it."""
        import bpy

        _teach(builder, arm6, 1, REACH_A, "JOINT", "a")
        _teach(builder, arm6, 21, REACH_B, "LINEAR", "b")
        _teach(builder, arm6, 41, REACH_C, "JOINT", "c")
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        played = _played(builder, arm6, range(1, 42))

        assert "FINISHED" in bpy.ops.kinema.bake_ik(frame_start=1, frame_end=41)
        arm6.kinema_solver_mode = "OFF"
        baked = _played(builder, arm6, range(1, 42))
        worst = max(float(np.max(np.abs(baked[f] - played[f]))) for f in played)
        assert worst < 1e-4, f"baking moved a joint by {worst:.4f} rad"

    def test_baking_switches_the_keyed_live_ik_off_over_its_range(self, arm6, builder):
        """Setting the property alone came undone on the next frame change.

        The job keys the switch, so its curve put the job's value back and live
        IK ran over the baked joints. Past the baked range the job keeps it.
        """
        import bpy

        _teach(builder, arm6, 1, REACH_A, "JOINT", "a")
        _teach(builder, arm6, 21, REACH_B, "LINEAR", "b")
        _teach(builder, arm6, 41, REACH_C, "JOINT", "c")
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        bpy.context.scene.frame_set(15)
        assert arm6.kinema_ik_enabled, "the job never switched live IK on"

        assert "FINISHED" in bpy.ops.kinema.bake_ik(frame_start=1, frame_end=10)

        for frame in (1, 5, 10):
            bpy.context.scene.frame_set(frame)
            assert not arm6.kinema_ik_enabled, f"live IK is back on at frame {frame}"
        bpy.context.scene.frame_set(15)
        assert arm6.kinema_ik_enabled, "the bake switched live IK off past its range"

    def test_a_layered_job_bakes_to_what_it_played(self, arm6, builder):
        """An NLA strip underneath changes what plays.

        A base strip holds joint1 at 0.3 and the job's action is added on top.
        ``keyframe_insert`` maps a value back through that stack, so the bake
        has to keep the played value; keeping the action's own took the base
        layer out again, 0.3 rad on every frame.
        """
        import bpy

        _teach(builder, arm6, 1, Q0, "JOINT", "home")
        _teach(builder, arm6, 21, Q1, "JOINT", "pick")
        assert "FINISHED" in bpy.ops.kinema.generate_motion()

        anim = arm6.animation_data
        job, job_slot = anim.action, anim.action_slot
        base = bpy.data.actions.new("base")
        anim.action = base
        first = builder.joint_bones(arm6)[0]
        first.rotation_euler[1] = 0.3
        first.keyframe_insert(data_path="rotation_euler", index=1, frame=1)
        base_slot = anim.action_slot
        anim.action = job
        anim.action_slot = job_slot
        strip = anim.nla_tracks.new().strips.new("base", 1, base)
        if hasattr(strip, "action_slot"):
            strip.action_slot = base_slot
        strip.extrapolation = "HOLD"
        anim.action_blend_type = "ADD"

        played = _played(builder, arm6, range(1, 22))
        assert played[1][0] == pytest.approx(Q0[0] + 0.3, abs=1e-5), (
            "the base layer is not in effect, so this proves nothing"
        )

        assert "FINISHED" in bpy.ops.kinema.bake_ik(frame_start=1, frame_end=21)
        arm6.kinema_solver_mode = "OFF"
        baked = _played(builder, arm6, range(1, 22))
        worst = max(float(np.max(np.abs(baked[f] - played[f]))) for f in played)
        assert worst < 1e-5, f"baking moved a joint by {worst:.4f} rad"


class TestMotionCheck:
    def test_generating_checks_every_move(self, arm6, builder):
        import bpy

        _teach(builder, arm6, 1, REACH_A, "JOINT", "a")
        _teach(builder, arm6, 21, REACH_B, "LINEAR", "b")
        _teach(builder, arm6, 41, REACH_C, "JOINT", "c")
        assert "FINISHED" in bpy.ops.kinema.generate_motion()

        rows = [
            (row.name, row.move, row.start_frame, row.end_frame)
            for row in arm6.kinema_motion_check
        ]
        assert rows == [("b", "LINEAR", 1, 21), ("c", "JOINT", 21, 41)]
        linear = arm6.kinema_motion_check[0]
        assert 0.0 <= linear.line_error < 1e-4
        assert arm6.kinema_motion_check[1].line_error < 0.0

    def test_a_joint_over_its_speed_limit_is_flagged(self, arm6, builder, addon):
        """Two radians in four frames: 12 rad/s against arm6's 3.15."""
        import bpy

        check = importlib.import_module(f"{addon.__name__}.rig.motion_check")
        _teach(builder, arm6, 1, Q0, "JOINT", "home")
        _teach(builder, arm6, 5, [2.0] + Q0[1:], "JOINT", "swing")
        assert "FINISHED" in bpy.ops.kinema.generate_motion()

        found = check.problems(arm6.kinema_motion_check[0])
        assert any(
            problem.startswith("joint1 at") and "speed limit" in problem
            for problem in found
        ), found

    def test_a_turn_the_wrist_cannot_make_is_flagged(self, arm6, builder, addon):
        """Joint 6 stops at 3.14 rad, so it cannot turn across 180 degrees.

        Twenty degrees the short way from +170 to -170 is out of its reach,
        and the tool cannot keep to the turn. Before the target turned the short
        way this went unnoticed: the tool went 340 degrees the long way instead.
        """
        import bpy

        check = importlib.import_module(f"{addon.__name__}.rig.motion_check")
        base = [0.0, -0.6, 1.0, 0.0, 0.4]
        _teach(builder, arm6, 1, base + [math.radians(170)], "JOINT", "a")
        _teach(builder, arm6, 21, base + [math.radians(-170)], "LINEAR", "b")
        assert "FINISHED" in bpy.ops.kinema.generate_motion()

        assert check.problems(arm6.kinema_motion_check[0])

    def test_check_motion_measures_the_job_again(self, arm6, builder):
        import bpy

        _teach(builder, arm6, 1, Q0, "JOINT", "home")
        _teach(builder, arm6, 21, Q1, "JOINT", "pick")
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        arm6.kinema_motion_check.clear()

        assert "FINISHED" in bpy.ops.kinema.check_motion()
        assert [row.name for row in arm6.kinema_motion_check] == ["pick"]

    def test_a_job_that_cannot_be_measured_is_not_reported_clean(self, arm6, builder):
        """Without a TCP there is no line to measure; no rows, and it says so."""
        import bpy

        _teach(builder, arm6, 1, Q0, "JOINT", "home")
        _teach(builder, arm6, 21, Q1, "JOINT", "pick")
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        arm6[builder.PROP_TCP_BONE] = "no such bone"

        assert bpy.ops.kinema.check_motion() == {"CANCELLED"}
        assert len(arm6.kinema_motion_check) == 0

    def test_the_check_knows_when_the_job_has_changed(self, arm6, builder, wp_ops):
        """Findings about a job since edited must not pass as current."""
        import bpy

        _teach(builder, arm6, 1, Q0, "JOINT", "home")
        _teach(builder, arm6, 21, Q1, "JOINT", "pick")
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        assert arm6.kinema_motion_check_job == wp_ops.job_signature(arm6)

        arm6.kinema_waypoints[1].frame = 30
        assert arm6.kinema_motion_check_job != wp_ops.job_signature(arm6)


def _curves(rig):
    """Every F-curve on the rig's own action, whatever the action layout."""
    anim = rig.animation_data
    action = anim.action if anim else None
    if action is None:
        return []
    layers = getattr(action, "layers", None)
    if layers:
        return [
            curve
            for layer in layers
            for strip in getattr(layer, "strips", ())
            for bag in getattr(strip, "channelbags", ())
            for curve in bag.fcurves
        ]
    return list(action.fcurves)
