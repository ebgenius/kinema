"""Waypoint recording and motion generation. Needs a real ``bpy``.

The claims worth holding onto:

* a waypoint remembers the *configuration* it was taught in, not just a pose,
* a JOINT span moves the joints and leaves the solver out of it,
* a LINEAR span drives the tool along the straight line between two poses,
* regenerating replaces the previous motion rather than layering on it.
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
def wp_ops(addon):
    return importlib.import_module(f"{addon.__name__}.ops.waypoints")


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


Q0 = [0.0, -0.6, 1.0, 0.0, 0.6, 0.0]
Q1 = [0.8, -0.9, 1.3, 0.2, 0.5, 0.3]


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
        from mathutils import Vector

        _teach(builder, arm6, 1, Q1, "JOINT", "start")
        start = _tool(builder, arm6).copy()

        # A second waypoint offset in space, taught by moving the IK goal.
        bpy.context.scene.frame_set(21)
        goal = arm6.pose.bones[arm6.get(builder.PROP_IK_BONE)]
        tcp_name = arm6.get(builder.PROP_TCP_BONE) or builder.TCP_BONE
        goal.matrix = arm6.pose.bones[tcp_name].matrix.copy()
        goal.matrix.translation = goal.matrix.translation + Vector((0.0, 0.2, -0.12))
        arm6.kinema_ik_enabled = True
        bpy.context.view_layer.update()
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
        from mathutils import Vector

        _teach(builder, arm6, 1, Q0, "JOINT", "home")
        _teach(builder, arm6, 21, Q1, "JOINT", "approach")

        bpy.context.scene.frame_set(41)
        _set_q(builder, arm6, Q1)
        goal = arm6.pose.bones[arm6.get(builder.PROP_IK_BONE)]
        tcp_name = arm6.get(builder.PROP_TCP_BONE) or builder.TCP_BONE
        goal.matrix = arm6.pose.bones[tcp_name].matrix.copy()
        goal.matrix.translation = goal.matrix.translation + Vector((0.0, 0.18, -0.1))
        arm6.kinema_ik_enabled = True
        bpy.context.view_layer.update()
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
        from mathutils import Vector

        _teach(builder, arm6, 1, Q1, "JOINT", "start")
        bpy.context.scene.frame_set(21)
        goal = arm6.pose.bones[arm6.get(builder.PROP_IK_BONE)]
        tcp_name = arm6.get(builder.PROP_TCP_BONE) or builder.TCP_BONE
        goal.matrix = arm6.pose.bones[tcp_name].matrix.copy()
        goal.matrix.translation = goal.matrix.translation + Vector((0.0, 0.2, 0.0))
        arm6.kinema_ik_enabled = True
        bpy.context.view_layer.update()
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
        from mathutils import Vector

        _teach(builder, arm6, 1, Q1, "JOINT", "start")
        goal = arm6.pose.bones[arm6.get(builder.PROP_IK_BONE)]
        tcp_name = arm6.get(builder.PROP_TCP_BONE) or builder.TCP_BONE
        bpy.context.scene.frame_set(21)
        goal.matrix = arm6.pose.bones[tcp_name].matrix.copy()
        goal.matrix.translation = goal.matrix.translation + Vector((0.0, 0.15, 0.0))
        arm6.kinema_ik_enabled = True
        bpy.context.view_layer.update()
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
