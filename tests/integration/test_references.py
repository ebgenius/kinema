"""Waypoints taught relative to an object (#96): ops/references.py. Needs a real ``bpy``.

The claims:

* a waypoint's marker is parented to its object without an inverse, so its own
  transform is the pose in the object's space, and changing the object keeps
  the waypoint where it is;
* Record goes on relative to what the highlighted waypoint is;
* a job taught relative to an Empty follows the Empty when it is moved, and
  when it is animated: a linear move, and a hold, stay on it on every frame;
* parenting a marker with Blender's own Make Parent without Inverse is read;
* the job's fingerprint follows the object, but not the frame it is shown on.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from ..conftest import requires_bpy
from .test_waypoints import FLIP, REACH_A, REACH_B, WRIST, _q, _set_q, _teach

pytestmark = requires_bpy

#: How close the tool has to be to where the object has the waypoint: 0.1 mm.
CLOSE = 1e-4


@pytest.fixture
def references(addon):
    return importlib.import_module(f"{addon.__name__}.ops.references")


@pytest.fixture
def builder(addon):
    return importlib.import_module(f"{addon.__name__}.rig.builder")


@pytest.fixture
def wp_ops(addon):
    return importlib.import_module(f"{addon.__name__}.ops.waypoints")


@pytest.fixture
def curves(addon):
    """An object's own F-curves, as Blender 5's slotted actions hold them."""
    ik = importlib.import_module(f"{addon.__name__}.ops.ik")
    return lambda obj: [c for container in ik.own_fcurve_containers(obj) for c in container]


@pytest.fixture
def arm6(addon, fixture_dir, clean_scene, builder):
    """A 6-DoF rig with an IK target, which linear moves and holds on a moving
    object need."""
    import bpy

    assert "FINISHED" in bpy.ops.kinema.build_robot(filepath=str(fixture_dir / "arm6.urdf"))
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


@pytest.fixture
def external_axes(addon):
    return importlib.import_module(f"{addon.__name__}.ops.external_axes")


def _empty(name, rig=None):
    """A new Empty in the scene, where ``rig`` is if given."""
    import bpy

    empty = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(empty)
    if rig is not None:
        empty.matrix_world = rig.matrix_world.copy()
    bpy.context.view_layer.update()
    return empty


def _parent_by_hand(marker, parent, active):
    """Blender's Make Parent without Inverse, Keep Transform, on ``marker``."""
    import bpy

    bpy.ops.object.select_all(action="DESELECT")
    marker.select_set(True)
    parent.select_set(True)
    bpy.context.view_layer.objects.active = parent
    assert "FINISHED" in bpy.ops.object.parent_no_inverse_set(keep_transform=True)
    bpy.context.view_layer.objects.active = active


def _leave(rig, start, end, first, last, move="JOINT"):
    """A motion from ``start`` to ``end`` over ``first``-``last``."""
    motion = rig.kinema_motions.add()
    motion.name, motion.start, motion.end = "leave", start.uid, end.uid
    motion.frame_start, motion.frame_end, motion.move = first, last, move
    return motion


@pytest.fixture
def fixture_empty(arm6):
    """A workpiece Empty in front of the robot, still for now."""
    import bpy

    empty = bpy.data.objects.new("Workpiece", None)
    bpy.context.scene.collection.objects.link(empty)
    empty.matrix_world = arm6.matrix_world.copy()
    bpy.context.view_layer.update()
    return empty


def _tool_world(builder, rig):
    """The tool's pose in world space, as the scene stands."""
    import bpy

    bpy.context.view_layer.update()
    name = rig.get(builder.PROP_TCP_BONE) or builder.TCP_BONE
    return np.array(rig.matrix_world @ rig.pose.bones[name].matrix)


def _on(empty, local):
    """Where ``local``, in ``empty``'s space, is in world space now."""
    return np.array(empty.matrix_world) @ np.array(local)


def _relative(rig, waypoint, empty):
    """Make ``empty`` the waypoint's frame of reference through its field."""
    waypoint.reference = empty
    assert waypoint.marker.parent == empty


def _key_slide(curves, empty, first, last, offset, back=False):
    """Animate ``empty`` sliding by ``offset`` from ``first`` to ``last``, linearly:
    and back again by ``last``, half-way there, with ``back``."""
    start = empty.location.copy()
    empty.location = start
    empty.keyframe_insert(data_path="location", frame=first)
    if back:
        empty.location = start + offset
        empty.keyframe_insert(data_path="location", frame=(first + last) // 2)
        empty.location = start
    else:
        empty.location = start + offset
    empty.keyframe_insert(data_path="location", frame=last)
    for curve in curves(empty):
        for point in curve.keyframe_points:
            point.interpolation = "LINEAR"


class TestTeaching:
    def test_the_marker_is_parented_without_an_inverse_and_stays_put(
        self, arm6, builder, fixture_empty
    ):
        """Its own transform is the pose in the object's space; nothing moved."""
        import bpy
        from mathutils import Matrix, Vector

        fixture_empty.location = Vector((0.1, -0.05, 0.02))
        bpy.context.view_layer.update()
        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        before = np.array(pick.marker.matrix_world)
        _relative(arm6, pick, fixture_empty)
        bpy.context.view_layer.update()

        assert pick.marker.matrix_parent_inverse == Matrix.Identity(4)
        np.testing.assert_allclose(np.array(pick.marker.matrix_world), before, atol=1e-6)
        np.testing.assert_allclose(
            _on(fixture_empty, pick.marker.matrix_basis), before, atol=1e-6
        )

    def test_record_goes_on_relative_to_the_highlighted_waypoint(
        self, arm6, builder, fixture_empty
    ):
        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _relative(arm6, pick, fixture_empty)
        place = _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        assert place.reference == fixture_empty
        assert place.marker.parent == fixture_empty
        np.testing.assert_allclose(
            np.array(place.marker.matrix_world), _tool_world(builder, arm6), atol=1e-6
        )

    def test_back_to_the_base_it_stays_put_and_is_taught_where_it_was(
        self, arm6, builder, wp_ops, fixture_empty
    ):
        import bpy
        from mathutils import Vector

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        taught = list(pick.pose)
        _relative(arm6, pick, fixture_empty)
        fixture_empty.location += Vector((0.02, 0.0, 0.0))
        bpy.context.view_layer.update()
        here = np.array(pick.marker.matrix_world)

        pick.reference = None
        bpy.context.view_layer.update()
        assert pick.marker.parent == arm6
        np.testing.assert_allclose(np.array(pick.marker.matrix_world), here, atol=1e-6)
        np.testing.assert_allclose(list(pick.pose), taught, atol=1e-9)
        assert wp_ops.marker_has_moved(pick), "the next generation walks the joints there"

    def test_the_robot_and_what_it_carries_are_no_frame_of_reference(
        self, arm6, builder, references, fixture_empty
    ):
        import bpy

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        carried = bpy.data.objects.new("Gripper", None)
        bpy.context.scene.collection.objects.link(carried)
        carried.parent = arm6
        assert not references.may_be_reference(pick, arm6)
        assert not references.may_be_reference(pick, carried)
        assert not references.may_be_reference(pick, pick.marker)
        assert references.may_be_reference(pick, fixture_empty)

    def test_blenders_parent_without_inverse_is_read(
        self, arm6, builder, wp_ops, fixture_empty
    ):
        """Shift+Ctrl+P with Keep Transform on the marker: it's relative to the
        Empty from then, and still where it was."""
        import bpy
        from mathutils import Vector

        fixture_empty.location = Vector((0.05, 0.02, 0.0))
        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        before = np.array(pick.marker.matrix_world)
        bpy.ops.object.select_all(action="DESELECT")
        pick.marker.select_set(True)
        fixture_empty.select_set(True)
        bpy.context.view_layer.objects.active = fixture_empty
        assert "FINISHED" in bpy.ops.object.parent_no_inverse_set(keep_transform=True)
        bpy.context.view_layer.objects.active = arm6

        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        assert pick.reference == fixture_empty
        bpy.context.scene.frame_set(1)
        np.testing.assert_allclose(_tool_world(builder, arm6), before, atol=CLOSE)


class TestFollowing:
    def test_a_job_on_an_empty_follows_it_when_it_is_moved(
        self, arm6, builder, fixture_empty
    ):
        """Done when, for #96: taught relative to an Empty, the job follows it."""
        import bpy
        from mathutils import Euler, Vector

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _relative(arm6, pick, fixture_empty)
        place = _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        local = {w.name: w.marker.matrix_basis.copy() for w in (pick, place)}
        fixture_empty.location += Vector((0.03, -0.02, 0.01))
        fixture_empty.rotation_euler = Euler((0.0, 0.0, 0.1))
        bpy.context.view_layer.update()

        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        for frame, name in ((1, "pick"), (21, "place")):
            bpy.context.scene.frame_set(frame)
            np.testing.assert_allclose(
                _tool_world(builder, arm6), _on(fixture_empty, local[name]), atol=CLOSE
            )

    def test_a_linear_move_on_a_moving_empty_stays_on_its_line_every_frame(
        self, arm6, builder, wp_ops, fixture_empty, curves
    ):
        """The line is the Empty's: on each frame, between where it has the two ends.
        The Empty goes 30 mm out and back across it: not tracked, the tool would
        run the straight line between where it has the ends on the first and
        last frames, the line as it stands at rest, and miss the moving one
        mid-way by 29.7 mm. A steady slide would not show it: that carries the
        moving line along a straight path too."""
        import bpy
        from mathutils import Vector

        _key_slide(curves, fixture_empty, 1, 41, Vector((0.0, 0.0, 0.03)), back=True)
        bpy.context.scene.frame_set(1)
        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _relative(arm6, pick, fixture_empty)
        place = _teach(builder, arm6, 41, REACH_B, "LINEAR", "place")
        a, b = pick.marker.matrix_basis.copy(), place.marker.matrix_basis.copy()

        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        worst = behind = 0.0
        for frame in range(1, 42):
            bpy.context.scene.frame_set(frame)
            start, end = _on(fixture_empty, a)[:3, 3], _on(fixture_empty, b)[:3, 3]
            tool = _tool_world(builder, arm6)[:3, 3]
            along = np.clip((tool - start) @ (end - start) / ((end - start) @ (end - start)), 0, 1)
            worst = max(worst, float(np.linalg.norm(tool - (start + along * (end - start)))))
            # And at its pace: where the moving line has the tool on this frame.
            paced = start + (frame - 1) / 40 * (end - start)
            behind = max(behind, float(np.linalg.norm(tool - paced)))
        assert worst < CLOSE, f"{worst * 1000:.3f} mm off the moving line"
        assert behind < CLOSE, f"{behind * 1000:.3f} mm from where it should be on the line"
        row = next(r for r in arm6.kinema_motion_check if r.name == "place")
        assert 0.0 <= row.line_error < CLOSE, "the Motion Check measures the moving line"

    def test_a_hold_on_a_moving_empty_follows_it(self, arm6, builder, fixture_empty, curves):
        """Held at 'place' over frames 21-41 while the Empty slides: the tool goes
        with it, as it would follow a conveyor."""
        import bpy
        from mathutils import Vector

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _relative(arm6, pick, fixture_empty)
        place = _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        local = place.marker.matrix_basis.copy()
        leave = arm6.kinema_motions.add()
        leave.name, leave.start, leave.end = "leave", place.uid, pick.uid
        leave.frame_start, leave.frame_end = 41, 61
        bpy.context.scene.frame_set(1)
        _key_slide(curves, fixture_empty, 21, 41, Vector((0.03, 0.0, 0.0)))

        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        for frame in (21, 26, 31, 36, 41):
            bpy.context.scene.frame_set(frame)
            np.testing.assert_allclose(
                _tool_world(builder, arm6), _on(fixture_empty, local), atol=CLOSE,
                err_msg=f"frame {frame}",
            )

    def test_go_to_puts_the_tool_where_the_empty_has_it(
        self, arm6, builder, fixture_empty
    ):
        import bpy
        from mathutils import Vector

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _relative(arm6, pick, fixture_empty)
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        local = pick.marker.matrix_basis.copy()
        fixture_empty.location += Vector((0.02, 0.01, 0.0))
        bpy.context.view_layer.update()
        _set_q(builder, arm6, REACH_B)

        assert "FINISHED" in bpy.ops.kinema.goto_waypoint(index=0)
        np.testing.assert_allclose(
            _tool_world(builder, arm6), _on(fixture_empty, local), atol=CLOSE
        )


class TestFingerprint:
    def test_moving_the_empty_outdates_the_job_and_scrubbing_does_not(
        self, arm6, builder, wp_ops, fixture_empty, curves
    ):
        import bpy
        from mathutils import Vector

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _relative(arm6, pick, fixture_empty)
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        _key_slide(curves, fixture_empty, 1, 21, Vector((0.02, 0.0, 0.0)))
        signed = wp_ops.job_signature(arm6)
        bpy.context.scene.frame_set(15)
        assert wp_ops.job_signature(arm6) == signed, "the frame shown is no edit"

        curve = curves(fixture_empty)[0]
        curve.keyframe_points[-1].co[1] += 0.01
        assert wp_ops.job_signature(arm6) != signed, "re-keyed, it goes elsewhere"

    def test_changing_how_the_keys_ease_outdates_the_job(
        self, arm6, builder, wp_ops, fixture_empty, curves
    ):
        """The same keys, held instead of slid: it is somewhere else between them."""
        from mathutils import Vector

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _relative(arm6, pick, fixture_empty)
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        _key_slide(curves, fixture_empty, 1, 21, Vector((0.02, 0.0, 0.0)))
        signed = wp_ops.job_signature(arm6)
        curves(fixture_empty)[0].keyframe_points[0].interpolation = "CONSTANT"
        assert wp_ops.job_signature(arm6) != signed

    def test_moving_a_channel_nothing_keys_outdates_the_job(
        self, arm6, builder, wp_ops, fixture_empty, curves
    ):
        import bpy
        from mathutils import Vector

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _relative(arm6, pick, fixture_empty)
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        _key_slide(curves, fixture_empty, 1, 21, Vector((0.02, 0.0, 0.0)))
        signed = wp_ops.job_signature(arm6)
        fixture_empty.rotation_euler.z += 0.1
        bpy.context.view_layer.update()
        assert wp_ops.job_signature(arm6) != signed, "turned, it has the waypoints elsewhere"

    def test_a_slide_pushed_down_to_the_nla_is_read_and_scrubbing_it_is_no_edit(
        self, arm6, builder, wp_ops, fixture_empty, curves
    ):
        import bpy
        from mathutils import Vector

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _relative(arm6, pick, fixture_empty)
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        _key_slide(curves, fixture_empty, 1, 21, Vector((0.02, 0.0, 0.0)))
        anim = fixture_empty.animation_data
        strip = anim.nla_tracks.new().strips.new("slide", 1, anim.action)
        anim.action = None
        bpy.context.scene.frame_set(1)
        signed = wp_ops.job_signature(arm6)
        bpy.context.scene.frame_set(15)
        assert wp_ops.job_signature(arm6) == signed, "the frame shown is no edit"
        strip.scale = 2.0
        assert wp_ops.job_signature(arm6) != signed, "slowed down, it is elsewhere"

    def test_moving_the_robot_outdates_a_job_on_an_object_and_no_other(
        self, arm6, builder, wp_ops, fixture_empty
    ):
        import bpy

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        at_base = wp_ops.job_signature(arm6)
        arm6.location.x += 0.03
        bpy.context.view_layer.update()
        assert wp_ops.job_signature(arm6) == at_base, "at the base, it goes with the robot"

        _relative(arm6, pick, fixture_empty)
        signed = wp_ops.job_signature(arm6)
        arm6.location.x += 0.03
        bpy.context.view_layer.update()
        assert wp_ops.job_signature(arm6) != signed, "the Empty is elsewhere from the robot"


class TestReparented:
    """A marker parented with Blender's own operators, which don't tell Kinema."""

    def test_choosing_an_object_after_it_keeps_where_the_joints_were_taught(
        self, arm6, builder, references, fixture_empty
    ):
        """Parented to one Empty by hand, then Relative To set to another."""
        import bpy
        from mathutils import Vector

        fixture_empty.location = Vector((0.05, 0.02, 0.0))
        bpy.context.view_layer.update()
        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        taught = np.array(pick.pose).reshape(4, 4)
        _parent_by_hand(pick.marker, fixture_empty, arm6)
        other = _empty("Shelf", arm6)

        pick.reference = other
        assert pick.marker.parent == other
        np.testing.assert_allclose(np.array(references.taught_pose(pick)), taught, atol=1e-6)

    def test_its_pose_is_read_in_its_new_frame(self, arm6, builder, wp_ops, fixture_empty):
        import bpy
        from mathutils import Vector

        fixture_empty.location = Vector((0.05, 0.02, 0.0))
        bpy.context.view_layer.update()
        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        _parent_by_hand(pick.marker, fixture_empty, arm6)

        wp_ops.job_plan(arm6)
        np.testing.assert_allclose(
            np.array(pick.pose).reshape(4, 4), np.array(pick.marker.matrix_basis), atol=1e-6
        )

    def test_adding_an_axis_moves_where_its_joints_were_taught(
        self, arm6, builder, wp_ops, references, external_axes, fixture_empty
    ):
        """Parented by hand, then an axis added before the robot: where its joints
        were taught goes with the robot, 100 mm along x."""
        import bpy
        from mathutils import Vector

        fixture_empty.location = Vector((0.05, 0.02, 0.0))
        bpy.context.view_layer.update()
        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        taught = np.array(pick.pose).reshape(4, 4)
        _parent_by_hand(pick.marker, fixture_empty, arm6)
        shift = np.eye(4)
        shift[0, 3] = 0.1

        external_axes._move_waypoints(arm6, shift)
        wp_ops.job_plan(arm6)
        np.testing.assert_allclose(
            np.array(references.taught_pose(pick)), shift @ taught, atol=1e-6
        )

    def test_parented_back_to_the_robot_an_axis_moves_where_its_joints_were_taught(
        self, arm6, builder, wp_ops, references, external_axes, fixture_empty
    ):
        """Taught on an Empty, parented back to the robot by hand, then an axis
        added before the robot: at the base again, it moves with the robot."""
        import bpy
        from mathutils import Vector

        fixture_empty.location = Vector((0.05, 0.02, 0.0))
        bpy.context.view_layer.update()
        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        taught = np.array(pick.pose).reshape(4, 4)
        _relative(arm6, pick, fixture_empty)
        _parent_by_hand(pick.marker, arm6, arm6)
        shift = np.eye(4)
        shift[0, 3] = 0.1

        external_axes._move_waypoints(arm6, shift)
        wp_ops.job_plan(arm6)
        np.testing.assert_allclose(np.array(pick.pose).reshape(4, 4), shift @ taught, atol=1e-6)

    def test_a_marker_given_a_parent_inverse_moves_with_the_robot_as_the_robot_does(
        self, arm6, builder, references, external_axes
    ):
        """On the rig with a parent inverse turned 90 degrees: the axis moves it
        100 mm along the robot's x, not the turned one."""
        import bpy
        from mathutils import Matrix

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        before = np.array(references.local_pose(pick))
        inverse = Matrix.Rotation(np.pi / 2, 4, "Z")
        pick.marker.matrix_parent_inverse = inverse
        pick.marker.matrix_basis = inverse.inverted() @ Matrix(before.tolist())
        bpy.context.view_layer.update()
        shift = np.eye(4)
        shift[0, 3] = 0.1

        external_axes._move_waypoints(arm6, shift)
        np.testing.assert_allclose(np.array(references.local_pose(pick)), shift @ before, atol=1e-6)

    def test_a_marker_parented_to_what_the_robot_carries_is_refused(
        self, arm6, builder, fixture_empty
    ):
        """The tool would chase its own tail: Generate, Record and Update say so."""
        import bpy

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        gripper = _empty("Gripper")
        gripper.parent = arm6
        pick.marker.parent = gripper
        arm6.kinema_active_waypoint = 0

        carried = "'pick': its marker is parented to 'Gripper', which the robot carries"
        with pytest.raises(RuntimeError, match=carried):
            bpy.ops.kinema.generate_motion()
        with pytest.raises(RuntimeError, match=carried):
            bpy.ops.kinema.add_waypoint(name="next")
        with pytest.raises(RuntimeError, match=carried):
            bpy.ops.kinema.update_waypoint(index=0)
        assert len(arm6.kinema_waypoints) == 2


class TestLost:
    def test_a_waypoint_whose_object_is_deleted_goes_back_where_it_was_taught(
        self, arm6, builder, fixture_empty
    ):
        """Its pose was in the Empty's space; with the Empty gone, it is where its
        joints were taught, relative to the robot."""
        import bpy
        from mathutils import Vector

        fixture_empty.location = Vector((0.05, 0.02, 0.0))
        bpy.context.view_layer.update()
        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        taught = np.array(pick.marker.matrix_world)
        _relative(arm6, pick, fixture_empty)
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        bpy.data.objects.remove(fixture_empty)

        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        assert pick.marker.parent == arm6
        assert pick.reference is None
        bpy.context.scene.frame_set(1)
        np.testing.assert_allclose(_tool_world(builder, arm6), taught, atol=CLOSE)

    def test_an_object_scaled_to_nothing_is_refused(self, arm6, builder, fixture_empty):
        """There is no space in it to be in: the field goes back, Record refuses."""
        import bpy

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        flat = _empty("Sheet", arm6)
        flat.scale = (0.0, 1.0, 1.0)
        bpy.context.view_layer.update()

        stored = list(pick.pose)
        pick.reference = flat
        assert pick.reference is None
        assert pick.marker.parent == arm6
        assert not any(pick.taught), "nothing about it changed"
        assert list(pick.pose) == stored

        _relative(arm6, pick, fixture_empty)
        fixture_empty.scale = (0.0, 1.0, 1.0)
        bpy.context.view_layer.update()
        arm6.kinema_active_waypoint = 0
        with pytest.raises(RuntimeError, match="'Workpiece' is scaled to nothing"):
            bpy.ops.kinema.add_waypoint(name="next")
        assert len(arm6.kinema_waypoints) == 1


class TestTurns:
    def test_a_pause_on_a_still_object_keeps_the_turn_the_line_arrived_with(
        self, wide6, builder
    ):
        """A line arrives at 'b' a whole turn from its taught wrist, then the robot
        pauses there over frames 21-41. The pause keeps the turn it arrived with:
        not the taught one by its end, a whole turn of the wrist while still."""
        import bpy

        empty = _empty("Workpiece", wide6)
        a = _teach(builder, wide6, 1, WRIST + [FLIP - 0.2], "JOINT", "a")
        _relative(wide6, a, empty)
        b = _teach(builder, wide6, 21, WRIST + [FLIP + 0.2], "LINEAR", "b")
        wound = list(b.q)
        wound[5] -= 2.0 * np.pi
        b.q = wound
        _leave(wide6, b, a, 41, 61)

        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        bpy.context.scene.frame_set(21)
        arrived = _q(builder, wide6)[5]
        for frame in range(22, 42):
            bpy.context.scene.frame_set(frame)
            assert _q(builder, wide6)[5] == pytest.approx(arrived, abs=1e-3), frame

    def test_a_hold_on_an_object_turning_a_whole_turn_turns_the_wrist_with_it(
        self, wide6, builder
    ):
        """Held at 'place' over frames 21-41 while its Empty turns once round the
        wrist. The wrist goes round with it, and ends the hold a turn on: the
        same pose as taught, but not the same joints."""
        import bpy

        pick = _teach(builder, wide6, 1, WRIST + [FLIP - 0.2], "JOINT", "pick")
        place = _teach(builder, wide6, 21, WRIST + [-2.9], "JOINT", "place")
        empty = _empty("Turntable")
        empty.matrix_world = place.marker.matrix_world.copy()
        bpy.context.view_layer.update()
        _relative(wide6, place, empty)
        previous = empty.rotation_euler.copy()
        for frame in range(21, 42):
            _set_q(builder, wide6, WRIST + [-2.9 + 2.0 * np.pi * (frame - 21) / 20])
            pose = _tool_world(builder, wide6)
            from mathutils import Matrix

            matrix = Matrix(pose.tolist())
            empty.location = matrix.translation
            empty.rotation_euler = matrix.to_euler("XYZ", previous)
            previous = empty.rotation_euler.copy()
            empty.keyframe_insert(data_path="location", frame=frame)
            empty.keyframe_insert(data_path="rotation_euler", frame=frame)
        _leave(wide6, place, pick, 41, 61)

        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        played = {}
        for frame in range(21, 42):
            bpy.context.scene.frame_set(frame)
            played[frame] = _q(builder, wide6)[5]
        jump = max(abs(played[f] - played[f - 1]) for f in range(22, 42))
        assert jump < 0.5, f"the wrist jumped {np.degrees(jump):.0f} degrees in a frame"
        assert played[41] == pytest.approx(-2.9 + 2.0 * np.pi, abs=0.05)


class TestPlacement:
    def test_go_to_out_of_reach_keeps_the_goal_on_the_taught_joints(
        self, arm6, builder, references, fixture_empty
    ):
        """Its Empty 5 m away: shown as taught, with the IK goal there too, so live
        IK does not drag the arm off them."""
        import bpy
        from mathutils import Vector

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _relative(arm6, pick, fixture_empty)
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        fixture_empty.location += Vector((5.0, 0.0, 0.0))
        bpy.context.view_layer.update()

        assert "FINISHED" in bpy.ops.kinema.goto_waypoint(index=0)
        goal = arm6.pose.bones[arm6[builder.PROP_IK_BONE]].matrix
        np.testing.assert_allclose(
            np.array(goal), np.array(references.taught_pose(pick)), atol=1e-5
        )

    def test_objects_of_one_name_from_two_files_are_told_apart(
        self, arm6, builder, tmp_path, fixture_empty
    ):
        """A 'Workpiece' here and one linked from a library, 100 mm apart: each
        waypoint is on its own."""
        import bpy
        from mathutils import Vector

        fixture_empty.name = "Bench"
        there = _empty("Workpiece", arm6)
        there.location += Vector((0.0, 0.1, 0.0))
        library = tmp_path / "shelf.blend"
        bpy.data.libraries.write(str(library), {there})
        bpy.data.objects.remove(there)
        fixture_empty.name = "Workpiece"
        with bpy.data.libraries.load(str(library), link=True) as (_, linked):
            linked.objects = ["Workpiece"]
        linked = linked.objects[0]
        bpy.context.scene.collection.objects.link(linked)
        bpy.context.view_layer.update()
        assert linked.name == fixture_empty.name and linked.library is not None

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _relative(arm6, pick, fixture_empty)
        place = _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        place.reference = linked
        local = {w.name: w.marker.matrix_basis.copy() for w in (pick, place)}
        fixture_empty.location += Vector((0.02, 0.0, 0.0))
        bpy.context.view_layer.update()

        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        for frame, name, on in ((1, "pick", fixture_empty), (21, "place", linked)):
            bpy.context.scene.frame_set(frame)
            np.testing.assert_allclose(
                _tool_world(builder, arm6), _on(on, local[name]), atol=CLOSE, err_msg=name
            )

    def test_reading_the_objects_puts_the_scene_back_on_its_subframe(
        self, arm6, builder, references, fixture_empty, curves
    ):
        import bpy
        from mathutils import Vector

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _relative(arm6, pick, fixture_empty)
        _key_slide(curves, fixture_empty, 1, 21, Vector((0.02, 0.0, 0.0)))
        scene = bpy.context.scene
        scene.frame_set(12, subframe=0.5)

        references.Placement(arm6, [pick], range(1, 22), scene)
        assert (scene.frame_current, scene.frame_subframe) == (12, pytest.approx(0.5))
