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
from .test_waypoints import REACH_A, REACH_B, _set_q, _teach

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
        worst = 0.0
        for frame in range(1, 42):
            bpy.context.scene.frame_set(frame)
            start, end = _on(fixture_empty, a)[:3, 3], _on(fixture_empty, b)[:3, 3]
            tool = _tool_world(builder, arm6)[:3, 3]
            along = np.clip((tool - start) @ (end - start) / ((end - start) @ (end - start)), 0, 1)
            worst = max(worst, float(np.linalg.norm(tool - (start + along * (end - start)))))
        assert worst < CLOSE, f"{worst * 1000:.3f} mm off the moving line"
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
