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


def _gantry(rig):
    """Another armature, where ``rig`` is: a 'Beam' bone, and a 'Carriage' on it."""
    import bpy

    armature = bpy.data.objects.new("Gantry", bpy.data.armatures.new("Gantry"))
    bpy.context.scene.collection.objects.link(armature)
    armature.matrix_world = rig.matrix_world.copy()
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT")
    beam = armature.data.edit_bones.new("Beam")
    beam.head, beam.tail = (0.0, 0.0, 0.0), (0.0, 0.0, 0.2)
    carriage = armature.data.edit_bones.new("Carriage")
    carriage.head, carriage.tail = (0.0, 0.0, 0.2), (0.0, 0.0, 0.3)
    carriage.parent = beam
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.objects.active = rig
    bpy.context.view_layer.update()
    return armature


def _on_bone(obj, armature, bone):
    """Parent ``obj`` to ``armature``'s ``bone``, as a script does, with no inverse."""
    import bpy
    from mathutils import Matrix

    obj.parent, obj.parent_type, obj.parent_bone = armature, "BONE", bone
    obj.matrix_parent_inverse = Matrix.Identity(4)
    bpy.context.view_layer.update()


def _at(references, rig, waypoint, frame):
    """Where ``waypoint`` is on ``frame``, in rig space, as the job places it."""
    return np.array(references.Placement(rig, [waypoint], [frame]).pose(waypoint, frame))


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
        shape = bpy.data.objects.new("WGT-shape", None)
        shape.use_fake_user = True
        assert not references.may_be_reference(pick, shape), "in no scene"
        elsewhere = bpy.data.scenes.new("Elsewhere")
        try:
            away = bpy.data.objects.new("Away", None)
            elsewhere.collection.objects.link(away)
            assert not references.may_be_reference(pick, away), "only in a scene not played"
        finally:
            # The scene-wide clean-up between tests clears objects, not scenes.
            bpy.data.scenes.remove(elsewhere)

    def test_an_object_under_its_own_marker_is_refused_by_the_field(
        self, arm6, builder, references, fixture_empty
    ):
        """It would be a loop. Not offered; set from a script, past the filter, the
        field goes back and nothing moves."""
        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _relative(arm6, pick, fixture_empty)
        stored = list(pick.pose)
        below = _empty("Below")
        below.parent = pick.marker
        assert not references.may_be_reference(pick, below)

        pick.reference = below
        assert pick.reference == fixture_empty
        assert pick.marker.parent == fixture_empty
        assert list(pick.pose) == stored

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

    def test_a_line_from_the_base_to_a_moving_empty_blends_on_every_frame(
        self, arm6, builder, fixture_empty, curves
    ):
        """'pick' at the base, 'place' on an Empty that drops 30 mm over the move.
        On each frame the tool is that frame's fraction of the way from 'pick' to
        where the Empty has 'place' then: with the drop keyed linearly, a parabola
        in Z. Kept to the line from 'pick' to where 'place' ends up instead, the
        tool would be 7.5 mm off it half-way."""
        import bpy
        from mathutils import Vector

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        place = _teach(builder, arm6, 41, REACH_B, "LINEAR", "place")
        _relative(arm6, place, fixture_empty)
        _key_slide(curves, fixture_empty, 1, 41, Vector((0.0, 0.0, -0.03)))
        start = np.array(pick.marker.matrix_world)[:3, 3]
        local = np.array(place.marker.matrix_basis)

        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        bpy.context.scene.frame_set(41)
        last = _on(fixture_empty, local)[:3, 3]
        worst = bent = 0.0
        for frame in range(1, 42):
            bpy.context.scene.frame_set(frame)
            fraction = (frame - 1) / 40
            end = _on(fixture_empty, local)[:3, 3]
            expected = start + fraction * (end - start)
            worst = max(worst, float(np.linalg.norm(_tool_world(builder, arm6)[:3, 3] - expected)))
            bent = max(bent, float(np.linalg.norm(expected - (start + fraction * (last - start)))))
        assert bent == pytest.approx(0.0075, abs=1e-4), "precondition: the blend bends"
        assert worst < CLOSE, f"{worst * 1000:.3f} mm from the blended line"

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

    def test_a_muted_curve_leaves_its_channel_to_the_value_it_has(
        self, arm6, builder, wp_ops, references, fixture_empty, curves
    ):
        """Muted, the slide's x curve keys nothing: the Empty's own x is where it is
        on every frame, so moving that is an edit. Scrubbing still isn't."""
        import bpy
        from mathutils import Vector

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _relative(arm6, pick, fixture_empty)
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        _key_slide(curves, fixture_empty, 1, 21, Vector((0.02, 0.0, 0.0)))
        next(c for c in curves(fixture_empty) if c.array_index == 0).mute = True
        bpy.context.scene.frame_set(1)
        signed = wp_ops.job_signature(arm6)
        bpy.context.scene.frame_set(15)
        assert wp_ops.job_signature(arm6) == signed, "the frame shown is no edit"

        before = _at(references, arm6, pick, 11)
        fixture_empty.location.x += 0.05
        bpy.context.view_layer.update()
        moved = np.linalg.norm(_at(references, arm6, pick, 11)[:3, 3] - before[:3, 3])
        assert moved == pytest.approx(0.05, abs=1e-6), "the waypoint is elsewhere"
        assert wp_ops.job_signature(arm6) != signed

    @pytest.mark.parametrize("unplayed", ["muted", "another soloed", "NLA off"])
    def test_a_track_not_played_leaves_its_channels_to_the_values_they_have(
        self, arm6, builder, wp_ops, references, fixture_empty, curves, unplayed
    ):
        import bpy
        from mathutils import Vector

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _relative(arm6, pick, fixture_empty)
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        _key_slide(curves, fixture_empty, 1, 21, Vector((0.02, 0.0, 0.0)))
        anim = fixture_empty.animation_data
        track = anim.nla_tracks.new()
        track.strips.new("slide", 1, anim.action)
        anim.action = None
        if unplayed == "muted":
            track.mute = True
        elif unplayed == "another soloed":
            anim.nla_tracks.new().is_solo = True
        else:
            anim.use_nla = False
        bpy.context.scene.frame_set(1)
        signed = wp_ops.job_signature(arm6)

        before = _at(references, arm6, pick, 11)
        fixture_empty.location.x += 0.05
        bpy.context.view_layer.update()
        moved = np.linalg.norm(_at(references, arm6, pick, 11)[:3, 3] - before[:3, 3])
        assert moved == pytest.approx(0.05, abs=1e-6), "the waypoint is elsewhere"
        assert wp_ops.job_signature(arm6) != signed

    def test_an_object_on_another_armatures_bone_follows_the_bones_keys(
        self, arm6, builder, wp_ops, references, fixture_empty, curves
    ):
        """A workpiece on a gantry's carriage: re-keying the carriage, or turning
        the beam it hangs from, has the waypoints elsewhere. Scrubbing doesn't."""
        import bpy
        from mathutils import Quaternion

        gantry = _gantry(arm6)
        _on_bone(fixture_empty, gantry, "Carriage")
        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _relative(arm6, pick, fixture_empty)
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        carriage = gantry.pose.bones["Carriage"]
        carriage.keyframe_insert("location", frame=1)
        carriage.location.x = 0.03
        carriage.keyframe_insert("location", frame=21)
        bpy.context.scene.frame_set(1)
        signed = wp_ops.job_signature(arm6)
        bpy.context.scene.frame_set(15)
        assert wp_ops.job_signature(arm6) == signed, "the frame shown is no edit"

        before = _at(references, arm6, pick, 21)
        path = carriage.path_from_id("location")
        curve = next(c for c in curves(gantry) if c.data_path == path and c.array_index == 0)
        curve.keyframe_points[-1].co[1] += 0.06
        moved = np.linalg.norm(_at(references, arm6, pick, 21)[:3, 3] - before[:3, 3])
        assert moved == pytest.approx(0.06, abs=1e-6), "the waypoint is elsewhere"
        assert wp_ops.job_signature(arm6) != signed, "re-keyed, it goes elsewhere"

        signed = wp_ops.job_signature(arm6)
        gantry.pose.bones["Beam"].rotation_quaternion = Quaternion((0.0, 0.0, 1.0), 0.1)
        bpy.context.view_layer.update()
        assert wp_ops.job_signature(arm6) != signed, "the beam turned, it is elsewhere"

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

    def test_a_strip_with_animated_influence_is_read_by_its_curve(
        self, arm6, builder, wp_ops, fixture_empty, curves
    ):
        """Its influence as it stands changes as the timeline moves; its curve is
        what changes when it is edited."""
        import bpy
        from mathutils import Vector

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _relative(arm6, pick, fixture_empty)
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        _key_slide(curves, fixture_empty, 1, 21, Vector((0.02, 0.0, 0.0)))
        anim = fixture_empty.animation_data
        strip = anim.nla_tracks.new().strips.new("slide", 1, anim.action)
        anim.action = None
        strip.use_animated_influence = True
        for frame, influence in ((1, 1.0), (21, 0.5)):
            strip.influence = influence
            strip.keyframe_insert("influence", frame=frame)
        bpy.context.scene.frame_set(1)
        signed = wp_ops.job_signature(arm6)
        bpy.context.scene.frame_set(15)
        assert wp_ops.job_signature(arm6) == signed, "the frame shown is no edit"
        strip.fcurves[0].keyframe_points[-1].co[1] = 0.8
        assert wp_ops.job_signature(arm6) != signed, "eased in less, it is elsewhere"

    def test_a_baked_slide_is_read_by_its_samples(
        self, arm6, builder, wp_ops, fixture_empty, curves
    ):
        """Baked to samples, a curve has no keys: its samples are where it goes."""
        from mathutils import Vector

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _relative(arm6, pick, fixture_empty)
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        _key_slide(curves, fixture_empty, 1, 21, Vector((0.02, 0.0, 0.0)))
        curve = curves(fixture_empty)[0]
        curve.convert_to_samples(1, 21)
        assert not len(curve.keyframe_points) and len(curve.sampled_points)
        signed = wp_ops.job_signature(arm6)
        curve.sampled_points[10].co[1] += 0.03
        assert wp_ops.job_signature(arm6) != signed

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

    def test_an_axis_added_with_a_carried_marker_moves_its_records_at_the_base(
        self, arm6, builder, wp_ops, external_axes
    ):
        """One marker on a gripper, one on a wrist bone: refused until parented
        elsewhere, but their stored poses move with the robot all the same, and
        stay at the base, where they were taught. The markers go with their parents."""
        import bpy

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        place = _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        before = {w.name: np.array(w.pose).reshape(4, 4) for w in (pick, place)}
        gripper = _empty("Gripper")
        gripper.parent = arm6
        pick.marker.parent = gripper
        place.marker.parent_type = "BONE"
        place.marker.parent_bone = builder.joint_bones(arm6)[-1].name
        bpy.context.view_layer.update()
        held = {w.name: np.array(w.marker.matrix_basis) for w in (pick, place)}
        shift = np.eye(4)
        shift[0, 3] = 0.1

        external_axes._move_waypoints(arm6, shift)
        for waypoint in (pick, place):
            np.testing.assert_allclose(
                np.array(waypoint.pose).reshape(4, 4), shift @ before[waypoint.name], atol=1e-6
            )
            assert not any(waypoint.taught), waypoint.name
            np.testing.assert_allclose(
                np.array(waypoint.marker.matrix_basis), held[waypoint.name], atol=1e-9
            )

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

    def test_clearing_its_parent_puts_it_at_the_base_where_the_marker_is(
        self, arm6, builder, wp_ops, references, fixture_empty
    ):
        """Blender's Clear Parent, Keep Transform, on a marker relative to an Empty
        that has moved since: as clearing Relative To, it stays where it is, at
        the base, and the next generation walks the joints there."""
        import bpy
        from mathutils import Vector

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        taught = np.array(pick.pose).reshape(4, 4)
        _relative(arm6, pick, fixture_empty)
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        fixture_empty.location += Vector((0.02, 0.0, 0.0))
        bpy.context.view_layer.update()
        here = np.array(pick.marker.matrix_world)
        bpy.ops.object.select_all(action="DESELECT")
        pick.marker.select_set(True)
        bpy.context.view_layer.objects.active = pick.marker
        assert "FINISHED" in bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
        bpy.context.view_layer.objects.active = arm6

        wp_ops.job_plan(arm6)
        bpy.context.view_layer.update()
        assert pick.marker.parent == arm6 and pick.reference is None
        np.testing.assert_allclose(np.array(pick.marker.matrix_world), here, atol=1e-6)
        np.testing.assert_allclose(np.array(pick.pose).reshape(4, 4), taught, atol=1e-6)
        assert not any(pick.taught)
        assert wp_ops.marker_has_moved(pick)

    def test_clearing_a_base_markers_parent_keeps_it_read(self, arm6, builder, wp_ops):
        import bpy

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        here = np.array(pick.marker.matrix_world)
        bpy.ops.object.select_all(action="DESELECT")
        pick.marker.select_set(True)
        bpy.context.view_layer.objects.active = pick.marker
        assert "FINISHED" in bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
        bpy.context.view_layer.objects.active = arm6

        wp_ops.job_plan(arm6)
        bpy.context.view_layer.update()
        assert pick.marker.parent == arm6
        np.testing.assert_allclose(np.array(pick.marker.matrix_world), here, atol=1e-6)
        assert not wp_ops.marker_has_moved(pick)

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
        with pytest.raises(RuntimeError, match=carried):
            bpy.ops.kinema.goto_waypoint(index=0)
        assert len(arm6.kinema_waypoints) == 2

    def test_a_marker_parented_to_one_of_the_robots_bones_is_refused(self, arm6, builder):
        """Its parent is the rig, but it moves with the bone, not the base."""
        import bpy

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        wrist = builder.joint_bones(arm6)[-1].name
        pick.marker.parent_type = "BONE"
        pick.marker.parent_bone = wrist

        with pytest.raises(RuntimeError, match=f"parented to the robot's bone '{wrist}'"):
            bpy.ops.kinema.generate_motion()

    def test_a_marker_on_another_armatures_bone_is_refused(
        self, addon, arm6, builder, references
    ):
        """Its numbers are in the bone's space, a metre or so from where its
        object's space would put it: every operator says so, and so does the panel."""
        import bpy

        panel = importlib.import_module(f"{addon.__name__}.ui.panel")

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        gantry = _gantry(arm6)
        _on_bone(pick.marker, gantry, "Carriage")
        arm6.kinema_active_waypoint = 0
        shown = np.array(arm6.matrix_world.inverted() @ pick.marker.matrix_world)
        assert np.linalg.norm(_at(references, arm6, pick, 1)[:3, 3] - shown[:3, 3]) > 0.1

        refused = "'pick': its marker is parented to the bone 'Carriage' of 'Gantry'"
        with pytest.raises(RuntimeError, match=refused):
            bpy.ops.kinema.generate_motion()
        with pytest.raises(RuntimeError, match=refused):
            bpy.ops.kinema.add_waypoint(name="next")
        with pytest.raises(RuntimeError, match=refused):
            bpy.ops.kinema.update_waypoint(index=0)
        with pytest.raises(RuntimeError, match=refused):
            bpy.ops.kinema.goto_waypoint(index=0)
        assert len(arm6.kinema_waypoints) == 2
        assert "a part of 'Gantry'" in panel.reference_note(arm6, pick)

    def test_a_marker_on_a_meshs_vertices_is_refused(self, arm6, builder):
        import bpy

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        mesh = bpy.data.meshes.new("Plate")
        mesh.from_pydata([(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0)], [], [(0, 1, 2)])
        plate = bpy.data.objects.new("Plate", mesh)
        bpy.context.scene.collection.objects.link(plate)
        pick.marker.parent = plate
        pick.marker.parent_type = "VERTEX"
        pick.marker.parent_vertices[0] = 1
        bpy.context.view_layer.update()

        with pytest.raises(RuntimeError, match="parented to vertices of 'Plate'"):
            bpy.ops.kinema.generate_motion()

    def test_relative_to_takes_a_marker_off_a_bone_where_it_is(
        self, arm6, builder, wp_ops, references, fixture_empty
    ):
        """Taught at the base, parented by hand to a gantry's bone, then given an
        object in the field: it stays where the marker is, on the object itself,
        and its joints stay where they were taught."""
        import bpy

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        taught = np.array(pick.pose).reshape(4, 4)
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        gantry = _gantry(arm6)
        _on_bone(pick.marker, gantry, "Carriage")
        here = np.array(pick.marker.matrix_world)

        pick.reference = fixture_empty
        bpy.context.view_layer.update()
        assert pick.marker.parent == fixture_empty and pick.marker.parent_type == "OBJECT"
        np.testing.assert_allclose(np.array(pick.marker.matrix_world), here, atol=1e-6)
        shown = np.array(arm6.matrix_world.inverted()) @ here
        np.testing.assert_allclose(_at(references, arm6, pick, 1), shown, atol=1e-6)
        np.testing.assert_allclose(np.array(references.taught_pose(pick)), taught, atol=1e-6)
        wp_ops.job_plan(arm6)

    def test_relative_to_the_armature_its_bone_belongs_to_takes_it_off_the_bone(
        self, arm6, builder, wp_ops, references
    ):
        """Parented by hand to a gantry's bone, then given the gantry itself in the
        field: parented to the gantry as an object, where the marker is, and no
        longer refused."""
        import bpy

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        gantry = _gantry(arm6)
        _on_bone(pick.marker, gantry, "Carriage")
        here = np.array(pick.marker.matrix_world)

        pick.reference = gantry
        bpy.context.view_layer.update()
        assert pick.marker.parent == gantry and pick.marker.parent_type == "OBJECT"
        np.testing.assert_allclose(np.array(pick.marker.matrix_world), here, atol=1e-6)
        assert references.on_a_part(arm6, pick) is None
        wp_ops.job_plan(arm6)

    def test_a_job_saved_before_motions_is_refused_a_carried_marker_too(
        self, arm6, builder, wp_ops, monkeypatch
    ):
        """One linked from a library isn't converted, so it is planned as saved: the
        marker is refused all the same."""
        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        gripper = _empty("Gripper")
        gripper.parent = arm6
        pick.marker.parent = gripper
        monkeypatch.setattr(wp_ops, "convert_saved_before_motions", lambda rig: False)
        arm6.kinema_job_version = 0

        with pytest.raises(wp_ops.WaypointError, match="which the robot carries"):
            wp_ops.job_plan(arm6)


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

    def test_go_to_one_whose_object_is_deleted_goes_where_it_was_taught(
        self, arm6, builder, fixture_empty
    ):
        """Before anything has generated since: Go To catches it up itself."""
        import bpy
        from mathutils import Vector

        fixture_empty.location = Vector((0.05, 0.02, 0.0))
        bpy.context.view_layer.update()
        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        taught = np.array(pick.marker.matrix_world)
        _relative(arm6, pick, fixture_empty)
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        bpy.data.objects.remove(fixture_empty)
        arm6.kinema_ik_enabled = True

        assert "FINISHED" in bpy.ops.kinema.goto_waypoint(index=0)
        assert pick.marker.parent == arm6
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


class TestFingerprintEdges:
    def test_a_curve_in_a_muted_group_still_plays(
        self, arm6, builder, wp_ops, fixture_empty, curves
    ):
        """Blender 5.2 evaluates a curve whose group is muted: the slide goes on, so
        scrubbing it is no edit."""
        import bpy
        from mathutils import Vector

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _relative(arm6, pick, fixture_empty)
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        start = fixture_empty.location.x
        _key_slide(curves, fixture_empty, 1, 21, Vector((0.02, 0.0, 0.0)))
        bag = fixture_empty.animation_data.action.layers[0].strips[0].channelbags[0]
        for group in bag.groups:
            group.mute = True
        bpy.context.scene.frame_set(1)
        signed = wp_ops.job_signature(arm6)
        bpy.context.scene.frame_set(15)
        assert fixture_empty.location.x != pytest.approx(start), "precondition: it plays"
        assert wp_ops.job_signature(arm6) == signed, "the frame shown is no edit"

    def test_scrubbing_a_turning_object_is_no_edit(self, arm6, builder, wp_ops, fixture_empty):
        """A turntable keyed on Euler rotation. Blender keeps an object's quaternion
        and axis-angle as values of their own, not views of the Euler, so they
        stand still as it turns: none of the fingerprint moves with the frame."""
        import bpy

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _relative(arm6, pick, fixture_empty)
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        fixture_empty.keyframe_insert("rotation_euler", frame=1)
        fixture_empty.rotation_euler.z += 1.5
        fixture_empty.keyframe_insert("rotation_euler", frame=21)
        bpy.context.scene.frame_set(1)
        signed = wp_ops.job_signature(arm6)
        quaternion = tuple(fixture_empty.rotation_quaternion)
        turned = fixture_empty.rotation_euler.z
        bpy.context.scene.frame_set(11)
        assert fixture_empty.rotation_euler.z != pytest.approx(turned), "precondition: it turns"
        assert tuple(fixture_empty.rotation_quaternion) == quaternion
        assert wp_ops.job_signature(arm6) == signed, "the frame shown is no edit"

    def test_moving_an_envelope_point_outdates_the_job(
        self, arm6, builder, wp_ops, fixture_empty, curves
    ):
        """An Envelope modifier's control points reshape the slide; they're kept in
        a collection of their own."""
        from mathutils import Vector

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _relative(arm6, pick, fixture_empty)
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        _key_slide(curves, fixture_empty, 1, 21, Vector((0.02, 0.0, 0.0)))
        x = next(c for c in curves(fixture_empty) if c.array_index == 0)
        envelope = x.modifiers.new("ENVELOPE")
        point = envelope.control_points.add(11)
        point.min, point.max = -1.0, 1.0
        signed = wp_ops.job_signature(arm6)
        point.max = 2.0
        assert wp_ops.job_signature(arm6) != signed

    def test_a_soloed_track_plays_though_muted(
        self, arm6, builder, wp_ops, fixture_empty, curves
    ):
        """Blender plays a soloed track whether muted or not: scrubbing it is no edit."""
        import bpy
        from mathutils import Vector

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _relative(arm6, pick, fixture_empty)
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        start = fixture_empty.location.x
        _key_slide(curves, fixture_empty, 1, 21, Vector((0.02, 0.0, 0.0)))
        anim = fixture_empty.animation_data
        track = anim.nla_tracks.new()
        track.strips.new("slide", 1, anim.action)
        anim.action = None
        track.mute = True
        track.is_solo = True
        bpy.context.scene.frame_set(1)
        signed = wp_ops.job_signature(arm6)
        bpy.context.scene.frame_set(15)
        assert fixture_empty.location.x != pytest.approx(start), "precondition: it plays"
        assert wp_ops.job_signature(arm6) == signed, "the frame shown is no edit"

    def test_sliding_along_the_bone_it_hangs_from_outdates_the_job(
        self, arm6, builder, wp_ops, references, fixture_empty
    ):
        """From the carriage's head to its tail, 100 mm on."""
        gantry = _gantry(arm6)
        _on_bone(fixture_empty, gantry, "Carriage")
        if not hasattr(fixture_empty, "parent_bone_head_tail_factor"):
            pytest.skip("this Blender hangs a child from the bone's tail only")
        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _relative(arm6, pick, fixture_empty)
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        signed = wp_ops.job_signature(arm6)
        before = _at(references, arm6, pick, 1)
        fixture_empty.parent_bone_head_tail_factor = (
            0.0 if fixture_empty.parent_bone_head_tail_factor else 1.0
        )
        moved = np.linalg.norm(_at(references, arm6, pick, 1)[:3, 3] - before[:3, 3])
        assert moved == pytest.approx(0.1, abs=1e-6), "precondition: it is elsewhere"
        assert wp_ops.job_signature(arm6) != signed


class TestParentingEdges:
    @staticmethod
    def _on_the_wrist(builder, rig, waypoint):
        waypoint.marker.parent = rig
        waypoint.marker.parent_type = "BONE"
        waypoint.marker.parent_bone = builder.joint_bones(rig)[-1].name

    def test_clearing_relative_to_takes_a_marker_off_the_robots_bone(
        self, arm6, builder, wp_ops, fixture_empty
    ):
        """On an Empty, then parented by hand to the robot's wrist: clearing the
        field puts the marker on the robot as an object, where it is."""
        import bpy

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _relative(arm6, pick, fixture_empty)
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        self._on_the_wrist(builder, arm6, pick)
        bpy.context.view_layer.update()
        here = np.array(pick.marker.matrix_world)

        pick.reference = None
        bpy.context.view_layer.update()
        assert pick.marker.parent == arm6 and pick.marker.parent_type == "OBJECT"
        np.testing.assert_allclose(np.array(pick.marker.matrix_world), here, atol=1e-6)
        assert wp_ops.refusal(arm6, pick) is None

    def test_a_refused_assignment_leaves_a_hand_parented_marker_to_be_caught_up(
        self, arm6, builder, wp_ops, references, fixture_empty
    ):
        """Taught at the base, parented by hand to the Empty, then given an object
        under its own marker from a script: refused, and the next operator still
        keeps where its joints were taught."""
        import bpy

        fixture_empty.location.x += 0.05
        bpy.context.view_layer.update()
        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        taught = np.array(pick.pose).reshape(4, 4)
        _parent_by_hand(pick.marker, fixture_empty, arm6)
        below = _empty("Below")
        below.parent = pick.marker

        pick.reference = below
        assert pick.reference is None
        wp_ops.job_plan(arm6)
        assert pick.reference == fixture_empty
        np.testing.assert_allclose(np.array(references.taught_pose(pick)), taught, atol=1e-6)

    def test_record_after_clear_parent_goes_on_at_the_base(
        self, arm6, builder, fixture_empty
    ):
        import bpy

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _relative(arm6, pick, fixture_empty)
        bpy.ops.object.select_all(action="DESELECT")
        pick.marker.select_set(True)
        bpy.context.view_layer.objects.active = pick.marker
        assert "FINISHED" in bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
        bpy.context.view_layer.objects.active = arm6
        arm6.kinema_active_waypoint = 0

        place = _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        assert place.reference is None
        assert place.marker.parent == arm6

    def test_an_object_taken_out_of_the_robots_scene_is_refused(
        self, arm6, builder, fixture_empty
    ):
        """Relative to an Empty that has since moved to another scene, which the job
        doesn't play: refused, naming it."""
        import bpy

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _relative(arm6, pick, fixture_empty)
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        elsewhere = bpy.data.scenes.new("Elsewhere")
        try:
            elsewhere.collection.objects.link(fixture_empty)
            bpy.context.scene.collection.objects.unlink(fixture_empty)
            with pytest.raises(RuntimeError, match="'Workpiece', which isn't in the scene"):
                bpy.ops.kinema.generate_motion()
        finally:
            bpy.data.scenes.remove(elsewhere)

    def test_an_object_only_in_another_of_the_robots_scenes_is_refused_here(
        self, arm6, builder, fixture_empty
    ):
        """The robot is in two scenes and the Empty only in the other one: the job is
        played in this one, where the Empty isn't."""
        import bpy

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _relative(arm6, pick, fixture_empty)
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        other = bpy.data.scenes.new("Other")
        try:
            other.collection.objects.link(arm6)
            other.collection.objects.link(fixture_empty)
            bpy.context.scene.collection.objects.unlink(fixture_empty)
            with pytest.raises(RuntimeError, match="isn't in the scene 'Scene'"):
                bpy.ops.kinema.generate_motion()
        finally:
            bpy.data.scenes.remove(other)

    def test_a_cleared_marker_isnt_held_back_by_the_object_it_left(
        self, arm6, builder, fixture_empty
    ):
        """Its parent cleared by hand, then its old Empty moved to another scene: it
        goes back to the base, and the job generates."""
        import bpy

        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _teach(builder, arm6, 21, REACH_B, "JOINT", "place")
        _relative(arm6, pick, fixture_empty)
        bpy.ops.object.select_all(action="DESELECT")
        pick.marker.select_set(True)
        bpy.context.view_layer.objects.active = pick.marker
        assert "FINISHED" in bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
        bpy.context.view_layer.objects.active = arm6
        elsewhere = bpy.data.scenes.new("Elsewhere")
        try:
            elsewhere.collection.objects.link(fixture_empty)
            bpy.context.scene.collection.objects.unlink(fixture_empty)
            assert "FINISHED" in bpy.ops.kinema.generate_motion()
            assert pick.marker.parent == arm6 and pick.reference is None
        finally:
            bpy.data.scenes.remove(elsewhere)

    def test_the_panel_says_a_marker_on_the_robot_is_refused(
        self, addon, arm6, builder, fixture_empty
    ):
        import bpy

        panel = importlib.import_module(f"{addon.__name__}.ui.panel")
        pick = _teach(builder, arm6, 1, REACH_A, "JOINT", "pick")
        _relative(arm6, pick, fixture_empty)
        self._on_the_wrist(builder, arm6, pick)
        bpy.context.view_layer.update()
        assert "robot's own bone" in panel.reference_note(arm6, pick)
