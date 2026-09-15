"""Viewport handles: where each gizmo sits, and what the IK control's handles write.

Needs a real ``bpy``. Gizmos only draw in a window, so nothing here drags one. The
claims are about what the gizmo group computes and writes, which is where its logic
lives:

* a joint's dial turns about that joint's axis at its head, and does not turn with
  the joint or with a limit clamping it -- on a rig moved, turned and scaled too,
* a slide's arrow reaches the bone's head at the channel's value,
* the IK control's move handle sits on the control; its slides and turns go along
  and about the tool's axes, key only with auto-keying on, and live IK follows them,
* the swivel's dial turns about the shoulder-to-wrist line and rides the arm,
* handles show for a selected rig or an object parented to it, in Object or Pose
  mode, and for nothing else.

Every rig solves on NumPy: no claim here depends on PyRoki.
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
def handlers(addon):
    return importlib.import_module(f"{addon.__name__}.handlers")


@pytest.fixture
def gizmos(addon):
    return importlib.import_module(f"{addon.__name__}.ui.gizmos")


def _rig_from(fixture_dir, builder, name: str):
    import bpy

    assert "FINISHED" in bpy.ops.kinema.build_robot(
        filepath=str(fixture_dir / name), enforce_limits=True
    )
    rig = next(o for o in bpy.data.objects if builder.is_kinema_rig(o))
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    rig.kinema_solver_mode = "NUMPY"
    bpy.ops.kinema.add_ik()
    return rig


@pytest.fixture
def arm7(addon, fixture_dir, clean_scene, builder):
    return _rig_from(fixture_dir, builder, "arm7.urdf")


@pytest.fixture
def rail6(addon, fixture_dir, clean_scene, builder):
    return _rig_from(fixture_dir, builder, "rail6.urdf")


#: arm7's seven joints: bent everywhere, and inside every limit.
BENT = [0.3, -0.8, 0.5, 1.0, 0.2, 0.6, 0.4]
ELSEWHERE = [-0.4, 0.5, -0.3, 0.7, -0.5, 0.3, -0.2]


def _np(matrix) -> np.ndarray:
    return np.array([list(row) for row in matrix], dtype=float)


def _unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)[:3]
    return vector / np.linalg.norm(vector)


def _pose(builder, rig, q) -> None:
    import bpy

    for pose_bone, value in zip(builder.joint_bones(rig), q, strict=True):
        if pose_bone.bone.get(builder.PROP_JOINT_TYPE) == "prismatic":
            pose_bone.location[1] = value
        else:
            pose_bone.rotation_euler[1] = value
    bpy.context.view_layer.update()


def _still(rig) -> None:
    """Live IK off: these tests place things by hand and read them back."""
    import bpy

    rig.kinema_ik_enabled = False
    bpy.context.view_layer.update()


def _move_rig_off_the_origin(rig) -> None:
    import bpy
    from mathutils import Matrix, Vector

    rig.matrix_world = (
        Matrix.Translation((1.5, -0.7, 0.3))
        @ Matrix.Rotation(0.8, 4, Vector((0.3, 0.4, 0.87)).normalized())
        @ Matrix.Diagonal((2.0, 2.0, 2.0, 1.0))
    )
    bpy.context.view_layer.update()


def _world(rig, pose_bone) -> np.ndarray:
    import bpy

    bpy.context.view_layer.update()
    return _np(rig.matrix_world @ pose_bone.matrix)


class TestJointDials:
    @pytest.mark.parametrize("off_origin", [False, True], ids=["at-origin", "moved-turned-scaled"])
    def test_each_dial_turns_about_its_joint_axis_at_its_head(
        self, arm7, builder, gizmos, off_origin
    ):
        rig = arm7
        _still(rig)
        _pose(builder, rig, BENT)
        if off_origin:
            _move_rig_off_the_origin(rig)

        for pose_bone in builder.joint_bones(rig):
            frame = _np(gizmos.axis_frame(rig, pose_bone))
            world = _world(rig, pose_bone)
            # A joint bone's Y is its axis, and turning the joint leaves it put.
            np.testing.assert_allclose(
                _unit(frame[:3, 2]), _unit(world[:3, 1]), atol=1e-5, err_msg=pose_bone.name
            )
            np.testing.assert_allclose(
                frame[:3, 3], world[:3, 3], atol=1e-5, err_msg=pose_bone.name
            )

    def test_a_dial_does_not_turn_with_its_own_joint(self, arm7, builder, gizmos):
        """Drawn in the joint's own frame, the dial would spin under the mouse."""
        rig = arm7
        _still(rig)
        _pose(builder, rig, BENT)
        pose_bone = builder.joint_bones(rig)[3]
        before = _np(gizmos.axis_frame(rig, pose_bone))

        pose_bone.rotation_euler[1] += 0.7
        _world(rig, pose_bone)

        np.testing.assert_allclose(_np(gizmos.axis_frame(rig, pose_bone)), before, atol=1e-6)

    def test_past_a_limit_the_dial_stays_where_the_channel_counts_from(
        self, arm7, builder, gizmos
    ):
        """The channel can run past a limit that the pose stops at.

        The dial is bound to the channel, so its zero has to stay the channel's zero.
        A frame recovered from what the pose shows would turn by however far the
        limit clamped it.
        """
        rig = arm7
        _still(rig)
        _pose(builder, rig, BENT)
        # A limit with room past it below pi: a Y angle beyond pi reads back as its
        # wrapped equivalent, which a +-pi limit then lets through unclamped.
        pose_bone = next(
            pb
            for pb in builder.joint_bones(rig)
            if builder.PROP_UPPER in pb.bone and float(pb.bone[builder.PROP_UPPER]) < 2.5
        )
        upper = float(pose_bone.bone[builder.PROP_UPPER])

        pose_bone.rotation_euler[1] = upper
        at_limit = _world(rig, pose_bone)
        frame_at_limit = _np(gizmos.axis_frame(rig, pose_bone))

        pose_bone.rotation_euler[1] = upper + 0.5
        past_limit = _world(rig, pose_bone)
        # The scenario: the limit really is holding the bone back.
        np.testing.assert_allclose(past_limit, at_limit, atol=1e-5)

        np.testing.assert_allclose(
            _np(gizmos.axis_frame(rig, pose_bone)), frame_at_limit, atol=1e-6
        )

    @pytest.mark.parametrize("off_origin", [False, True], ids=["at-origin", "moved-turned-scaled"])
    def test_a_slide_arrow_reaches_the_bone_at_the_channel_value(
        self, rail6, builder, gizmos, off_origin
    ):
        """The arrow is drawn ``value`` along its Z from the frame's origin.

        On a scaled rig the frame carries the scale, so the channel's own units
        still land on the bone.
        """
        from mathutils import Vector

        rig = rail6
        _still(rig)
        if off_origin:
            _move_rig_off_the_origin(rig)
        rail = rig.pose.bones["rail"]
        rail.location[1] = 0.3

        head = _world(rig, rail)[:3, 3]
        tip = gizmos.axis_frame(rig, rail) @ Vector((0.0, 0.0, 0.3))
        np.testing.assert_allclose(np.array(tip), head, atol=1e-5)


class TestIkHandles:
    def test_the_move_handle_sits_on_the_ik_control(self, arm7, builder, gizmos):
        """Bound to the control's location, so drawn in the frame location counts in.

        The Root is turned first. The control hangs from it, and a frame that left
        the parent out would agree with an unposed Root by coincidence.
        """
        from mathutils import Quaternion, Vector

        rig = arm7
        _still(rig)
        rig.pose.bones[builder.ROOT_BONE].rotation_quaternion = Quaternion((0, 0, 1), 0.6)
        ik = rig.pose.bones[rig[builder.PROP_IK_BONE]]
        ik.location = (0.05, -0.03, 0.08)

        head = _world(rig, ik)[:3, 3]
        drawn = gizmos.rest_frame(rig, ik) @ Vector(ik.location)
        np.testing.assert_allclose(np.array(drawn), head, atol=1e-5)

    def test_every_axis_handle_works_on_a_tool_axis(self, arm7, builder, gizmos):
        """The IK handles point where the TCP panel's tool frame does."""
        rig = arm7
        _still(rig)
        ik = rig.pose.bones[rig[builder.PROP_IK_BONE]]
        tool = _world(rig, ik) @ np.array(builder.BONE_TO_TOOL)
        for handle in gizmos.handles(rig):
            if handle.role not in (gizmos.ROLE_IK_SLIDE, gizmos.ROLE_IK_TURN):
                continue
            drawn = _np(gizmos.handle_matrix(rig, handle))
            np.testing.assert_allclose(
                _unit(drawn[:3, 2]), _unit(tool[:3, handle.axis]), atol=1e-6
            )
            np.testing.assert_allclose(drawn[:3, 3], tool[:3, 3], atol=1e-6)

    def test_a_slide_moves_the_control_along_the_tool_axis_only(
        self, arm7, builder, gizmos
    ):
        import bpy

        rig = arm7
        _still(rig)
        ik = rig.pose.bones[rig[builder.PROP_IK_BONE]]
        start = _np(ik.matrix)
        # Tool Z, the approach direction: the bone's Y, not its Z.
        handle = gizmos.Handle(gizmos.ARROW, ik.name, gizmos.ROLE_IK_SLIDE, 2)

        gizmos.apply_ik_handle(rig, handle, start, 0.05, bpy.context.scene)
        bpy.context.view_layer.update()

        after = _np(ik.matrix)
        np.testing.assert_allclose(after[:3, :3], start[:3, :3], atol=1e-5)
        np.testing.assert_allclose(
            after[:3, 3] - start[:3, 3], 0.05 * _unit(start[:3, 1]), atol=1e-5
        )

    def test_a_turn_keeps_the_tool_point_and_turns_about_the_tool_axis(
        self, arm7, builder, gizmos
    ):
        import bpy

        rig = arm7
        _still(rig)
        ik = rig.pose.bones[rig[builder.PROP_IK_BONE]]
        start = _np(ik.matrix)
        # Tool X is the bone's Z.
        handle = gizmos.Handle(gizmos.DIAL, ik.name, gizmos.ROLE_IK_TURN, 0)

        gizmos.apply_ik_handle(rig, handle, start, 0.4, bpy.context.scene)
        bpy.context.view_layer.update()

        after = _np(ik.matrix)
        axis = _unit(start[:3, 2])
        x, y, z = axis
        k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
        turn = np.eye(3) + math.sin(0.4) * k + (1.0 - math.cos(0.4)) * (k @ k)
        np.testing.assert_allclose(after[:3, 3], start[:3, 3], atol=1e-5)
        np.testing.assert_allclose(after[:3, :3], turn @ start[:3, :3], atol=1e-5)

    def test_every_step_is_measured_from_where_the_drag_began(
        self, arm7, builder, gizmos
    ):
        """Blender reports a drag as a running total, not as increments.

        The scene updates between mouse events, as it does here: without that,
        a step measured from wherever the control now stands would read the
        pose from before the drag and pass by accident.
        """
        import bpy

        rig = arm7
        _still(rig)
        ik = rig.pose.bones[rig[builder.PROP_IK_BONE]]
        start = _np(ik.matrix)
        handle = gizmos.Handle(gizmos.ARROW, ik.name, gizmos.ROLE_IK_SLIDE, 1)

        for value in (0.01, 0.02, 0.03):
            gizmos.apply_ik_handle(rig, handle, start, value, bpy.context.scene)
            bpy.context.view_layer.update()

        moved = float(np.linalg.norm(_np(ik.matrix)[:3, 3] - start[:3, 3]))
        assert moved == pytest.approx(0.03, abs=1e-5)

    @pytest.mark.parametrize("auto_key", [True, False], ids=["auto-key-on", "auto-key-off"])
    def test_axis_handles_key_only_when_auto_keying_is_on(
        self, arm7, builder, gizmos, auto_key
    ):
        import bpy
        from bpy_extras import anim_utils

        rig = arm7
        _still(rig)
        ik = rig.pose.bones[rig[builder.PROP_IK_BONE]]
        scene = bpy.context.scene
        settings = scene.tool_settings
        was = settings.use_keyframe_insert_auto
        settings.use_keyframe_insert_auto = auto_key
        try:
            handle = gizmos.Handle(gizmos.DIAL, ik.name, gizmos.ROLE_IK_TURN, 2)
            gizmos.apply_ik_handle(rig, handle, _np(ik.matrix), 0.3, scene)
        finally:
            settings.use_keyframe_insert_auto = was

        animation = rig.animation_data
        paths = set()
        if animation is not None and animation.action is not None:
            bag = anim_utils.action_get_channelbag_for_slot(
                animation.action, animation.action_slot
            )
            paths = {fcurve.data_path for fcurve in bag.fcurves} if bag else set()
        keyed = {
            f'pose.bones["{ik.name}"].location',
            f'pose.bones["{ik.name}"].rotation_quaternion',
        }
        if auto_key:
            assert keyed <= paths
        else:
            assert not keyed & paths

    def test_live_ik_follows_a_handle_drag(self, arm7, builder, handlers, gizmos):
        import bpy

        rig = arm7
        _still(rig)
        _pose(builder, rig, BENT)
        bpy.ops.kinema.snap_ik()
        rig.kinema_ik_enabled = True
        handlers.solve_rig(rig, force=True)
        bpy.context.view_layer.update()

        ik = rig.pose.bones[rig[builder.PROP_IK_BONE]]
        handle = gizmos.Handle(gizmos.ARROW, ik.name, gizmos.ROLE_IK_SLIDE, 2)
        with handlers.suspended():
            gizmos.apply_ik_handle(rig, handle, _np(ik.matrix), 0.03, bpy.context.scene)
            bpy.context.view_layer.update()
        assert handlers.solve_rig(rig), "the handle's write never reached a solve"
        bpy.context.view_layer.update()

        tcp = rig.pose.bones[rig.get(builder.PROP_TCP_BONE) or builder.TCP_BONE]
        gap = float(np.linalg.norm(np.array(tcp.matrix.translation - ik.matrix.translation)))
        assert gap < 2e-3, f"the tool stayed {gap * 1000:.2f} mm from the moved target"


class TestSwivelDial:
    def test_it_turns_about_the_shoulder_to_wrist_line_and_rides_the_arm(
        self, arm7, builder, gizmos
    ):
        import bpy

        rig = arm7
        _still(rig)
        _pose(builder, rig, BENT)
        bpy.ops.kinema.snap_ik()
        assert "FINISHED" in bpy.ops.kinema.add_swivel()
        _still(rig)
        pose = rig.pose.bones
        ring = pose[rig[builder.PROP_SWIVEL_BONE]]

        for q in (BENT, ELSEWHERE):
            _pose(builder, rig, q)
            shoulder = _world(rig, pose[rig[builder.PROP_SWIVEL_SHOULDER]])[:3, 3]
            wrist = _world(rig, pose[rig[builder.PROP_SWIVEL_WRIST]])[:3, 3]
            frame = _np(gizmos.axis_frame(rig, ring))
            np.testing.assert_allclose(_unit(frame[:3, 2]), _unit(wrist - shoulder), atol=1e-5)
            np.testing.assert_allclose(frame[:3, 3], _world(rig, ring)[:3, 3], atol=1e-5)

    def test_it_is_a_handle_only_once_a_swivel_is_added(self, arm7, gizmos):
        import bpy

        def swivels():
            return [h for h in gizmos.handles(arm7) if h.role == gizmos.ROLE_SWIVEL]

        assert swivels() == []
        assert "FINISHED" in bpy.ops.kinema.add_swivel()
        assert len(swivels()) == 1


class TestWhichHandlesExist:
    def test_a_dial_per_rotary_joint_an_arrow_per_slide(self, rail6, builder, gizmos):
        joints = {
            h.bone: h.gizmo_type for h in gizmos.handles(rail6) if h.role == gizmos.ROLE_JOINT
        }
        assert list(joints) == [pb.name for pb in builder.joint_bones(rail6)]
        assert joints.pop("rail") == gizmos.ARROW
        assert set(joints.values()) == {gizmos.DIAL}

    def test_the_ik_control_gets_a_move_and_three_slides_and_turns(self, arm7, gizmos):
        ik = [h for h in gizmos.handles(arm7) if h.role.startswith("ik_")]
        assert [h.role for h in ik].count(gizmos.ROLE_IK_MOVE) == 1
        for role in (gizmos.ROLE_IK_SLIDE, gizmos.ROLE_IK_TURN):
            assert sorted(h.axis for h in ik if h.role == role) == [0, 1, 2]

    @pytest.mark.parametrize(
        ("shown", "joints", "ik"),
        [({"JOINTS"}, True, False), ({"IK"}, False, True), ({"JOINTS", "IK"}, True, True)],
    )
    def test_the_panel_toggles_choose_which(self, arm7, gizmos, shown, joints, ik):
        arm7.kinema_gizmos = shown
        roles = {h.role for h in gizmos.handles(arm7)}
        assert (gizmos.ROLE_JOINT in roles) is joints
        assert (gizmos.ROLE_IK_MOVE in roles) is ik

    def test_a_bone_the_viewport_hides_hides_its_handle(self, arm7, builder, gizmos):
        rig = arm7
        joint = builder.joint_bones(rig)[0]
        assert gizmos.bone_shown(joint)

        joint.bone.hide = True
        assert not gizmos.bone_shown(joint)
        joint.bone.hide = False

        for collection in joint.bone.collections:
            collection.is_visible = False
        assert not gizmos.bone_shown(joint)


class TestWhenHandlesShow:
    def test_a_selected_rig_shows_them_in_object_and_pose_mode(self, arm7, gizmos):
        import bpy

        assert gizmos.gizmo_rig(bpy.context) == arm7
        bpy.ops.object.mode_set(mode="POSE")
        try:
            assert gizmos.gizmo_rig(bpy.context) == arm7
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")

    def test_edit_mode_shows_none(self, arm7, gizmos):
        import bpy

        bpy.ops.object.mode_set(mode="EDIT")
        try:
            assert gizmos.gizmo_rig(bpy.context) is None
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")

    def test_an_object_parented_to_the_rig_shows_its_rig_s(self, arm7, gizmos):
        import bpy

        child = bpy.data.objects.new("link stand-in", None)
        bpy.context.scene.collection.objects.link(child)
        child.parent = arm7
        arm7.select_set(False)
        child.select_set(True)
        bpy.context.view_layer.objects.active = child

        assert gizmos.gizmo_rig(bpy.context) == arm7

    def test_an_unrelated_object_shows_none(self, arm7, gizmos):
        import bpy

        other = bpy.data.objects.new("unrelated", None)
        bpy.context.scene.collection.objects.link(other)
        arm7.select_set(False)
        other.select_set(True)
        bpy.context.view_layer.objects.active = other

        assert gizmos.gizmo_rig(bpy.context) is None

    def test_both_toggles_off_shows_none(self, arm7, gizmos):
        import bpy

        arm7.kinema_gizmos = set()
        assert gizmos.gizmo_rig(bpy.context) is None


def test_the_gizmo_group_registers_with_the_add_on(gizmos):
    import bpy

    def registered() -> bool:
        return bpy.types.GizmoGroup.bl_rna_get_subclass_py("KINEMA_GGT_rig") is not None

    assert registered()
    bpy.utils.unregister_class(gizmos.KINEMA_GGT_rig)
    try:
        assert not registered()
    finally:
        bpy.utils.register_class(gizmos.KINEMA_GGT_rig)
    assert registered()
