"""Rig builder tests. Require a real ``bpy``; run via ``dev.py test``.

These are the tests that matter most: they assert the properties that make
Kinema's armature different from every other URDF importer's.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from ..conftest import requires_bpy

pytestmark = requires_bpy


@pytest.fixture
def kin(addon):
    return importlib.import_module(f"{addon.__name__}.rig.kinematics")


@pytest.fixture
def builder(addon):
    return importlib.import_module(f"{addon.__name__}.rig.builder")


@pytest.fixture
def arm3_urdf(fixture_dir):
    yourdfpy = pytest.importorskip("yourdfpy")
    return yourdfpy.URDF.load(str(fixture_dir / "arm3.urdf"))


@pytest.fixture
def arm3_rig(kin, builder, arm3_urdf, clean_scene):
    model = kin.model_from_urdf(arm3_urdf)
    result = builder.build_rig(model, builder.RigBuildOptions())
    return model, result


def _np4(matrix) -> np.ndarray:
    return np.array([[matrix[r][c] for c in range(4)] for r in range(4)])


class TestArmatureShape:
    def test_exactly_one_armature(self, arm3_rig):
        """Phobos-style nested armatures are the thing being avoided."""
        import bpy

        armatures = [o for o in bpy.data.objects if o.type == "ARMATURE"]
        assert len(armatures) == 1

    def test_one_bone_per_actuated_joint_plus_root_and_tcp(self, arm3_rig):
        model, result = arm3_rig
        bones = result.armature_object.data.bones
        assert len(model.actuated_joints) == 3
        assert len(bones) == 3 + 2
        assert set(result.joint_bones) == {"joint1", "joint2", "joint3"}

    def test_fixed_joints_get_no_bones(self, arm3_rig):
        _, result = arm3_rig
        bones = {b.name for b in result.armature_object.data.bones}
        assert "base_to_mount" not in bones
        assert "l3_to_tool" not in bones

    def test_bones_are_never_connected(self, arm3_rig):
        """A connected bone's head is pinned to its parent's tail, which would
        drag every joint origin away from where the URDF puts it."""
        _, result = arm3_rig
        assert not any(b.use_connect for b in result.armature_object.data.bones)

    def test_bone_collections_exist(self, arm3_rig):
        _, result = arm3_rig
        names = {c.name for c in result.armature_object.data.collections_all}
        assert {"Kinema", "FK", "IK", "TCP", "Mechanism"} <= names


class TestBoneAxisAlignment:
    def test_bone_local_y_is_the_joint_axis(self, arm3_rig):
        """The core invariant: Y on the axis makes one channel = one joint."""
        from mathutils import Vector

        model, result = arm3_rig
        joint_frames = model.joint_frames()
        bones = result.armature_object.data.bones

        for joint in model.actuated_joints:
            bone = bones[result.joint_bones[joint.name]]
            bone_y = np.array(bone.matrix_local.to_3x3() @ Vector((0, 1, 0)))
            expected = joint_frames[joint.name][:3, :3] @ joint.axis
            assert np.allclose(bone_y, expected, atol=1e-6), joint.name

    def test_bone_head_sits_at_the_joint_origin(self, arm3_rig):
        model, result = arm3_rig
        frames = model.joint_frames()
        bones = result.armature_object.data.bones
        for joint in model.actuated_joints:
            head = np.array(bones[result.joint_bones[joint.name]].head_local)
            assert np.allclose(head, frames[joint.name][:3, 3], atol=1e-9), joint.name


class TestPosedForwardKinematics:
    def test_posed_rig_matches_urdf_fk(self, arm3_rig, arm3_urdf, builder):
        """Drive the bones, read the tool, compare against yourdfpy.

        This is the acceptance test for the whole rig design. If bone rest
        matrices compose the way URDF joint origins do, this holds to
        floating-point precision for any configuration.
        """
        import bpy

        model, result = arm3_rig
        armature = result.armature_object

        # The TCP bone's rest matrix is not the tool link's frame -- the bone's
        # Y is its own direction -- so a constant correction maps one to the other.
        rest_bone = _np4(armature.data.bones[builder.TCP_BONE].matrix_local)
        rest_link = model.link_frames()[result.tcp_link]
        correction = np.linalg.inv(rest_bone) @ rest_link

        rng = np.random.default_rng(0)
        for _ in range(6):
            cfg = {}
            for joint in model.actuated_joints:
                low, high = joint.lower, joint.upper
                value = float(rng.uniform(low * 0.8, high * 0.8))
                cfg[joint.name] = value
                pose_bone = armature.pose.bones[result.joint_bones[joint.name]]
                if joint.is_revolute:
                    pose_bone.rotation_euler[1] = value
                else:
                    pose_bone.location[1] = value
            bpy.context.view_layer.update()

            got = _np4(armature.pose.bones[builder.TCP_BONE].matrix) @ correction
            arm3_urdf.update_cfg(cfg)
            want = np.asarray(arm3_urdf.get_transform(result.tcp_link, model.root_link))
            assert np.allclose(got, want, atol=1e-5), f"FK mismatch at {cfg}"


class TestPoseBoneSetup:
    def test_revolute_exposes_only_rotation_y(self, arm3_rig):
        _, result = arm3_rig
        pose_bone = result.armature_object.pose.bones[result.joint_bones["joint1"]]
        assert tuple(pose_bone.lock_rotation) == (True, False, True)
        assert tuple(pose_bone.lock_location) == (True, True, True)
        assert pose_bone.rotation_mode == "YXZ"

    def test_prismatic_exposes_only_location_y(self, arm3_rig):
        _, result = arm3_rig
        pose_bone = result.armature_object.pose.bones[result.joint_bones["joint3"]]
        assert tuple(pose_bone.lock_location) == (True, False, True)
        assert tuple(pose_bone.lock_rotation) == (True, True, True)

    def test_limits_become_constraints(self, arm3_rig, builder):
        _, result = arm3_rig
        pose_bone = result.armature_object.pose.bones[result.joint_bones["joint1"]]
        constraint = pose_bone.constraints.get(builder.LIMIT_CONSTRAINT)
        assert constraint is not None
        assert constraint.type == "LIMIT_ROTATION"
        assert constraint.min_y == pytest.approx(-1.5)
        assert constraint.max_y == pytest.approx(1.5)
        assert constraint.owner_space == "LOCAL"

    def test_limits_can_be_turned_off(self, kin, builder, arm3_urdf, clean_scene):
        model = kin.model_from_urdf(arm3_urdf)
        result = builder.build_rig(model, builder.RigBuildOptions(enforce_limits=False))
        pose_bone = result.armature_object.pose.bones[result.joint_bones["joint1"]]
        assert builder.LIMIT_CONSTRAINT not in [c.name for c in pose_bone.constraints]

    def test_bones_carry_their_urdf_facts(self, arm3_rig, builder):
        """The rig is self-describing, so a saved .blend needs no URDF."""
        _, result = arm3_rig
        bone = result.armature_object.pose.bones[result.joint_bones["joint2"]].bone
        assert bone[builder.PROP_JOINT_NAME] == "joint2"
        assert bone[builder.PROP_JOINT_TYPE] == "revolute"
        assert np.allclose(list(bone[builder.PROP_AXIS]), [0, 1, 0])

    def test_every_joint_bone_has_a_custom_shape(self, arm3_rig):
        _, result = arm3_rig
        for bone_name in result.joint_bones.values():
            pose_bone = result.armature_object.pose.bones[bone_name]
            assert pose_bone.custom_shape is not None, bone_name


class TestVisuals:
    def test_link_meshes_are_parented_to_the_right_bone(self, arm3_rig):
        _, result = arm3_rig
        by_name = {o.name: o for o in result.mesh_objects}
        assert by_name, "no visuals imported"
        for obj in result.mesh_objects:
            assert obj.parent is result.armature_object
            assert obj.parent_type == "BONE"
            assert obj.parent_bone

    def test_meshes_are_rigid_not_skinned(self, arm3_rig):
        """Robot links are rigid bodies; an armature modifier would be wrong."""
        _, result = arm3_rig
        for obj in result.mesh_objects:
            assert not any(m.type == "ARMATURE" for m in obj.modifiers)
            assert not obj.vertex_groups

    def test_visual_origin_is_respected(self, arm3_rig):
        """l2's cylinder is offset 0.05 along x from its link frame."""
        model, result = arm3_rig
        obj = next(o for o in result.mesh_objects if o.name.startswith("l2"))
        expected = model.link_frames()["l2"][:3, 3] + np.array([0.05, 0, 0])
        assert np.allclose(np.array(obj.matrix_world.translation), expected, atol=1e-6)

    def test_can_skip_visuals(self, kin, builder, arm3_urdf, clean_scene):
        model = kin.model_from_urdf(arm3_urdf)
        result = builder.build_rig(model, builder.RigBuildOptions(import_visuals=False))
        assert result.mesh_objects == []


class TestResumableBuild:
    """build_rig_iter is what lets the import operator stay responsive.

    build_rig is a thin drain of it, so the two must not be able to disagree.
    """

    def test_yields_once_per_visual(self, kin, builder, arm3_urdf, clean_scene):
        model = kin.model_from_urdf(arm3_urdf)
        expected = sum(len(link.visuals) for link in model.links.values())
        steps = list(builder.build_rig_iter(model, builder.RigBuildOptions()))

        assert len(steps) == expected
        assert [done for done, _ in steps] == list(range(1, expected + 1))
        assert all(total == expected for _, total in steps)

    def test_draining_it_matches_build_rig(self, kin, builder, arm3_urdf, clean_scene):
        model = kin.model_from_urdf(arm3_urdf)
        steps = builder.build_rig_iter(model, builder.RigBuildOptions())
        while True:
            try:
                next(steps)
            except StopIteration as stop:
                chunked = stop.value
                break
        chunked_counts = (
            len(chunked.joint_bones),
            len(chunked.mesh_objects),
            chunked.tcp_link,
            sorted(bone.name for bone in chunked.armature_object.pose.bones),
        )
        builder.discard_rig(chunked)

        direct = builder.build_rig(kin.model_from_urdf(arm3_urdf),
                                   builder.RigBuildOptions())
        assert (
            len(direct.joint_bones),
            len(direct.mesh_objects),
            direct.tcp_link,
            sorted(bone.name for bone in direct.armature_object.pose.bones),
        ) == chunked_counts

    def test_caller_supplied_result_tracks_the_partial_rig(
        self, kin, builder, arm3_urdf, clean_scene
    ):
        """The operator needs a handle on what exists in order to cancel."""
        model = kin.model_from_urdf(arm3_urdf)
        result = builder.RigBuildResult()
        steps = builder.build_rig_iter(model, builder.RigBuildOptions(), result=result)

        next(steps)  # one visual in
        assert result.armature_object is not None
        assert len(result.mesh_objects) >= 1
        steps.close()

    def test_discard_removes_a_partial_rig(
        self, kin, builder, arm3_urdf, clean_scene
    ):
        import bpy

        model = kin.model_from_urdf(arm3_urdf)
        result = builder.RigBuildResult()
        steps = builder.build_rig_iter(model, builder.RigBuildOptions(), result=result)
        next(steps)
        steps.close()

        collection_name = result.collection.name
        builder.discard_rig(result)

        assert collection_name not in bpy.data.collections
        assert not [o for o in bpy.data.objects if o.get(builder.PROP_IS_RIG)]

    def test_no_visuals_still_returns_a_result(
        self, kin, builder, arm3_urdf, clean_scene
    ):
        """A zero-yield generator must not break the drain loop."""
        model = kin.model_from_urdf(arm3_urdf)
        steps = builder.build_rig_iter(
            model, builder.RigBuildOptions(import_visuals=False)
        )
        with pytest.raises(StopIteration) as stop:
            next(steps)
        assert stop.value.value.armature_object is not None


class TestRigIdentification:
    def test_rig_is_detectable(self, arm3_rig, builder):
        _, result = arm3_rig
        assert builder.is_kinema_rig(result.armature_object)
        assert len(builder.joint_bones(result.armature_object)) == 3

    def test_plain_armature_is_not_a_kinema_rig(self, builder, clean_scene):
        import bpy

        obj = bpy.data.objects.new("plain", bpy.data.armatures.new("plain"))
        assert not builder.is_kinema_rig(obj)
