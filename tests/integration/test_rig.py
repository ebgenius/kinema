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


def _world_extent(obj):
    """The object's bounding-box size in world space, as (x, y, z)."""
    import bpy
    from mathutils import Vector

    bpy.context.view_layer.update()
    points = [obj.matrix_world @ Vector(v.co) for v in obj.data.vertices]
    return tuple(
        max(p[axis] for p in points) - min(p[axis] for p in points) for axis in range(3)
    )


@pytest.fixture
def scale_rig(kin, builder, addon, fixture_dir, clean_scene):
    """A rig whose visuals are real mesh files with a <mesh scale>.

    Every other fixture uses primitives, so neither the scale attribute nor a
    mesh file's own unit/up-axis correction was reachable through build_rig.
    """
    yourdfpy = pytest.importorskip("yourdfpy")
    resolve = importlib.import_module(f"{addon.__name__}.io.resolve")
    path = fixture_dir / "mesh_scale.urdf"

    urdf = yourdfpy.URDF.load(str(path))
    # Without a resolver the filenames stay relative to the cwd, which is the
    # repo root rather than the fixture directory.
    model = kin.model_from_urdf(urdf, resolve.make_mesh_resolver(path))
    result = builder.build_rig(model, builder.RigBuildOptions())
    assert not result.warnings, f"visuals failed to load: {result.warnings}"
    return {obj.name.split(".")[0]: obj for obj in result.mesh_objects}, result


class TestMeshScale:
    def test_the_files_own_units_survive_the_build(self, scale_rig):
        """Regression: the DAE correction was computed and then overwritten.

        io/dae.py puts the file's millimetre scale and Y-up rotation into
        matrix_world; the builder then assigned matrix_world outright and threw
        it away, so this mesh came out 1000x too large and lying on its side.
        test_dae.py proved the correction was computed; nothing proved it
        survived a rig build, which is how it lived.
        """
        meshes, _ = scale_rig
        extent = _world_extent(meshes["base"])

        assert extent[0] == pytest.approx(2.0, abs=1e-4), "millimetres were not applied"
        # Y_UP -> Z_UP sends the file's +Z to Blender's -Y, so the 3 m edge
        # lands on Y and the mesh is flat in Z.
        assert extent[1] == pytest.approx(3.0, abs=1e-4), "up-axis was not corrected"
        assert extent[2] == pytest.approx(0.0, abs=1e-4)

    def test_mesh_scale_lands_in_the_geometry(self, scale_rig):
        """scale="2 2 2" must double the mesh, not park 2.0 in obj.scale.

        Scale left in the transform channel is invisible until something works
        in object space -- modifiers, physics, exporters that do not bake
        transforms all quietly use the unscaled mesh.
        """
        meshes, _ = scale_rig
        obj = meshes["scaled"]

        assert tuple(round(v, 6) for v in obj.scale) == (1.0, 1.0, 1.0)
        extent = _world_extent(obj)
        assert extent[0] == pytest.approx(4.0, abs=1e-4)
        assert extent[1] == pytest.approx(6.0, abs=1e-4)

    def test_single_value_scale_imports(self, scale_rig):
        """Regression: scale="0.5" aborted the entire rig build.

        The URDF parser hands a bare float back for the shorthand, which became
        a 0-d array and then a TypeError inside Matrix.Diagonal -- not a
        MeshLoadError, so it escaped the per-visual handler and took the whole
        import with it.
        """
        meshes, _ = scale_rig
        assert "shorthand" in meshes, "the single-value scale aborted the build"

        extent = _world_extent(meshes["shorthand"])
        assert extent[0] == pytest.approx(1.0, abs=1e-4), "0.5 was not applied uniformly"
        assert extent[1] == pytest.approx(1.5, abs=1e-4)

    def test_a_mirrored_mesh_keeps_its_normals(self, scale_rig):
        """A negative scale reverses winding; symmetric robots use it to reuse
        one file for a left and a right part."""
        meshes, _ = scale_rig
        obj = meshes["mirrored"]

        assert tuple(round(v, 6) for v in obj.scale) == (1.0, 1.0, 1.0)
        assert _world_extent(obj)[0] == pytest.approx(2.0, abs=1e-4)
        # Mirroring flips the winding; flip_normals puts it back, so the
        # polygon normals still face the way the unmirrored mesh's do.
        reference = meshes["base"]
        assert obj.data.polygons and reference.data.polygons
        mirrored_z = obj.data.polygons[0].normal.z
        reference_z = reference.data.polygons[0].normal.z
        assert mirrored_z * reference_z > 0.0, "normals were left inside out"


class TestLinkMeshRestore:
    def test_the_rest_transform_is_recorded(self, arm3_rig, builder):
        meshes = builder.link_meshes(arm3_rig[1].armature_object)
        assert meshes, "no link meshes found by the marker"
        for obj in meshes:
            assert builder.link_rest_matrix(obj) is not None
            assert builder.PROP_LINK_NAME in obj

    def test_link_meshes_are_locked(self, arm3_rig, builder):
        """The accident this prevents is a stray G/R/S in the viewport."""
        for obj in builder.link_meshes(arm3_rig[1].armature_object):
            assert tuple(obj.lock_location) == (True, True, True)
            assert tuple(obj.lock_rotation) == (True, True, True)
            assert tuple(obj.lock_scale) == (True, True, True)

    def test_a_moved_mesh_goes_back(self, arm3_rig, builder):
        """The issue: Rest Pose cannot reach a mesh, and nothing else could."""
        import bpy
        from mathutils import Euler, Vector

        model, result = arm3_rig
        rig = result.armature_object
        obj = next(o for o in result.mesh_objects if o.name.startswith("l2"))
        bpy.context.view_layer.update()
        before = obj.matrix_world.copy()

        obj.location = Vector((0.7, -0.3, 1.4))
        obj.rotation_euler = Euler((0.6, -0.2, 1.1), "XYZ")
        obj.scale = (2.0, 0.5, 3.0)
        bpy.context.view_layer.objects.active = rig
        bpy.context.view_layer.update()
        assert (obj.matrix_world.translation - before.translation).length > 1e-3

        assert bpy.ops.kinema.reset_link_meshes() == {"FINISHED"}
        bpy.context.view_layer.update()

        for row in range(4):
            for col in range(4):
                assert abs(obj.matrix_world[row][col] - before[row][col]) < 1e-6

        # And it is still where the URDF says, which is the invariant that
        # matters rather than merely "back where it was".
        expected = model.link_frames()["l2"][:3, 3] + np.array([0.05, 0, 0])
        assert np.allclose(np.array(obj.matrix_world.translation), expected, atol=1e-6)

    def test_it_restores_from_a_posed_rig(self, arm3_rig, builder):
        """Why the *basis* is stored and not the world matrix: the rig need not
        be at rest when the mistake is noticed."""
        import bpy
        from mathutils import Vector

        _, result = arm3_rig
        rig = result.armature_object
        obj = next(o for o in result.mesh_objects if o.name.startswith("l2"))

        rig.pose.bones["joint2"].rotation_euler[1] = 0.7
        bpy.context.view_layer.update()
        posed = obj.matrix_world.copy()

        obj.location = Vector((1.0, 1.0, 1.0))
        bpy.context.view_layer.objects.active = rig
        bpy.ops.kinema.reset_link_meshes()
        bpy.context.view_layer.update()

        for row in range(4):
            for col in range(4):
                assert abs(obj.matrix_world[row][col] - posed[row][col]) < 1e-6

    def test_attachments_are_left_alone(self, arm3_rig, builder, addon):
        """A user's own object must never be moved by this."""
        import bpy
        from mathutils import Vector

        attach = importlib.import_module(f"{addon.__name__}.ops.attach")
        _, result = arm3_rig
        rig = result.armature_object

        source = bpy.data.objects.new("payload", bpy.data.meshes.new("payload_mesh"))
        bpy.context.scene.collection.objects.link(source)
        copy = attach.attach(rig, "joint2", source)
        copy.location = Vector((0.0, 0.0, 0.25))
        bpy.context.view_layer.update()
        placed = copy.matrix_world.copy()

        assert copy not in builder.link_meshes(rig)
        bpy.context.view_layer.objects.active = rig
        bpy.ops.kinema.reset_link_meshes()
        bpy.context.view_layer.update()

        for row in range(4):
            for col in range(4):
                assert abs(copy.matrix_world[row][col] - placed[row][col]) < 1e-6

    def test_a_rig_without_the_marker_says_to_re_import(self, arm3_rig, builder):
        """Rigs imported before this was recorded cannot be restored, and the
        placement genuinely is not recoverable from anything on the rig."""
        import bpy

        _, result = arm3_rig
        rig = result.armature_object
        for obj in builder.link_meshes(rig):
            del obj[builder.PROP_LINK_REST]

        bpy.context.view_layer.objects.active = rig
        assert bpy.ops.kinema.reset_link_meshes() == {"CANCELLED"}

    def test_a_rig_with_no_visuals_is_not_told_to_re_import(
        self, kin, builder, arm3_urdf, clean_scene
    ):
        """Importing without visuals is a choice, not damage.

        Both states reach the same early return, so the message has to tell
        them apart -- "re-import this" is wrong advice for a rig that was
        deliberately built without meshes.
        """
        import bpy

        model = kin.model_from_urdf(arm3_urdf)
        result = builder.build_rig(model, builder.RigBuildOptions(import_visuals=False))
        rig = result.armature_object

        assert builder.link_meshes(rig) == []
        assert not [c for c in rig.children if c.parent_type == "BONE"], (
            "this rig has bone-parented children, so it cannot test the other branch"
        )

        bpy.context.view_layer.objects.active = rig
        assert bpy.ops.kinema.reset_link_meshes() == {"CANCELLED"}

    def test_a_stale_delta_quaternion_is_cleared(self, arm3_rig, builder, addon):
        """Regression: the "already in place" check ignored one delta channel.

        _restore_basis clears all four, so testing only three could report a
        mesh as untouched while it still carried a delta that displaces it.

        The predicate is asserted directly as well as end-to-end: whether a
        delta quaternion also perturbs matrix_basis depends on the rotation
        mode, so the round-trip alone could pass for the wrong reason.
        """
        import bpy
        from mathutils import Quaternion

        pose_ops = importlib.import_module(f"{addon.__name__}.ops.pose")
        _, result = arm3_rig
        rig = result.armature_object
        obj = next(o for o in result.mesh_objects if o.name.startswith("l2"))
        bpy.context.view_layer.update()
        before = obj.matrix_world.copy()

        assert pose_ops._deltas_are_clear(obj), "the fixture starts with deltas set"
        obj.rotation_mode = "QUATERNION"
        obj.delta_rotation_quaternion = Quaternion((1.0, 0.0, 1.0, 0.0)).normalized()

        # The check the bug was in, tested for its own sake.
        assert not pose_ops._deltas_are_clear(obj), (
            "a delta quaternion reads as no delta at all"
        )

        bpy.context.view_layer.objects.active = rig
        bpy.context.view_layer.update()
        assert bpy.ops.kinema.reset_link_meshes() == {"FINISHED"}
        bpy.context.view_layer.update()

        assert tuple(obj.delta_rotation_quaternion) == (1.0, 0.0, 0.0, 0.0)
        for row in range(4):
            for col in range(4):
                assert abs(obj.matrix_world[row][col] - before[row][col]) < 1e-6


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


class TestFailedImportCleanup:
    """A build that dies part-way must not leave its wreckage in the scene.

    This used to be the modal operator's ``_abort``, which ran on Esc or when
    Blender tore the operator down from outside. The import no longer has a
    modal lifecycle -- it is one blocking call -- so the only way it ends early
    is an exception, and the cleanup moved into that handler.
    """

    @pytest.fixture
    def ops(self, addon):
        return importlib.import_module(f"{addon.__name__}.ops.import_robot")

    def test_a_failure_part_way_leaves_nothing_behind(self, ops, builder,
                                                      fixture_dir, clean_scene,
                                                      monkeypatch):
        import bpy

        # A generator that builds the armature for real, hands control back
        # once, then dies -- the shape of a mesh importer failing half way. The
        # point is that `rig` is *partially* populated when it does.
        real_iter = builder.build_rig_iter

        def explode(model, options=None, result=None):
            steps = real_iter(model, options, result=result)
            next(steps)
            yield 0, 1
            raise RuntimeError("mesh importer fell over")

        monkeypatch.setattr(builder, "build_rig_iter", explode)

        with pytest.raises(RuntimeError, match="Could not build rig"):
            bpy.ops.kinema.build_robot(filepath=str(fixture_dir / "arm3.urdf"))

        # The shared WGT-kinema-* bone shapes survive on purpose -- they are
        # created once and reused by every rig, so discard_rig leaves them. The
        # half-built rig itself must be gone.
        assert not [o for o in bpy.data.objects if builder.is_kinema_rig(o)]
        assert not [o for o in bpy.data.objects if not o.name.startswith("WGT-")]

    def test_a_missing_file_creates_nothing(self, ops, clean_scene):
        """bpy.ops turns a reported ERROR into a RuntimeError, so the operator
        cannot simply be compared to CANCELLED here -- but the scene must still
        come out untouched."""
        import bpy

        before = set(bpy.data.objects)
        with pytest.raises(RuntimeError, match="No such file"):
            bpy.ops.kinema.build_robot(filepath="no-such-robot.urdf")
        assert set(bpy.data.objects) == before


class TestRigIdentification:
    def test_rig_is_detectable(self, arm3_rig, builder):
        _, result = arm3_rig
        assert builder.is_kinema_rig(result.armature_object)
        assert len(builder.joint_bones(result.armature_object)) == 3

    def test_plain_armature_is_not_a_kinema_rig(self, builder, clean_scene):
        import bpy

        obj = bpy.data.objects.new("plain", bpy.data.armatures.new("plain"))
        assert not builder.is_kinema_rig(obj)
