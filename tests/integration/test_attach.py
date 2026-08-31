"""Attaching objects and collections to rig bones. Run via ``dev.py test``."""

from __future__ import annotations

import importlib

import pytest

from ..conftest import requires_bpy

pytestmark = requires_bpy


@pytest.fixture
def builder(addon):
    return importlib.import_module(f"{addon.__name__}.rig.builder")


@pytest.fixture
def attach(addon):
    return importlib.import_module(f"{addon.__name__}.ops.attach")


@pytest.fixture
def rig(addon, builder, fixture_dir, clean_scene):
    """A 6-DoF rig imported through the real operator, folded off its rest pose.

    Posed rather than at rest so "does the attachment ride the bone?" and "does
    detaching leave it in place?" are questions with a visible answer.
    """
    import bpy

    bpy.ops.kinema.import_urdf(filepath=str(fixture_dir / "arm6.urdf"))
    armature = next(o for o in bpy.data.objects if builder.is_kinema_rig(o))
    bpy.context.view_layer.objects.active = armature
    armature.pose.bones["joint2"].rotation_euler[1] = -0.7
    armature.pose.bones["joint4"].rotation_euler[1] = 0.9
    bpy.context.view_layer.update()
    return armature


@pytest.fixture
def cube(clean_scene):
    """A source object to attach, deliberately not at the world origin."""
    import bpy

    mesh = bpy.data.meshes.new("payload_mesh")
    obj = bpy.data.objects.new("payload", mesh)
    obj.location = (1.0, 2.0, 3.0)
    bpy.context.scene.collection.objects.link(obj)
    # matrix_world is evaluated, not stored: without this it reads as identity
    # and every "did it move?" assertion below would compare against the wrong
    # starting point.
    bpy.context.view_layer.update()
    return obj


def _bone_head_world(rig, bone_name):
    """Where the bone's head actually is, in world space."""
    import bpy

    bpy.context.view_layer.update()
    return (rig.matrix_world @ rig.pose.bones[bone_name].matrix).translation.copy()


def _bone_tail_world(rig, bone_name):
    import bpy
    from mathutils import Vector

    bpy.context.view_layer.update()
    pose_bone = rig.pose.bones[bone_name]
    matrix = rig.matrix_world @ pose_bone.matrix
    return (matrix @ Vector((0.0, pose_bone.bone.length, 0.0))).copy()


class TestAttach:
    def test_parents_the_copy_to_the_bone(self, rig, builder, attach, cube):
        copy = attach.attach(rig, "joint3", cube)

        assert copy is not None
        assert copy.parent is rig
        assert copy.parent_type == "BONE"
        assert copy.parent_bone == "joint3"
        assert copy[builder.PROP_ATTACHMENT] == "joint3"
        assert copy[builder.PROP_ATTACH_SOURCE] == "payload"

    def test_sits_on_the_bone_head_not_its_tail(self, rig, attach, cube):
        """The regression matrix_parent_inverse exists to prevent.

        Blender's bone parenting measures from the tail, so a freshly parented
        object lands at the far end of the bone and every offset the user then
        dials in is measured from a frame that has nothing to do with the joint.
        """
        copy = attach.attach(rig, "joint3", cube)

        head = _bone_head_world(rig, "joint3")
        tail = _bone_tail_world(rig, "joint3")
        placed = copy.matrix_world.translation

        assert (placed - head).length < 1e-6
        assert (placed - tail).length > 1e-4, "landed on the tail, not the head"

    def test_offset_is_the_copys_own_transform(self, rig, attach, cube):
        """A location of (0, 0, 0.1) must mean 100 mm along the bone's own Z."""
        import bpy
        from mathutils import Vector

        copy = attach.attach(rig, "joint3", cube)
        copy.location = (0.0, 0.0, 0.1)
        bpy.context.view_layer.update()

        expected = (
            rig.matrix_world
            @ rig.pose.bones["joint3"].matrix
            @ Vector((0.0, 0.0, 0.1))
        )
        assert (copy.matrix_world.translation - expected).length < 1e-6

    def test_a_quaternion_source_still_lands_on_the_bone(self, rig, attach, cube):
        """A copy keeps its source's rotation_mode.

        Zeroing rotation_euler leaves a quaternion source rotated, because that
        is not the channel Blender is reading -- so the "zero" offset would put
        the attachment somewhere other than the bone head.
        """
        import bpy
        from mathutils import Quaternion

        cube.rotation_mode = "QUATERNION"
        cube.rotation_quaternion = Quaternion((1.0, 1.0, 0.0, 0.0)).normalized()
        bpy.context.view_layer.update()

        copy = attach.attach(rig, "joint3", cube)
        bpy.context.view_layer.update()

        assert copy.rotation_mode == "QUATERNION"
        head = _bone_head_world(rig, "joint3")
        assert (copy.matrix_world.translation - head).length < 1e-6
        expected = (rig.matrix_world @ rig.pose.bones["joint3"].matrix).to_3x3()
        assert (copy.matrix_world.to_3x3() - expected).median_scale < 1e-6

    def test_an_axis_angle_source_still_lands_on_the_bone(self, rig, attach, cube):
        import bpy

        cube.rotation_mode = "AXIS_ANGLE"
        cube.rotation_axis_angle = (1.2, 0.0, 0.0, 1.0)
        bpy.context.view_layer.update()

        copy = attach.attach(rig, "joint3", cube)
        bpy.context.view_layer.update()

        assert (copy.matrix_world.translation - _bone_head_world(rig, "joint3")).length < 1e-6

    def test_source_delta_transforms_do_not_survive(self, rig, attach, cube):
        """Deltas are copied too, and feed matrix_basis on top of everything else."""
        import bpy

        cube.delta_location = (0.3, -0.2, 0.7)
        cube.delta_rotation_euler = (0.4, 0.0, 0.0)
        cube.delta_scale = (2.0, 2.0, 2.0)
        bpy.context.view_layer.update()

        copy = attach.attach(rig, "joint3", cube)
        bpy.context.view_layer.update()

        assert (copy.matrix_world.translation - _bone_head_world(rig, "joint3")).length < 1e-6
        assert tuple(copy.delta_location) == (0.0, 0.0, 0.0)

    def test_source_constraints_do_not_come_along(self, rig, attach, cube, clean_scene):
        """A constraint would go on driving the copy, so its transform would no
        longer be the offset the panel claims it is."""
        import bpy

        anchor = bpy.data.objects.new("anchor", None)
        bpy.context.scene.collection.objects.link(anchor)
        anchor.location = (5.0, 5.0, 5.0)
        constraint = cube.constraints.new("COPY_LOCATION")
        constraint.target = anchor
        bpy.context.view_layer.update()

        copy = attach.attach(rig, "joint3", cube)
        bpy.context.view_layer.update()

        assert len(copy.constraints) == 0
        assert (copy.matrix_world.translation - _bone_head_world(rig, "joint3")).length < 1e-6
        assert len(cube.constraints) == 1, "the source lost its own constraint"

    def test_the_copy_is_linked_not_duplicated(self, rig, attach, cube):
        """Fix the harness once, every link wearing it updates."""
        copy = attach.attach(rig, "joint3", cube)

        assert copy is not cube
        assert copy.data is cube.data, "attachment does not share its source's mesh"

    def test_the_same_source_serves_several_bones(self, rig, attach, cube):
        """The cable-harness case: one model, many links, different offsets."""
        import bpy

        first = attach.attach(rig, "joint2", cube)
        second = attach.attach(rig, "joint4", cube)
        bpy.context.view_layer.update()

        assert first is not second
        assert first.data is second.data is cube.data
        assert (
            first.matrix_world.translation - second.matrix_world.translation
        ).length > 1e-4

    def test_leaves_the_source_alone(self, rig, attach, cube):
        import bpy
        from mathutils import Vector

        before = cube.matrix_world.copy()
        attach.attach(rig, "joint3", cube)
        bpy.context.view_layer.update()

        assert cube.parent is None
        assert (cube.matrix_world.translation - Vector(before.translation)).length < 1e-9

    def test_posing_the_bone_carries_the_attachment(self, rig, attach, cube):
        import bpy

        copy = attach.attach(rig, "joint3", cube)
        before = copy.matrix_world.translation.copy()

        rig.pose.bones["joint2"].rotation_euler[1] = 0.6
        bpy.context.view_layer.update()

        moved = copy.matrix_world.translation
        assert (moved - before).length > 1e-3
        assert (moved - _bone_head_world(rig, "joint3")).length < 1e-6

    def test_a_collection_becomes_an_instance(self, rig, attach, cube, clean_scene):
        import bpy

        harness = bpy.data.collections.new("harness")
        harness.objects.link(cube)

        copy = attach.attach(rig, "joint3", harness)

        assert copy.type == "EMPTY"
        assert copy.instance_type == "COLLECTION"
        assert copy.instance_collection is harness
        assert copy.parent_bone == "joint3"

        bpy.data.collections.remove(harness)

    def test_replacing_removes_the_superseded_copy(self, rig, attach, cube, builder):
        import bpy

        first = attach.attach(rig, "joint3", cube)
        first_name = first.name

        other = bpy.data.objects.new("other", bpy.data.meshes.new("other_mesh"))
        bpy.context.scene.collection.objects.link(other)
        second = attach.attach(rig, "joint3", other)

        assert first_name not in bpy.data.objects, "superseded copy was orphaned"
        assert builder.bone_attachment(rig, "joint3") is second

    def test_a_source_in_another_scene_attaches(self, rig, attach, builder):
        """bpy.data is file-wide, so the picker can reach across scenes."""
        import bpy

        other_scene = bpy.data.scenes.new("elsewhere")
        far = bpy.data.objects.new("far", bpy.data.meshes.new("far_mesh"))
        other_scene.collection.objects.link(far)
        assert far.name not in bpy.context.scene.objects

        copy = attach.attach(rig, "joint3", far)

        assert copy is not None
        assert copy.name in bpy.context.scene.objects, "copy did not land in this scene"
        assert far.name not in bpy.context.scene.objects, "the source was moved"

        bpy.data.scenes.remove(other_scene)


class TestDetach:
    def test_leaves_the_object_where_it_appears(self, rig, attach, cube):
        import bpy

        copy = attach.attach(rig, "joint3", cube)
        copy.location = (0.0, 0.05, 0.02)
        bpy.context.view_layer.update()
        before = copy.matrix_world.copy()

        attach.detach(copy)
        bpy.context.view_layer.update()

        assert copy.name in bpy.data.objects, "detach deleted the object"
        assert copy.parent is None
        assert copy.parent_bone == ""
        for row in range(4):
            for column in range(4):
                assert abs(copy.matrix_world[row][column] - before[row][column]) < 1e-6

    def test_clears_the_marker(self, rig, builder, attach, cube):
        copy = attach.attach(rig, "joint3", cube)
        attach.detach(copy)

        assert builder.PROP_ATTACHMENT not in copy
        assert builder.bone_attachment(rig, "joint3") is None

    def test_operator_detaches_the_named_bone(self, rig, builder, attach, cube):
        import bpy

        attach.attach(rig, "joint3", cube)
        bpy.context.view_layer.objects.active = rig

        assert bpy.ops.kinema.detach_from_bone(bone="joint3") == {"FINISHED"}
        assert builder.bone_attachment(rig, "joint3") is None

    def test_operator_reports_when_there_is_nothing_to_detach(self, rig):
        import bpy

        bpy.context.view_layer.objects.active = rig
        assert bpy.ops.kinema.detach_from_bone(bone="joint3") == {"CANCELLED"}


class TestLookup:
    def test_link_visuals_are_not_mistaken_for_attachments(self, rig, builder):
        """The importer bone-parents link meshes too; only the marker separates them."""
        assert builder.attachments(rig) == []
        assert any(
            child.parent_type == "BONE" for child in rig.children
        ), "fixture has no bone-parented link meshes, so this proves nothing"

    def test_finds_only_the_requested_bone(self, rig, builder, attach, cube):
        attach.attach(rig, "joint2", cube)

        assert builder.bone_attachment(rig, "joint2") is not None
        assert builder.bone_attachment(rig, "joint4") is None
        assert len(builder.attachments(rig)) == 1


class TestPanelWiring:
    """Headless runs never call draw(), so check what draw() reaches for.

    A mistyped operator idname or property name in a UIList row is invisible
    until someone opens the sidebar, which is exactly the kind of breakage a
    background test suite is bad at catching.
    """

    @pytest.fixture
    def panel(self, addon):
        return importlib.import_module(f"{addon.__name__}.ui.panel")

    def test_the_operators_the_rows_call_are_registered(self, addon):
        import bpy

        for name in (
            "attach_to_bone",
            "detach_from_bone",
            "reset_attachment_offset",
            "select_attachment",
            "set_ik_tip",
        ):
            assert hasattr(bpy.ops.kinema, name), f"kinema.{name} is not registered"

    def test_the_properties_the_rows_draw_exist(self, rig):
        pose_bone = rig.pose.bones["joint3"]

        assert pose_bone.kinema_attach_type == "OBJECT"
        assert pose_bone.kinema_attach_object is None
        assert pose_bone.kinema_attach_collection is None
        assert rig.kinema_ik_tip == -1
        assert rig.kinema_active_bone_index == 0

    def test_the_list_and_its_panel_are_registered(self, addon):
        import bpy

        assert hasattr(bpy.types, "KINEMA_UL_bones")
        assert hasattr(bpy.types, "KINEMA_PT_bones")

    def test_active_bone_follows_the_index(self, rig, panel):
        rig.kinema_active_bone_index = 2
        # By name, not identity: RNA hands out a fresh PoseBone wrapper on
        # every access, so `is` compares two objects that are never the same.
        assert panel.active_bone(rig).name == rig.pose.bones[2].name

    def test_active_bone_survives_an_index_past_the_end(self, rig, panel):
        """The list index outlives the bones when a rig is rebuilt smaller."""
        rig.kinema_active_bone_index = 999
        assert panel.active_bone(rig) is None

    def test_joint_indices_match_what_set_ik_tip_expects(self, rig, panel, builder):
        indices = panel._joint_indices(rig)
        joints = builder.joint_bones(rig)

        assert indices == {bone.name: i for i, bone in enumerate(joints)}
        assert builder.TCP_BONE not in indices, "the TCP is not a joint"

    def test_the_offset_panel_draws_the_active_rotation_channel(
        self, rig, panel, attach, cube
    ):
        """Drawing rotation_euler on a quaternion attachment offers a dead field."""
        copy = attach.attach(rig, "joint3", cube)

        copy.rotation_mode = "XYZ"
        assert panel._rotation_channel(copy) == "rotation_euler"
        copy.rotation_mode = "QUATERNION"
        assert panel._rotation_channel(copy) == "rotation_quaternion"
        copy.rotation_mode = "AXIS_ANGLE"
        assert panel._rotation_channel(copy) == "rotation_axis_angle"

    def test_the_tcp_row_offers_a_radio(self, rig, panel, builder):
        """Without one there is no way back to the default target from the list."""
        tcp_name = rig.get(builder.PROP_TCP_BONE)

        assert panel.tip_index_of(rig, rig.pose.bones[tcp_name]) == -1
        assert panel.tip_index_of(rig, rig.pose.bones["joint3"]) == 2
        assert panel.tip_index_of(rig, rig.pose.bones[builder.ROOT_BONE]) is None

    def test_every_radio_index_round_trips_through_set_ik_tip(self, rig, panel, builder):
        """Whatever the list can offer, the operator must accept."""
        import bpy

        bpy.context.view_layer.objects.active = rig
        offered = [
            panel.tip_index_of(rig, bone)
            for bone in rig.pose.bones
            if panel.tip_index_of(rig, bone) is not None
        ]
        assert -1 in offered and len(offered) == len(builder.joint_bones(rig)) + 1

        for index in offered:
            assert bpy.ops.kinema.set_ik_tip(index=index) == {"FINISHED"}
            assert rig.kinema_ik_tip == index


class TestOperators:
    def test_attach_operator_places_a_linked_copy(self, rig, builder, cube):
        import bpy

        bpy.context.view_layer.objects.active = rig
        assert bpy.ops.kinema.attach_to_bone(
            bone="joint3", source="payload"
        ) == {"FINISHED"}

        placed = builder.bone_attachment(rig, "joint3")
        assert placed is not None and placed.data is cube.data

    def test_attach_operator_takes_a_rig_by_name(self, rig, builder, cube):
        """The timer path names the rig, because the active object may be anything."""
        import bpy

        bpy.context.view_layer.objects.active = cube
        assert bpy.ops.kinema.attach_to_bone(
            rig=rig.name, bone="joint3", source="payload"
        ) == {"FINISHED"}

        assert builder.bone_attachment(rig, "joint3") is not None

    def test_attach_operator_sets_the_row(self, rig, cube):
        import bpy

        bpy.context.view_layer.objects.active = rig
        bpy.ops.kinema.attach_to_bone(bone="joint3", source="payload")

        assert rig.pose.bones["joint3"].kinema_attach_object is cube

    def test_attach_operator_rejects_a_cycle(self, rig, builder):
        """The operator is script-reachable, so the picker's poll is not enough."""
        import bpy

        bpy.context.view_layer.objects.active = rig
        with pytest.raises(RuntimeError, match="own parent"):
            bpy.ops.kinema.attach_to_bone(bone="joint3", source=rig.name)

        link_mesh = next(c for c in rig.children if c.parent_type == "BONE")
        with pytest.raises(RuntimeError, match="own parent"):
            bpy.ops.kinema.attach_to_bone(bone="joint3", source=link_mesh.name)

        assert builder.bone_attachment(rig, "joint3") is None

    def test_attach_operator_rejects_a_collection_holding_the_rig(self, rig, builder):
        import bpy

        bpy.context.view_layer.objects.active = rig
        holder = bpy.data.collections.new("holder")
        holder.objects.link(rig)
        try:
            with pytest.raises(RuntimeError, match="own parent"):
                bpy.ops.kinema.attach_to_bone(
                    bone="joint3", source="holder", is_collection=True
                )
            assert builder.bone_attachment(rig, "joint3") is None
        finally:
            bpy.data.collections.remove(holder)

    def test_attach_operator_rejects_an_empty_instancing_the_rig(self, rig, builder):
        """The parent walk cannot see a cycle that lives inside an instance.

        An Empty instancing a collection that holds the rig is not parented to
        the rig at all, so the ancestry check passes -- but copy() carries the
        instance_collection along, and the copy then instances its own parent.
        """
        import bpy

        bpy.context.view_layer.objects.active = rig
        holder = bpy.data.collections.new("holder")
        holder.objects.link(rig)
        proxy = bpy.data.objects.new("proxy", None)
        proxy.instance_type = "COLLECTION"
        proxy.instance_collection = holder
        bpy.context.scene.collection.objects.link(proxy)
        try:
            assert proxy.parent is None, "the ancestry check would catch it otherwise"
            with pytest.raises(RuntimeError, match="own parent"):
                bpy.ops.kinema.attach_to_bone(bone="joint3", source="proxy")
            assert builder.bone_attachment(rig, "joint3") is None
        finally:
            bpy.data.objects.remove(proxy, do_unlink=True)
            bpy.data.collections.remove(holder)

    def test_attach_operator_accepts_an_unrelated_collection_instance(self, rig, builder):
        """...while an Empty instancing anything else is a normal attachment."""
        import bpy

        bpy.context.view_layer.objects.active = rig
        harness = bpy.data.collections.new("harness")
        harness.objects.link(bpy.data.objects.new("bracket", None))
        proxy = bpy.data.objects.new("proxy", None)
        proxy.instance_type = "COLLECTION"
        proxy.instance_collection = harness
        bpy.context.scene.collection.objects.link(proxy)
        try:
            assert bpy.ops.kinema.attach_to_bone(
                bone="joint3", source="proxy"
            ) == {"FINISHED"}
            assert builder.bone_attachment(rig, "joint3") is not None
        finally:
            bpy.data.collections.remove(harness)

    def test_attach_operator_rejects_an_unknown_source(self, rig):
        import bpy

        bpy.context.view_layer.objects.active = rig
        with pytest.raises(RuntimeError, match="not in this file"):
            bpy.ops.kinema.attach_to_bone(bone="joint3", source="nope")

    def test_detach_can_delete_instead_of_keeping(self, rig, builder, attach, cube):
        import bpy

        copy = attach.attach(rig, "joint3", cube)
        name = copy.name
        bpy.context.view_layer.objects.active = rig

        assert bpy.ops.kinema.detach_from_bone(bone="joint3", keep=False) == {"FINISHED"}
        assert name not in bpy.data.objects
        assert builder.bone_attachment(rig, "joint3") is None

    def test_reset_offset_puts_it_back_on_the_bone(self, rig, attach, cube):
        import bpy

        copy = attach.attach(rig, "joint3", cube)
        copy.location = (0.1, 0.2, 0.3)
        copy.scale = (2.0, 2.0, 2.0)
        bpy.context.view_layer.objects.active = rig

        assert bpy.ops.kinema.reset_attachment_offset(bone="joint3") == {"FINISHED"}
        assert tuple(copy.location) == (0.0, 0.0, 0.0)
        assert tuple(copy.scale) == (1.0, 1.0, 1.0)
        assert (copy.matrix_world.translation - _bone_head_world(rig, "joint3")).length < 1e-6

    def test_reset_offset_handles_a_quaternion_attachment(self, rig, attach, cube):
        """Reset must clear whichever rotation channel is actually active."""
        import bpy
        from mathutils import Quaternion

        copy = attach.attach(rig, "joint3", cube)
        copy.rotation_mode = "QUATERNION"
        copy.rotation_quaternion = Quaternion((1.0, 0.0, 1.0, 0.0)).normalized()
        copy.location = (0.1, 0.0, 0.0)
        bpy.context.view_layer.objects.active = rig
        bpy.context.view_layer.update()

        assert bpy.ops.kinema.reset_attachment_offset(bone="joint3") == {"FINISHED"}
        bpy.context.view_layer.update()

        expected = rig.matrix_world @ rig.pose.bones["joint3"].matrix
        assert (copy.matrix_world.to_3x3() - expected.to_3x3()).median_scale < 1e-6
        assert (copy.matrix_world.translation - expected.translation).length < 1e-6


class TestPickerCallback:
    """The path the UI actually takes: a name, applied one tick later."""

    def test_applying_a_pick_attaches(self, rig, builder, attach, cube):
        attach._apply_pick(rig.name, "joint3", cube.name, False)

        placed = builder.bone_attachment(rig, "joint3")
        assert placed is not None
        assert placed.data is cube.data

    def test_clearing_a_pick_removes_the_attachment(self, rig, builder, attach, cube):
        attach._apply_pick(rig.name, "joint3", cube.name, False)
        attach._apply_pick(rig.name, "joint3", None, False)

        assert builder.bone_attachment(rig, "joint3") is None
        assert rig.pose.bones["joint3"].kinema_attach_object is None

    def test_a_deleted_source_is_survivable(self, rig, builder, attach):
        """The tick can land after the picked object was removed."""
        attach._apply_pick(rig.name, "joint3", "gone", False)
        assert builder.bone_attachment(rig, "joint3") is None

    def test_a_vanished_rig_is_survivable(self, attach, cube):
        """The tick can land after an undo or a file load took the rig away."""
        attach._apply_pick("no-such-rig", "joint3", cube.name, False)

    def test_the_picker_rejects_the_rig_itself(self, rig, attach):
        """Attaching a rig to its own bone is a depsgraph cycle."""
        pose_bone = rig.pose.bones["joint3"]

        assert not attach._can_attach(pose_bone, rig)
        # ...and its own link meshes, which are already parented to it.
        child = next(c for c in rig.children if c.parent_type == "BONE")
        assert not attach._can_attach(pose_bone, child)

    def test_the_picker_accepts_an_unrelated_object(self, rig, attach, cube):
        assert attach._can_attach(rig.pose.bones["joint3"], cube)
