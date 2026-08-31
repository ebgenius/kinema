"""Placing and reading the tool centre point. Run via ``dev.py test``."""

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
def manager(addon):
    return importlib.import_module(f"{addon.__name__}.solver.manager")


@pytest.fixture
def handlers(addon):
    return importlib.import_module(f"{addon.__name__}.handlers")


@pytest.fixture
def rig(addon, builder, fixture_dir, clean_scene):
    """A 6-DoF rig straight from the importer, untouched and in Object mode.

    Deliberately not posed and nothing selected: that is the state issue #16
    was reported from, and the state the panel is first seen in.
    """
    import bpy

    bpy.ops.kinema.import_urdf(filepath=str(fixture_dir / "arm6.urdf"))
    armature = next(o for o in bpy.data.objects if builder.is_kinema_rig(o))
    bpy.context.view_layer.objects.active = armature
    return armature


def _flange(rig, builder):
    """The joint bone the TCP rides on a freshly imported rig."""
    return rig.data.bones[builder.TCP_BONE].parent.name


def _tool_frame(rig, builder):
    """The TCP's rest tool frame, in armature space."""
    return rig.data.bones[builder.TCP_BONE].matrix_local @ builder.BONE_TO_TOOL


def _close(a, b, tol=1e-6):
    return all(abs(a[r][c] - b[r][c]) < tol for r in range(4) for c in range(4))


class TestIssue16:
    def test_no_bone_anywhere_reports_instead_of_raising(self, rig):
        """The crash: Bone has no `select` in Blender 5.x, and the fallback read it.

        Straight after import there is no active pose bone, so the operator went
        to a selection scan that raised AttributeError on the first bone. It
        must decline politely instead.
        """
        import bpy

        assert bpy.context.object.mode == "OBJECT"
        assert bpy.context.active_pose_bone is None
        rig.data.bones.active = None

        # Not pytest.raises: the point is that nothing is raised at all.
        assert bpy.ops.kinema.set_tcp() == {"CANCELLED"}

    def test_bone_type_really_has_no_select(self, rig):
        """Lock in the API fact the fix rests on, so a revert is loud."""
        assert not hasattr(rig.data.bones[0], "select")
        assert hasattr(rig.pose.bones[0], "select")


class TestSourceResolution:
    def test_an_explicit_bone_needs_no_mode_or_selection(self, rig, builder):
        import bpy

        assert bpy.context.object.mode == "OBJECT"
        assert bpy.ops.kinema.set_tcp(bone="joint4") == {"FINISHED"}
        assert rig.data.bones[builder.TCP_BONE].parent.name == "joint4"

    def test_the_armature_active_bone_is_enough(self, rig, builder):
        """Object mode, no selection -- just an active bone."""
        import bpy

        rig.data.bones.active = rig.data.bones["joint3"]
        assert bpy.ops.kinema.set_tcp() == {"FINISHED"}
        assert rig.data.bones[builder.TCP_BONE].parent.name == "joint3"

    def test_the_pose_mode_path_still_works(self, rig, builder):
        """The demo scripts drive it this way; they must keep working."""
        import bpy

        bpy.ops.object.mode_set(mode="POSE")
        rig.data.bones.active = rig.data.bones["joint5"]
        assert bpy.ops.kinema.set_tcp() == {"FINISHED"}
        bpy.ops.object.mode_set(mode="OBJECT")
        assert rig.data.bones[builder.TCP_BONE].parent.name == "joint5"

    def test_non_joint_bones_are_refused(self, rig, builder):
        import bpy

        for name in (builder.ROOT_BONE, builder.TCP_BONE):
            assert bpy.ops.kinema.set_tcp(bone=name) == {"CANCELLED"}
        assert rig.data.bones[builder.TCP_BONE].parent.name == _flange(rig, builder)

    def test_the_ik_control_is_refused(self, rig, builder, manager):
        """Parenting the TCP to the goal it defines silently disables PyRoki.

        _link_target_for would walk TCP -> .ik -> Root, find no child link, and
        return None, after which the rig falls back to NumPy with the reason
        buried in a collapsed box.
        """
        import bpy

        rig.kinema_solver_mode = "NUMPY"
        bpy.ops.kinema.add_ik()
        ik_name = rig.get(builder.PROP_IK_BONE)

        assert bpy.ops.kinema.set_tcp(bone=ik_name) == {"CANCELLED"}
        assert manager.build_solver(rig, ik_name).link_target is not None


def _bone_where_frames_differ(rig, builder):
    """A joint bone whose own axes differ from its link's.

    The bug only shows on such a bone: where the two frames coincide, building
    from either one gives the same answer and proves nothing.
    """
    for bone in rig.data.bones:
        link = builder.link_frame_of(bone)
        if link is None:
            continue
        if (bone.matrix_local.col[2].xyz - link.col[2].xyz).length > 1e-3:
            return bone.name
    return None


class TestOrientation:
    def test_the_importer_orients_the_tool_from_the_link_frame(self, rig, builder):
        """The invariant everything else is measured against.

        Only the rotation: the tool frame is the *deepest* link, which usually
        sits behind fixed joints from the last actuated one, so its origin is
        offset from the flange -- that distance is what the tool offset holds.
        """
        link = builder.link_frame_of(rig.data.bones[_flange(rig, builder)])
        assert link is not None
        tool = _tool_frame(rig, builder)
        for column in range(3):
            assert (tool.col[column].xyz - link.col[column].xyz).length < 1e-6

    def test_the_seeded_offset_reproduces_the_import(self, rig, builder):
        """Update TCP with the offset the importer recorded is a no-op."""
        import bpy

        before = _tool_frame(rig, builder).copy()
        assert bpy.ops.kinema.set_tcp(bone=_flange(rig, builder)) == {"FINISHED"}
        assert _close(_tool_frame(rig, builder), before, tol=1e-5)

    def test_resetting_onto_the_same_bone_does_not_flip_it(self, rig, builder):
        """Regression: set_tcp built from the *bone's* frame, not the link's.

        Kinema aligns every joint bone's local Y to the joint axis, so the
        bone's Z need not be anywhere near the link's Z -- re-setting the TCP
        left it rolled onto the bone's axes, recoverable only by hand in Edit
        mode.
        """
        import bpy

        name = _bone_where_frames_differ(rig, builder)
        if name is None:
            pytest.skip("no bone on this fixture whose frames differ")

        rig.kinema_tcp_offset = (0.0, 0.0, 0.0)
        rig.kinema_tcp_rpy = (0.0, 0.0, 0.0)
        assert bpy.ops.kinema.set_tcp(bone=name) == {"FINISHED"}

        link = builder.link_frame_of(rig.data.bones[name])
        tool = _tool_frame(rig, builder)
        assert _close(tool, link, tol=1e-5), "the TCP took the bone's axes, not the link's"

        bone_local = rig.data.bones[name].matrix_local
        assert not _close(tool, bone_local, tol=1e-3), (
            "the two frames coincide here, so this proves nothing"
        )

    def test_a_second_pass_is_idempotent(self, rig, builder):
        """Setting the same bone twice must not drift."""
        import bpy

        bpy.ops.kinema.set_tcp(bone="joint3")
        once = _tool_frame(rig, builder).copy()
        bpy.ops.kinema.set_tcp(bone="joint3")
        assert _close(_tool_frame(rig, builder), once, tol=1e-6)


class TestToolOffset:
    def test_a_z_offset_runs_along_the_flange_z(self, rig, builder):
        import bpy
        from mathutils import Vector

        flange = _flange(rig, builder)
        link = builder.link_frame_of(rig.data.bones[flange]).copy()

        rig.kinema_tcp_offset = (0.0, 0.0, 0.15)
        assert bpy.ops.kinema.set_tcp(bone=flange) == {"FINISHED"}

        expected = link @ Vector((0.0, 0.0, 0.15))
        assert (_tool_frame(rig, builder).translation - expected).length < 1e-6

    def test_rotation_follows_the_urdf_rpy_convention(self, rig, builder):
        """Euler((r, p, y), "XYZ") composes as Rz(y) @ Ry(p) @ Rx(r), which is
        what <origin rpy="r p y"> means -- so a tool offset copied out of a
        description can be typed in unchanged."""
        import bpy
        from mathutils import Euler

        flange = _flange(rig, builder)
        link = builder.link_frame_of(rig.data.bones[flange]).copy()

        rpy = (0.0, math.pi / 2.0, 0.0)
        # Translation zeroed, so this is about the rotation alone -- the rig
        # arrives with the flange-to-tool distance already seeded in.
        rig.kinema_tcp_offset = (0.0, 0.0, 0.0)
        rig.kinema_tcp_rpy = rpy
        assert bpy.ops.kinema.set_tcp(bone=flange) == {"FINISHED"}

        expected = link @ Euler(rpy, "XYZ").to_matrix().to_4x4()
        assert _close(_tool_frame(rig, builder), expected, tol=1e-5)

    def test_zero_offset_lands_exactly_on_the_link_frame(self, rig, builder):
        import bpy

        flange = _flange(rig, builder)
        rig.kinema_tcp_offset = (0.0, 0.0, 0.0)
        rig.kinema_tcp_rpy = (0.0, 0.0, 0.0)
        bpy.ops.kinema.set_tcp(bone=flange)

        link = builder.link_frame_of(rig.data.bones[flange])
        assert _close(_tool_frame(rig, builder), link, tol=1e-5)

    def test_the_importer_records_the_fixed_joint_distance(self, rig, builder):
        """The tool link sits behind fixed joints from the last actuated one.

        Fixed joints get no bone, so that distance is invisible on the rig. The
        seeded offset is what puts it in front of the user.
        """
        flange = builder.link_frame_of(rig.data.bones[_flange(rig, builder)])
        expected = flange.inverted_safe() @ _tool_frame(rig, builder)

        assert (
            rig.kinema_tcp_offset - expected.translation
        ).length < 1e-6
        assert rig.kinema_tcp_offset.length > 1e-4, (
            "this fixture has no fixed joint below the last one, so the seeding "
            "is untested here"
        )

    def test_reset_zeroes_the_fields_and_moves_the_marker(self, rig, builder):
        import bpy

        flange = _flange(rig, builder)
        rig.kinema_tcp_offset = (0.0, 0.0, 0.2)
        bpy.ops.kinema.set_tcp(bone=flange)

        assert bpy.ops.kinema.reset_tcp_offset() == {"FINISHED"}
        assert tuple(rig.kinema_tcp_offset) == (0.0, 0.0, 0.0)
        assert tuple(rig.kinema_tcp_rpy) == (0.0, 0.0, 0.0)

        link = builder.link_frame_of(rig.data.bones[flange])
        assert _close(_tool_frame(rig, builder), link, tol=1e-5)

    def test_ik_still_reaches_an_offset_tcp(self, rig, builder, handlers):
        """The chain's tool_offset has to follow the marker."""
        import bpy
        from mathutils import Vector

        rig.kinema_solver_mode = "NUMPY"
        rig.kinema_tcp_offset = (0.0, 0.0, 0.12)
        bpy.ops.kinema.set_tcp(bone=_flange(rig, builder))
        bpy.ops.kinema.add_ik()

        ik_name = rig.get(builder.PROP_IK_BONE)
        matrix = rig.pose.bones[ik_name].matrix.copy()
        matrix.translation += Vector((0.0, 0.0, -0.05))
        rig.pose.bones[ik_name].matrix = matrix
        bpy.context.view_layer.update()
        handlers.solve_rig(rig, force=True)
        bpy.context.view_layer.update()

        goal = rig.pose.bones[ik_name].matrix.translation
        reached = rig.pose.bones[builder.TCP_BONE].matrix.translation
        assert (goal - reached).length < 5e-3


class TestBookkeeping:
    def test_the_link_property_holds_a_urdf_link(self, rig, builder):
        """The panel labels it "Link:", and the importer writes a link there.

        set_tcp used to write the *bone* name, so the label showed a bone as
        soon as the button had been used once.
        """
        import bpy

        bone_names = {b.name for b in rig.data.bones}
        imported = rig.get(builder.PROP_TCP_LINK)
        # The importer records the deepest link, which is behind a fixed joint
        # and so is not any bone's child link -- but it is never a bone name.
        assert imported not in bone_names

        bpy.ops.kinema.set_tcp(bone="joint3")
        after = rig.get(builder.PROP_TCP_LINK)
        assert after == rig.data.bones["joint3"][builder.PROP_CHILD_LINK]
        assert after not in bone_names, "the bone name leaked into the link property"

    def test_the_parent_field_is_seeded_and_maintained(self, rig, builder):
        import bpy

        assert rig.kinema_tcp_parent == _flange(rig, builder)
        bpy.ops.kinema.set_tcp(bone="joint2")
        assert rig.kinema_tcp_parent == "joint2"

    def test_the_marker_keeps_its_widget_and_locks(self, rig, builder):
        import bpy

        bpy.ops.kinema.set_tcp(bone="joint4")
        tcp = rig.pose.bones[builder.TCP_BONE]
        assert tcp.custom_shape is not None
        assert tuple(tcp.lock_location) == (True, True, True)


class TestWidget:
    def test_the_tool_axes_are_distinguishable(self, addon):
        """Three arms of the same length say where the tool is but not which
        way it faces, which is what made the marker unreadable."""
        widgets = importlib.import_module(f"{addon.__name__}.rig.widgets")
        shape = widgets.ensure_widgets()["tcp"]

        vertices = [v.co for v in shape.data.vertices]
        along_bone = max(v.y for v in vertices)
        tool_x = max(v.z for v in vertices)
        tool_y = max(v.x for v in vertices)

        assert along_bone > tool_x > tool_y > 0.0
        assert len(shape.data.edges) > 4, "no arrowhead on the approach axis"


class TestPanelWiring:
    """Headless runs never call draw(), so check what draw() reaches for."""

    def test_the_operators_the_panel_calls_are_registered(self, addon):
        import bpy

        for name in ("set_tcp", "reset_tcp_offset"):
            assert hasattr(bpy.ops.kinema, name), f"kinema.{name} is not registered"

    def test_the_properties_the_panel_draws_exist(self, rig):
        assert isinstance(rig.kinema_tcp_parent, str)
        assert len(rig.kinema_tcp_offset) == 3
        assert len(rig.kinema_tcp_rpy) == 3

    def test_prop_search_targets_a_real_collection(self, rig):
        """The parent field is prop_search(rig, ..., rig.pose, "bones")."""
        assert rig.kinema_tcp_parent in rig.pose.bones

    def test_the_offset_fields_read_as_angles_and_distances(self, addon):
        """subtype drives the units Blender shows -- degrees, and metres."""
        import bpy

        properties = bpy.types.Object.bl_rna.properties
        assert properties["kinema_tcp_rpy"].subtype == "EULER"
        assert properties["kinema_tcp_rpy"].unit == "ROTATION"
        assert properties["kinema_tcp_offset"].subtype in {"TRANSLATION", "XYZ"}
        assert properties["kinema_tcp_offset"].unit == "LENGTH"


class TestPanelReadout:
    def test_the_reported_frame_is_the_tool_frame(self, rig, builder):
        """Not the bone's own matrix: a bone's Y is always head-to-tail, so its
        raw matrix describes the marker rather than the tool."""
        import bpy

        bpy.context.view_layer.update()
        tcp = rig.pose.bones[builder.TCP_BONE]
        tool = rig.matrix_world @ tcp.matrix @ builder.BONE_TO_TOOL

        # The tool's Z is the bone's Y, which is what "Z out of the flange"
        # means for a marker that is itself a bone.
        assert (tool.col[2].xyz - tcp.matrix.col[1].xyz).length < 1e-6
        assert (tool.col[0].xyz - tcp.matrix.col[2].xyz).length < 1e-6

    def test_bone_to_tool_is_a_proper_rotation(self, builder):
        assert abs(builder.BONE_TO_TOOL.to_3x3().determinant() - 1.0) < 1e-9
        assert np.allclose(
            np.array(builder.BONE_TO_TOOL.to_3x3()) @ np.array([0.0, 0.0, 1.0]),
            [0.0, 1.0, 0.0],
        )
