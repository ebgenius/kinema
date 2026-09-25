"""Changing the TCP: what it costs, what moves, and the Edit TCP dialog. Run via ``dev.py test``.

The claims:

* a TCP moved or re-offset on the same link keeps its compiled PyRoki solver -- it
  depends on the model and the link, not the tool offset -- so nothing recompiles,
  and neither does taking the IK target off and putting it back;
* on another link the compile is deferred: live IK solves on NumPy until the edit
  settles, while a solve that is not live -- a bake, a render -- builds PyRoki anyway;
* redefining the tool does not move the robot: the IK goal comes to the new TCP;
* Edit TCP places the TCP by typed offset, on the 3D cursor or on an object's origin,
  measured from the parent link where it stands now -- a posed arm on a posed Root
  included -- and stores an offset either way;
* the dialog previews without touching the scene, and remembers its fields.

Two tests compile PyRoki; the rest use the NumPy backend.
"""

from __future__ import annotations

import importlib
import math
import types

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
def manager(addon):
    return importlib.import_module(f"{addon.__name__}.solver.manager")


@pytest.fixture
def pyroki_backend(addon):
    return importlib.import_module(f"{addon.__name__}.solver.pyroki_backend")


@pytest.fixture
def tcp(addon):
    return importlib.import_module(f"{addon.__name__}.ops.tcp")


@pytest.fixture
def preview(addon):
    module = importlib.import_module(f"{addon.__name__}.ui.tcp_preview")
    yield module
    module.stop()


@pytest.fixture(autouse=True)
def _free_compiled_solvers(addon, manager):
    yield
    import gc

    manager.invalidate()
    gc.collect()


def _np4(matrix) -> np.ndarray:
    return np.array([[matrix[r][c] for c in range(4)] for r in range(4)])


@pytest.fixture
def arm6(addon, fixture_dir, clean_scene, builder):
    """arm6, posed off its rest so the arm has somewhere to go if anything moves it."""
    import bpy

    assert "FINISHED" in bpy.ops.kinema.build_robot(
        filepath=str(fixture_dir / "arm6.urdf"), enforce_limits=True
    )
    rig = next(o for o in bpy.data.objects if builder.is_kinema_rig(o))
    for other in bpy.context.view_layer.objects:
        other.select_set(False)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    for name, value in {"joint2": -0.6, "joint3": 1.0, "joint5": 0.4}.items():
        rig.pose.bones[name].rotation_euler[1] = value
    bpy.context.view_layer.update()
    return rig


def _joints(builder, rig) -> np.ndarray:
    values = np.array([pb.rotation_euler[1] for pb in builder.joint_bones(rig)])
    assert len(values) == 6
    return values


def _tool(builder, rig) -> np.ndarray:
    import bpy

    bpy.context.view_layer.update()
    return _np4(rig.pose.bones[builder.TCP_BONE].matrix @ builder.BONE_TO_TOOL)


def _goal_on_tool(builder, rig) -> float:
    import bpy

    bpy.context.view_layer.update()
    goal = _np4(rig.pose.bones[rig[builder.PROP_IK_BONE]].matrix)
    tcp = _np4(rig.pose.bones[builder.TCP_BONE].matrix)
    return float(np.linalg.norm(goal[:3, 3] - tcp[:3, 3]))


def _compiled(manager, rig):
    return [entry[0] for entry in manager._pyroki_cache.values() if entry[2] == rig.name]


def _no_builds(monkeypatch, manager):
    """Make any PyRoki build fail loudly, so a test sees one it did not expect."""

    def refuse(*_args, **_kwargs):
        raise AssertionError("PyRoki was built again")

    monkeypatch.setattr(manager.pyroki_backend, "build", refuse)


class TestChangingTheTcp:
    def test_the_same_link_keeps_its_solver_and_the_arm_its_pose(
        self, arm6, builder, handlers, manager, monkeypatch
    ):
        """Re-offsetting the tool, and taking IK off and on, compile nothing.

        Set TCP threw every compiled solver away and its own depsgraph update
        compiled again -- 5 s here, far more on a laptop -- and live IK then dragged
        the arm to put the moved TCP on the old goal.
        """
        import bpy

        bpy.ops.kinema.add_ik()
        solver = manager.get_solver(arm6)
        assert solver.pyroki_error is None, solver.pyroki_error
        compiled = _compiled(manager, arm6)
        assert len(compiled) == 1, "Add IK Target compiled it"
        joints = _joints(builder, arm6)
        _no_builds(monkeypatch, manager)

        parent = arm6.kinema_tcp_parent
        arm6.kinema_tcp_offset = tuple(np.add(arm6.kinema_tcp_offset, (0.0, 0.02, 0.05)))
        assert "FINISHED" in bpy.ops.kinema.set_tcp(bone=parent)
        assert not manager.deferred(arm6.name), "nothing to wait for"
        bpy.context.view_layer.update()
        handlers.solve_rig(arm6, force=True)
        assert manager.get_solver(arm6).pyroki_error is None
        assert _compiled(manager, arm6) == compiled, "the compiled solver was kept"
        np.testing.assert_allclose(_joints(builder, arm6), joints, atol=1e-6)
        assert _goal_on_tool(builder, arm6) < 1e-6, "the goal came to the new TCP"

        assert "FINISHED" in bpy.ops.kinema.remove_ik()
        assert "FINISHED" in bpy.ops.kinema.add_ik()
        assert manager.get_solver(arm6).pyroki_error is None
        assert _compiled(manager, arm6) == compiled, "re-adding IK reused it"
        np.testing.assert_allclose(_joints(builder, arm6), joints, atol=1e-6)

    def test_another_link_waits_for_its_compile_but_a_bake_does_not(
        self, arm6, builder, handlers, manager
    ):
        """Moved onto joint5, the TCP needs a new solver: live IK waits, a bake builds."""
        import bpy

        arm6.kinema_solver_mode = "NUMPY"
        bpy.ops.kinema.add_ik()
        arm6.kinema_solver_mode = "PYROKI"
        joints = _joints(builder, arm6)

        assert "FINISHED" in bpy.ops.kinema.set_tcp(bone="joint5")
        assert manager.deferred(arm6.name)
        assert _goal_on_tool(builder, arm6) < 1e-6
        np.testing.assert_allclose(_joints(builder, arm6), joints, atol=1e-6)
        assert not _compiled(manager, arm6), "live IK compiled nothing"

        # Not live: a bake's solve, which must not quietly run on NumPy.
        handlers.solve_rig(arm6, force=True)
        assert _compiled(manager, arm6), "PyRoki was built for it"
        assert not manager.deferred(arm6.name), "and there is nothing left to wait for"
        assert manager.get_solver(arm6).last_result.summary.startswith("PyRoki")

    def test_a_tcp_that_moves_the_chain_still_rebuilds_it(self, arm6, builder, manager):
        """Kept compiled is not kept stale: the chain is rebuilt for the new tip."""
        import bpy

        arm6.kinema_solver_mode = "NUMPY"
        bpy.ops.kinema.add_ik()
        assert manager.get_solver(arm6).chain.dof == 6
        assert "FINISHED" in bpy.ops.kinema.set_tcp(bone="joint3")
        assert manager.get_solver(arm6).chain.dof == 3

    def test_a_keyed_goal_goes_back_to_its_keys(self, arm6, builder, handlers):
        """The snap is for the frame you are on; an animated path is the tool's path."""
        import bpy

        arm6.kinema_solver_mode = "NUMPY"
        bpy.ops.kinema.add_ik()
        goal = arm6.pose.bones[arm6[builder.PROP_IK_BONE]]
        goal.keyframe_insert(data_path="location", frame=1)
        goal.keyframe_insert(data_path="rotation_quaternion", frame=1)
        keyed = _np4(goal.matrix)

        arm6.kinema_tcp_offset = tuple(np.add(arm6.kinema_tcp_offset, (0.0, 0.0, 0.05)))
        assert "FINISHED" in bpy.ops.kinema.set_tcp(bone=arm6.kinema_tcp_parent)
        assert _goal_on_tool(builder, arm6) < 1e-6
        bpy.context.scene.frame_set(2)
        bpy.context.scene.frame_set(1)
        np.testing.assert_allclose(_np4(goal.matrix), keyed, atol=1e-6)


    @staticmethod
    def _without_a_link_frame(builder, rig, bone: str) -> None:
        """A joint bone Set TCP accepts, whose placement then cancels."""
        del rig.data.bones[bone][builder.PROP_LINK_CORRECTION]

    def test_a_placement_that_cancels_leaves_no_wait(self, arm6, builder, manager, addon):
        """Deferred before the edit, so a cancelled edit takes its deferral back."""
        import bpy

        deferral = importlib.import_module(f"{addon.__name__}.ops.deferral")
        arm6.kinema_solver_mode = "NUMPY"
        bpy.ops.kinema.add_ik()
        self._without_a_link_frame(builder, arm6, "joint5")
        with pytest.raises(RuntimeError, match="records no link frame"):
            bpy.ops.kinema.set_tcp(bone="joint5")
        assert not manager.deferred(arm6.name)
        assert arm6.name not in deferral._waiting

    def test_a_cancel_keeps_a_wait_an_earlier_edit_started(
        self, arm6, builder, manager, addon
    ):
        import bpy

        deferral = importlib.import_module(f"{addon.__name__}.ops.deferral")
        arm6.kinema_solver_mode = "NUMPY"
        bpy.ops.kinema.add_ik()
        deferral.defer(arm6)  # an edit before this one, still settling
        self._without_a_link_frame(builder, arm6, "joint5")
        with pytest.raises(RuntimeError, match="records no link frame"):
            bpy.ops.kinema.set_tcp(bone="joint5")
        assert manager.deferred(arm6.name)
        assert arm6.name in deferral._waiting


class TestEditTcp:
    def _posed_root(self, builder, rig) -> None:
        import bpy
        from mathutils import Quaternion

        root = rig.pose.bones[builder.ROOT_BONE]
        root.location = (0.2, -0.1, 0.3)
        root.rotation_quaternion = Quaternion((0.0, 0.0, 1.0), 0.7)
        rig.location = (1.0, 0.5, 0.0)
        rig.rotation_euler = (0.0, 0.0, 0.4)
        bpy.context.view_layer.update()

    def test_typed_offsets_are_set_tcps_own(self, arm6, builder):
        import bpy

        assert "FINISHED" in bpy.ops.kinema.edit_tcp(
            parent="joint6", source="OFFSET", offset=(0.01, 0.02, 0.15),
            rotation=(0.0, 0.0, math.pi / 2),
        )
        assert tuple(arm6.kinema_tcp_offset) == pytest.approx((0.01, 0.02, 0.15))
        assert tuple(arm6.kinema_tcp_rpy) == pytest.approx((0.0, 0.0, math.pi / 2))
        assert arm6.data.bones[builder.TCP_BONE].parent.name == "joint6"

    @pytest.mark.parametrize("use_rotation", [True, False])
    def test_the_3d_cursor_places_it_on_a_posed_robot(
        self, arm6, builder, tcp, use_rotation
    ):
        """Measured where the link stands now, so the TCP lands on the cursor."""
        import bpy
        from mathutils import Euler, Matrix

        self._posed_root(builder, arm6)
        tool = _tool(builder, arm6)
        world_tool = _np4(arm6.matrix_world) @ tool
        cursor = Matrix(world_tool.tolist()) @ Matrix.Translation((0.03, -0.02, 0.12))
        cursor = cursor @ Euler((0.3, -0.2, 0.5), "XYZ").to_matrix().to_4x4()
        bpy.context.scene.cursor.matrix = cursor
        typed = (0.0, 0.1, 0.0)

        assert "FINISHED" in bpy.ops.kinema.edit_tcp(
            parent="joint6", source="CURSOR", use_rotation=use_rotation, rotation=typed,
        )
        landed = _np4(arm6.matrix_world) @ _tool(builder, arm6)
        np.testing.assert_allclose(landed[:3, 3], np.array(cursor.translation), atol=1e-5)
        if use_rotation:
            np.testing.assert_allclose(landed[:3, :3], _np4(cursor)[:3, :3], atol=1e-5)
        else:
            assert tuple(arm6.kinema_tcp_rpy) == pytest.approx(typed)

    def test_an_objects_origin_places_it(self, arm6, builder):
        """An empty on the tool: its origin and orientation become the TCP."""
        import bpy
        from mathutils import Euler, Matrix

        empty = bpy.data.objects.new("tool tip", None)
        bpy.context.scene.collection.objects.link(empty)
        tool = _np4(arm6.matrix_world) @ _tool(builder, arm6)
        empty.matrix_world = (
            Matrix(tool.tolist()) @ Matrix.Translation((0.0, 0.05, 0.2))
            @ Euler((0.0, 0.4, 0.0), "XYZ").to_matrix().to_4x4()
        )
        bpy.context.view_layer.update()

        assert "FINISHED" in bpy.ops.kinema.edit_tcp(
            parent="joint6", source="OBJECT", object_name=empty.name, use_rotation=True,
        )
        landed = _np4(arm6.matrix_world) @ _tool(builder, arm6)
        np.testing.assert_allclose(landed, _np4(empty.matrix_world), atol=1e-5)

    def test_a_scaled_object_gives_a_frame_not_its_scale(self, arm6, builder):
        import bpy

        empty = bpy.data.objects.new("scaled", None)
        bpy.context.scene.collection.objects.link(empty)
        empty.matrix_world = arm6.matrix_world @ arm6.pose.bones["joint6"].matrix
        empty.scale = (3.0, 3.0, 3.0)
        bpy.context.view_layer.update()
        assert "FINISHED" in bpy.ops.kinema.edit_tcp(
            parent="joint6", source="OBJECT", object_name=empty.name,
        )
        landed = _tool(builder, arm6)
        np.testing.assert_allclose(landed[:3, :3].T @ landed[:3, :3], np.eye(3), atol=1e-6)

    @pytest.mark.parametrize(
        ("fields", "complaint"),
        [
            ({"parent": "", "source": "OFFSET"}, "Pick the joint"),
            ({"parent": "Root", "source": "OFFSET"}, "not a joint bone"),
            ({"parent": "joint6", "source": "OBJECT", "object_name": ""}, "Pick the object"),
        ],
    )
    def test_what_cannot_be_placed_is_refused(self, arm6, fields, complaint):
        import bpy

        with pytest.raises(RuntimeError, match=complaint):
            bpy.ops.kinema.edit_tcp(**fields)

    def test_the_arm_stays_and_an_uncompiled_link_waits(self, arm6, builder, manager):
        """Through the dialog as through Set TCP. Nothing is compiled yet for this link
        -- the rig solved on NumPy so far -- so its compile waits for the edit to settle.
        A link that has one waits for nothing: see the first test in TestChangingTheTcp.
        """
        import bpy

        arm6.kinema_solver_mode = "NUMPY"
        bpy.ops.kinema.add_ik()
        joints = _joints(builder, arm6)
        assert not manager.deferred(arm6.name)
        assert "FINISHED" in bpy.ops.kinema.edit_tcp(
            parent=arm6.kinema_tcp_parent, source="OFFSET", offset=(0.0, 0.0, 0.2),
        )
        assert manager.deferred(arm6.name)
        np.testing.assert_allclose(_joints(builder, arm6), joints, atol=1e-6)
        assert _goal_on_tool(builder, arm6) < 1e-6


class TestTheTcpPreview:
    def test_a_triad_runs_along_the_frames_axes(self, preview):
        frame = np.eye(4)
        frame[:3, 3] = (1.0, 2.0, 3.0)
        points, colors = preview.triad(frame, 0.5, 0.7)
        np.testing.assert_allclose(points[1::2] - points[::2], np.eye(3) * 0.5)
        assert np.allclose(colors[:, 3], 0.7)
        assert colors[0, 0] > colors[0, 1], "X is red"

    def test_what_is_drawn(self, preview):
        new, current, link = np.eye(4), np.eye(4), np.eye(4)
        new[:3, 3], current[:3, 3] = (0.0, 0.0, 0.3), (0.0, 0.0, 0.1)
        points, colors = preview.shapes(new, current, link, 0.1)
        assert len(points) == 6 + 6 + 2, "two triads and the link line"
        np.testing.assert_allclose(points[-2:], [[0, 0, 0], [0, 0, 0.3]])
        points, _ = preview.shapes(None, current, link, 0.1)
        assert len(points) == 6, "fields that place nothing: only the current TCP"

    def test_opening_and_closing_leaves_nothing(self, arm6, tcp, preview):
        import bpy

        objects = len(bpy.data.objects)
        tcp._drafts.clear()
        dialog = _Dialog(tcp.KINEMA_OT_edit_tcp)
        tcp.KINEMA_OT_edit_tcp.invoke(dialog, _NoDialogContext(bpy.context), None)
        assert preview.active()
        assert dialog.parent == arm6.kinema_tcp_parent, "starts from the TCP as it is"
        dialog.offset = (0.0, 0.0, 0.3)
        dialog.layout = _AnyLayout()
        tcp.KINEMA_OT_edit_tcp.draw(dialog, bpy.context)
        points, _ = preview._state["drawn"]
        assert len(points) == 14
        assert len(bpy.data.objects) == objects
        tcp.KINEMA_OT_edit_tcp.cancel(None, bpy.context)
        assert not preview.active()

    def test_reopening_brings_back_every_field(self, arm6, handlers, tcp, preview):
        import bpy

        tcp._drafts.clear()
        first = _Dialog(tcp.KINEMA_OT_edit_tcp)
        tcp.KINEMA_OT_edit_tcp.invoke(first, _NoDialogContext(bpy.context), None)
        first.source, first.object_name, first.use_rotation = "OBJECT", "Cube", False
        first.offset = (0.0, 0.01, 0.02)
        first.layout = _AnyLayout()
        tcp.KINEMA_OT_edit_tcp.draw(first, bpy.context)
        tcp.KINEMA_OT_edit_tcp.cancel(None, bpy.context)

        second = _Dialog(tcp.KINEMA_OT_edit_tcp)
        tcp.KINEMA_OT_edit_tcp.invoke(second, _NoDialogContext(bpy.context), None)
        for name in ("parent", "source", "object_name", "use_rotation"):
            assert getattr(second, name) == getattr(first, name), name
        assert tuple(second.offset) == pytest.approx((0.0, 0.01, 0.02))

        handlers.on_load_post()
        assert not tcp._drafts


class _AnyLayout:
    """A layout that accepts any drawing call, for running a draw function headless."""

    def __getattr__(self, name):
        return lambda *args, **kwargs: _AnyLayout()


class _NoDialogContext:
    """The real context, but a dialog opened in it is taken as open: invoke runs in the
    background, where Blender has no window to show one in."""

    def __init__(self, real):
        self.real = real
        self.window_manager = types.SimpleNamespace(
            invoke_props_dialog=lambda operator, **_: {"RUNNING_MODAL"}
        )

    def __getattr__(self, name):
        return getattr(self.real, name)


def _Dialog(operator_class):
    """A stand-in for the operator, with every property at its default."""
    import bpy

    properties = bpy.ops.kinema.edit_tcp.get_rna_type().properties
    values = {
        p.identifier: (tuple(p.default_array) if getattr(p, "is_array", False) else p.default)
        for p in properties if p.identifier != "rna_type"
    }
    dialog = types.SimpleNamespace(**values)
    dialog.placement = types.MethodType(operator_class.placement, dialog)
    return dialog
