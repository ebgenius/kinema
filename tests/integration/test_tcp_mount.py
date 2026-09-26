"""A TCP offset of zero is the mounting flange (#65). Run via ``dev.py test``.

flange6 is arm6 with a KR10-style tail: ``flange`` (X out of the face) listed before
``tool0`` (Z out), both on the face 28.5 mm along the last link's -Z. The claims:

* the importer puts the TCP on ``tool0`` with zero offsets, Z out of the face --
  not on ``flange``, which a pick by depth landed on;
* a TCP typed from CAD, measured from the flange, lands where the CAD says, through
  Set TCP and Edit TCP alike;
* a rig with no flange recorded -- arm6, or one built before this -- measures from
  the link frame as before, so an old ``.blend`` keeps its TCP;
* the same TCP pose gives the same IK, flange or not;
* an axis on the tool comes off leaving the TCP where it was.
"""

from __future__ import annotations

import importlib
import math

import numpy as np
import pytest

from ..conftest import requires_bpy

pytestmark = requires_bpy

#: tool0 from the last link, as flange6 (and the KR10 R1100-2) define it.
TOOL0_XYZ, TOOL0_RPY = (0.0, 0.0, -0.0285), (math.pi, 0.0, math.pi)


@pytest.fixture
def builder(addon):
    return importlib.import_module(f"{addon.__name__}.rig.builder")


@pytest.fixture
def kinematics(addon):
    return importlib.import_module(f"{addon.__name__}.rig.kinematics")


@pytest.fixture
def manager(addon):
    return importlib.import_module(f"{addon.__name__}.solver.manager")


@pytest.fixture(autouse=True)
def _free_compiled_solvers(addon, manager):
    yield
    manager.invalidate()


def _np4(matrix) -> np.ndarray:
    return np.array([[matrix[r][c] for c in range(4)] for r in range(4)])


def _import(fixture_dir, builder, name: str):
    import bpy

    existing = {o.name for o in bpy.data.objects}
    assert "FINISHED" in bpy.ops.kinema.build_robot(
        filepath=str(fixture_dir / name), enforce_limits=True
    )
    rig = next(o for o in bpy.data.objects if builder.is_kinema_rig(o) and o.name not in existing)
    for other in bpy.context.view_layer.objects:
        other.select_set(False)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    return rig


@pytest.fixture
def flange6(addon, fixture_dir, clean_scene, builder):
    return _import(fixture_dir, builder, "flange6.urdf")


def _tool(builder, rig) -> np.ndarray:
    """The TCP's tool frame at rest, in armature space."""
    return _np4(rig.data.bones[builder.TCP_BONE].matrix_local @ builder.BONE_TO_TOOL)


def _link(builder, rig, bone="joint6") -> np.ndarray:
    return _np4(builder.link_frame_of(rig.data.bones[bone]))


def _tool0(kinematics) -> np.ndarray:
    return kinematics.make_transform(TOOL0_XYZ, TOOL0_RPY)


def _as_an_old_rig(builder, rig) -> None:
    """What a rig built before the flange was recorded looks like."""
    for bone in rig.data.bones:
        for key in (builder.PROP_MOUNT_FRAME, builder.PROP_MOUNT_LINK):
            if key in bone:
                del bone[key]


class TestTheImport:
    def test_the_tcp_lands_on_tool0_with_zero_offsets(self, flange6, builder, kinematics):
        assert flange6[builder.PROP_TCP_LINK] == "tool0", "not the flange that came first"
        assert tuple(flange6.kinema_tcp_offset) == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
        assert tuple(flange6.kinema_tcp_rpy) == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
        relative = np.linalg.inv(_link(builder, flange6)) @ _tool(builder, flange6)
        np.testing.assert_allclose(relative, _tool0(kinematics), atol=1e-6)

    def test_its_approach_axis_is_out_of_the_flange_face(self, flange6, builder):
        """The face is along the link's -Z; the flange pick left Z in the face."""
        relative = np.linalg.inv(_link(builder, flange6)) @ _tool(builder, flange6)
        np.testing.assert_allclose(relative[:3, 2], [0.0, 0.0, -1.0], atol=1e-6)

    def test_the_flange_is_recorded_on_its_joint(self, flange6, builder, kinematics):
        bone = flange6.data.bones["joint6"]
        assert bone[builder.PROP_MOUNT_LINK] == "tool0"
        np.testing.assert_allclose(
            np.linalg.inv(_link(builder, flange6)) @ _np4(builder.tool_zero_frame(bone)),
            _tool0(kinematics), atol=1e-6,
        )
        assert builder.PROP_MOUNT_FRAME not in flange6.data.bones["joint5"]

    def test_a_robot_without_one_is_as_before(self, addon, fixture_dir, clean_scene, builder):
        """arm6's tool is a plain fixed link: zero stays the link, the offset is seeded."""
        arm6 = _import(fixture_dir, builder, "arm6.urdf")
        assert builder.PROP_MOUNT_FRAME not in arm6.data.bones["joint6"]
        assert arm6[builder.PROP_TCP_LINK] == "tool"
        assert tuple(arm6.kinema_tcp_offset) == pytest.approx((0.05, 0.0, 0.0), abs=1e-6)


class TestATcpFromCad:
    def test_set_tcp_measures_from_the_flange(self, flange6, builder, kinematics):
        """150 mm out of the face and a quarter turn about the tool axis, as CAD gives it."""
        import bpy

        cad = kinematics.make_transform((0.01, 0.0, 0.15), (0.0, 0.0, math.pi / 2))
        flange6.kinema_tcp_offset, flange6.kinema_tcp_rpy = (0.01, 0.0, 0.15), (0, 0, math.pi / 2)
        assert "FINISHED" in bpy.ops.kinema.set_tcp(bone="joint6")
        expected = _link(builder, flange6) @ _tool0(kinematics) @ cad
        np.testing.assert_allclose(_tool(builder, flange6), expected, atol=1e-6)

    def test_edit_tcp_measures_from_the_flange_on_a_posed_robot(self, flange6, builder):
        """The cursor, placed off tool0 by a known amount, reads back as that offset."""
        import bpy
        from mathutils import Matrix

        for name, value in {"joint2": -0.5, "joint3": 0.9, "joint6": 0.7}.items():
            flange6.pose.bones[name].rotation_euler[1] = value
        bpy.context.view_layer.update()
        posed_tool = flange6.pose.bones[builder.TCP_BONE].matrix @ builder.BONE_TO_TOOL
        bpy.context.scene.cursor.matrix = (
            flange6.matrix_world @ posed_tool @ Matrix.Translation((0.0, 0.02, 0.12))
        )
        assert "FINISHED" in bpy.ops.kinema.edit_tcp(
            parent="joint6", source="CURSOR", use_rotation=True,
        )
        assert tuple(flange6.kinema_tcp_offset) == pytest.approx((0.0, 0.02, 0.12), abs=1e-6)
        assert tuple(flange6.kinema_tcp_rpy) == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)


class TestOldRigs:
    def test_an_old_rig_keeps_the_meaning_of_its_offsets(self, flange6, builder):
        """No flange recorded: zero is the link, and the old importer's offset -- the
        flange's placement from it -- reproduces where it put the TCP."""
        import bpy

        _as_an_old_rig(builder, flange6)
        old_offset, old_rpy = (0.0, 0.0, -0.0285), (0.0, math.pi / 2, 0.0)
        flange6.kinema_tcp_offset, flange6.kinema_tcp_rpy = old_offset, old_rpy
        assert "FINISHED" in bpy.ops.kinema.set_tcp(bone="joint6")
        relative = np.linalg.inv(_link(builder, flange6)) @ _tool(builder, flange6)
        np.testing.assert_allclose(relative[:3, 3], old_offset, atol=1e-6)
        np.testing.assert_allclose(
            relative[:3, 0], [0.0, 0.0, -1.0], atol=1e-6,
            err_msg="the flange frame, as the old importer placed it: X out of the face",
        )

    def test_the_panel_says_what_zero_is(self, addon, flange6, builder):
        panel = importlib.import_module(f"{addon.__name__}.ui.panel")
        assert panel._offset_zero(flange6) == "From the mounting flange (tool0)"
        _as_an_old_rig(builder, flange6)
        assert panel._offset_zero(flange6) == "From the joint's link frame"


class TestTheSolver:
    def test_the_same_tcp_pose_solves_the_same(
        self, addon, fixture_dir, clean_scene, builder, manager
    ):
        """Where the TCP stands is all the solver sees; how its offset was measured
        is not. One rig measures from the flange, one from the link, same TCP."""
        import bpy
        from mathutils import Matrix

        new = _import(fixture_dir, builder, "flange6.urdf")
        old = _import(fixture_dir, builder, "flange6.urdf")
        _as_an_old_rig(builder, old)
        old.kinema_tcp_offset, old.kinema_tcp_rpy = (0.0, 0.0, -0.0285), (math.pi, 0.0, math.pi)
        bpy.context.view_layer.objects.active = old
        assert "FINISHED" in bpy.ops.kinema.set_tcp(bone="joint6")
        np.testing.assert_allclose(_tool(builder, old), _tool(builder, new), atol=1e-6)

        answers = []
        for rig in (new, old):
            rig.kinema_solver_mode = "NUMPY"
            for name, value in {"joint2": -0.6, "joint3": 1.0, "joint5": 0.4}.items():
                rig.pose.bones[name].rotation_euler[1] = value
            bpy.context.view_layer.objects.active = rig
            bpy.ops.kinema.add_ik()
            ik = rig.pose.bones[rig[builder.PROP_IK_BONE]]
            goal = _np4(ik.matrix)
            goal[:3, 3] += (0.03, -0.02, 0.02)
            ik.matrix = Matrix(goal.tolist())
            bpy.context.view_layer.update()
            solver = manager.get_solver(rig)
            solver.solve(rig, "NUMPY")
            answers.append([pb.rotation_euler[1] for pb in builder.joint_bones(rig)])
        assert manager.get_solver(new).link_target[0] == manager.get_solver(old).link_target[0]
        np.testing.assert_allclose(answers[0], answers[1], atol=1e-6)


class TestAnAxisOnTheTool:
    def test_it_comes_off_leaving_the_tcp_where_it_was(self, addon, flange6, builder):
        """The axis stores the TCP's offset from its zero and puts it back from it."""
        axes = importlib.import_module(f"{addon.__name__}.ops.external_axes")
        spec = importlib.import_module(f"{addon.__name__}.rig.external_axes")
        before = _tool(builder, flange6)
        axes.add_axis(flange6, spec.AxisSpec(
            "spindle", mount="AFTER", kind="ROTARY", direction="Z", continuous=True,
            offset_location=(0.0, 0.0, 0.05),
        ))
        assert not np.allclose(_tool(builder, flange6), before, atol=1e-4)
        axes.remove_axis(flange6, "spindle")
        np.testing.assert_allclose(_tool(builder, flange6), before, atol=1e-6)
        assert tuple(flange6.kinema_tcp_offset) == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
