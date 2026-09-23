"""External axes on a real rig. Run via ``dev.py test``.

The claims:

* a track added under arm6 in Blender *is* rail6 -- the same arm on the same rail,
  written in URDF by hand -- joint for joint and pose for pose,
* the model PyRoki is given for a rig with axes moves exactly as the rig does, and
  on a rig without them agrees with the description it came from,
* PyRoki solves through it, a released track included,
* an axis under the robot that moves it moves everything riding the robot -- the
  TCP, the IK target, the base link's meshes, the waypoints -- and removing it puts
  all of it back, in either order when two are stacked,
* joint indices follow names: waypoints and the IK tip, keys included,
* an axis on the tool carries the TCP, and removing it puts the TCP back,
* a standalone axis moves nothing of the robot,
* placeholders ride their bones and leave with their axis.

Only one test compiles PyRoki; the rest use the NumPy backend or no solver.
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
def manager(addon):
    return importlib.import_module(f"{addon.__name__}.solver.manager")


@pytest.fixture
def ops(addon):
    return importlib.import_module(f"{addon.__name__}.ops.external_axes")


@pytest.fixture
def ext(addon):
    return importlib.import_module(f"{addon.__name__}.rig.external_axes")


@pytest.fixture
def rig_model(addon):
    return importlib.import_module(f"{addon.__name__}.solver.rig_model")


@pytest.fixture(autouse=True)
def _free_compiled_solvers(addon, manager):
    yield
    import gc

    manager.invalidate()
    gc.collect()


def _np4(matrix) -> np.ndarray:
    return np.array([[matrix[r][c] for c in range(4)] for r in range(4)])


def _import(fixture_dir, builder, name: str):
    import bpy

    existing = {o.name for o in bpy.data.objects}
    assert "FINISHED" in bpy.ops.kinema.build_robot(
        filepath=str(fixture_dir / name), enforce_limits=True
    )
    rig = next(
        o for o in bpy.data.objects if builder.is_kinema_rig(o) and o.name not in existing
    )
    _activate(rig)
    return rig


def _activate(rig) -> None:
    import bpy

    for other in bpy.context.view_layer.objects:
        other.select_set(False)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig


def _add(ops, ext, rig, **fields) -> str:
    _activate(rig)
    return ops.add_axis(rig, ext.AxisSpec(**fields))


def _remove(ops, rig, name: str) -> None:
    _activate(rig)
    ops.remove_axis(rig, name)


def _pose(builder, rig, values: dict) -> None:
    import bpy

    for name, value in values.items():
        pose_bone = rig.pose.bones[name]
        if pose_bone.bone.get(builder.PROP_JOINT_TYPE) == "prismatic":
            pose_bone.location[1] = value
        else:
            pose_bone.rotation_euler[1] = value
    bpy.context.view_layer.update()


def _tcp(builder, rig) -> np.ndarray:
    import bpy

    bpy.context.view_layer.update()
    return _np4(rig.pose.bones[builder.TCP_BONE].matrix)


def _tcp_rest(builder, rig) -> np.ndarray:
    return _np4(rig.data.bones[builder.TCP_BONE].matrix_local)


def _joint_names(builder, rig) -> list[str]:
    return [pb.name for pb in builder.joint_bones(rig)]


@pytest.fixture
def arm6(addon, fixture_dir, clean_scene, builder):
    return _import(fixture_dir, builder, "arm6.urdf")


TRACK = dict(
    name="track", mount="BEFORE", kind="LINEAR", direction="X", lower=-1.5, upper=1.5,
    velocity=1.0, base_location=(0.0, 0.0, 0.04),
)
SPINDLE = dict(name="spindle", mount="AFTER", kind="ROTARY", direction="Z", continuous=True)
POSITIONER = dict(
    name="positioner", mount="EXTERNAL", kind="ROTARY", direction="Z", lower=-3.0,
    upper=3.0, base_location=(1.0, 0.0, 0.0),
)


class TestTrackUnderTheRobot:
    def test_it_is_the_rail_the_description_would_have_had(
        self, arm6, fixture_dir, builder, ops, ext
    ):
        """arm6 plus a track is rail6: same bones at rest, same tool at any pose."""
        _add(ops, ext, arm6, **TRACK)
        rail6 = _import(fixture_dir, builder, "rail6.urdf")
        names = {"track": "rail"}

        for pose_bone in builder.joint_bones(arm6):
            ours = _np4(pose_bone.bone.matrix_local)
            theirs = _np4(rail6.data.bones[names.get(pose_bone.name, pose_bone.name)].matrix_local)
            np.testing.assert_allclose(ours, theirs, atol=1e-6, err_msg=pose_bone.name)

        rng = np.random.default_rng(0)
        for _ in range(5):
            q = rng.uniform(-1.2, 1.2, 7)
            _pose(builder, arm6, dict(zip(_joint_names(builder, arm6), q, strict=True)))
            _pose(builder, rail6, dict(zip(_joint_names(builder, rail6), q, strict=True)))
            np.testing.assert_allclose(_tcp(builder, arm6), _tcp(builder, rail6), atol=1e-5)

    def test_it_is_a_joint_like_any_other(self, arm6, builder, ops, ext):
        _add(ops, ext, arm6, **TRACK)
        assert _joint_names(builder, arm6) == [
            "track", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"
        ]
        pose_bone = arm6.pose.bones["track"]
        bone = pose_bone.bone
        assert bone[builder.PROP_EXTERNAL] == "BEFORE"
        assert bone[builder.PROP_JOINT_TYPE] == "prismatic"
        assert (bone[builder.PROP_LOWER], bone[builder.PROP_UPPER]) == (-1.5, 1.5)
        assert bone[builder.PROP_VELOCITY] == 1.0
        assert pose_bone.kinema_ik_hold, "an external axis starts held"
        limit = pose_bone.constraints[builder.LIMIT_CONSTRAINT]
        assert (limit.min_y, limit.max_y) == pytest.approx((-1.5, 1.5))
        assert arm6.pose.bones["joint1"].parent.name == "track"
        assert any(c.name == builder.COLLECTION_FK for c in bone.collections)

    def test_a_lift_moves_everything_riding_and_removal_puts_it_back(
        self, arm6, builder, ops, ext
    ):
        import bpy

        arm6.kinema_solver_mode = "NUMPY"
        _pose(builder, arm6, {"joint2": -0.6, "joint3": 1.0, "joint5": 0.4})
        bpy.ops.kinema.add_ik()
        bpy.ops.kinema.add_waypoint(name="pick")
        arm6.kinema_ik_tip = 2  # joint3
        ik_name = arm6[builder.PROP_IK_BONE]
        waypoint = arm6.kinema_waypoints[0]
        before = {
            "tcp": _tcp_rest(builder, arm6),
            "ik": _np4(arm6.data.bones[ik_name].matrix_local),
            "marker": _np4(waypoint.marker.matrix_basis),
            "pose": np.array(waypoint.pose).reshape(4, 4),
            "q": list(waypoint.q)[: waypoint.dof],
            "joints": _joint_names(builder, arm6),
        }
        lift = np.eye(4)
        lift[2, 3] = 0.2

        _add(ops, ext, arm6, **{**TRACK, "base_location": (0.0, 0.0, 0.0),
                                "offset_location": (0.0, 0.0, 0.2)})
        np.testing.assert_allclose(_tcp_rest(builder, arm6), lift @ before["tcp"], atol=1e-6)
        np.testing.assert_allclose(
            _np4(arm6.data.bones[ik_name].matrix_local), lift @ before["ik"], atol=1e-6
        )
        np.testing.assert_allclose(
            _np4(waypoint.marker.matrix_basis), lift @ before["marker"], atol=1e-6
        )
        np.testing.assert_allclose(
            np.array(waypoint.pose).reshape(4, 4), lift @ before["pose"], atol=1e-6
        )
        assert waypoint.dof == 7
        assert list(waypoint.q)[:7] == pytest.approx([0.0, *before["q"]])
        assert arm6.kinema_ik_tip == 3, "the tip index follows joint3"
        np.testing.assert_allclose(ops.robot_base(arm6), lift, atol=1e-9)

        _remove(ops, arm6, "track")
        np.testing.assert_allclose(_tcp_rest(builder, arm6), before["tcp"], atol=1e-6)
        np.testing.assert_allclose(
            _np4(arm6.data.bones[ik_name].matrix_local), before["ik"], atol=1e-6
        )
        np.testing.assert_allclose(_np4(waypoint.marker.matrix_basis), before["marker"], atol=1e-6)
        np.testing.assert_allclose(
            np.array(waypoint.pose).reshape(4, 4), before["pose"], atol=1e-6
        )
        assert waypoint.dof == 6
        assert list(waypoint.q)[:6] == pytest.approx(before["q"])
        assert arm6.kinema_ik_tip == 2
        assert _joint_names(builder, arm6) == before["joints"]
        np.testing.assert_allclose(ops.robot_base(arm6), np.eye(4), atol=1e-9)

    def test_the_base_link_rides_it_and_stays_resettable(
        self, fixture_dir, clean_scene, builder, ops, ext
    ):
        """rail6's base mesh hangs off Root; under a new axis it rides that axis.

        Its recorded rest moves with it, so Reset Meshes keeps it on the axis
        rather than snapping it back to where it stood before.
        """
        import bpy

        rig = _import(fixture_dir, builder, "rail6.urdf")
        base = next(o for o in builder.link_meshes(rig) if o.parent_bone == builder.ROOT_BONE)
        before = _np4(base.matrix_world)
        _add(ops, ext, rig, name="riser", mount="BEFORE", kind="ROTARY", direction="Z",
             offset_location=(0.0, 0.0, 0.3))
        bpy.context.view_layer.update()
        assert base.parent_bone == "riser"
        lift = np.eye(4)
        lift[2, 3] = 0.3
        np.testing.assert_allclose(_np4(base.matrix_world), lift @ before, atol=1e-6)
        bpy.ops.kinema.reset_link_meshes()
        np.testing.assert_allclose(_np4(base.matrix_world), lift @ before, atol=1e-6)

        _pose(builder, rig, {"riser": 0.5})
        assert not np.allclose(_np4(base.matrix_world), lift @ before, atol=1e-3)

        _pose(builder, rig, {"riser": 0.0})
        _remove(ops, rig, "riser")
        bpy.context.view_layer.update()
        assert base.parent_bone == builder.ROOT_BONE
        np.testing.assert_allclose(_np4(base.matrix_world), before, atol=1e-6)

    def test_stacked_axes_come_off_in_either_order(
        self, arm6, fixture_dir, builder, ops, ext
    ):
        """Take the lower one off first, and the robot is as if it was never there.

        The IK target is made at the TCP, and must stay on it through every step:
        the second axis stands on the first, so what rides the robot is found from
        there, and the target hangs off Root rather than off either.
        """
        import bpy

        arm6.kinema_solver_mode = "NUMPY"
        bpy.ops.kinema.add_ik()
        ik_name = arm6[builder.PROP_IK_BONE]

        def target_on_tool():
            np.testing.assert_allclose(
                _np4(arm6.data.bones[ik_name].matrix_local), _tcp_rest(builder, arm6),
                atol=1e-5,
            )

        original = _tcp_rest(builder, arm6)
        _add(ops, ext, arm6, **{**TRACK, "base_location": (0.5, 0.0, 0.04),
                                "base_rotation": (0.0, 0.0, math.pi / 2)})
        target_on_tool()
        riser = dict(name="riser", mount="BEFORE", kind="ROTARY", direction="Z",
                     offset_location=(0.1, 0.0, 0.2), offset_rotation=(0.0, 0.0, 0.3))
        _add(ops, ext, arm6, **riser)
        assert _joint_names(builder, arm6)[:2] == ["track", "riser"]
        target_on_tool()

        _remove(ops, arm6, "track")
        target_on_tool()
        alone = _import(fixture_dir, builder, "arm6.urdf")
        _add(ops, ext, alone, **riser)
        np.testing.assert_allclose(_tcp_rest(builder, arm6), _tcp_rest(builder, alone), atol=1e-5)
        np.testing.assert_allclose(ops.robot_base(arm6), ops.robot_base(alone), atol=1e-6)

        _remove(ops, arm6, "riser")
        target_on_tool()
        np.testing.assert_allclose(_tcp_rest(builder, arm6), original, atol=1e-5)
        np.testing.assert_allclose(ops.robot_base(arm6), np.eye(4), atol=1e-6)

    def test_placeholders_ride_their_bones_and_leave_with_them(self, arm6, builder, ops, ext):
        import bpy

        _add(ops, ext, arm6, **TRACK)
        parts = {o.name.split()[-1]: o for o in ops.placeholders(arm6, "track")}
        assert set(parts) == {"base", "moving"}
        assert parts["base"].parent_bone == builder.ROOT_BONE
        assert parts["moving"].parent_bone == "track"
        rest = {name: _np4(obj.matrix_world) for name, obj in parts.items()}
        meshes = [obj.data.name for obj in parts.values()]

        _pose(builder, arm6, {"track": 0.5})
        np.testing.assert_allclose(_np4(parts["base"].matrix_world), rest["base"], atol=1e-6)
        moved = _np4(parts["moving"].matrix_world)[:3, 3] - rest["moving"][:3, 3]
        np.testing.assert_allclose(moved, [0.5, 0.0, 0.0], atol=1e-6)

        _remove(ops, arm6, "track")
        assert not ops.placeholders(arm6, "track")
        assert not any(name in bpy.data.meshes for name in meshes)


class TestJointIndices:
    def test_keyed_tips_and_axis_keys(self, arm6, builder, ops, ext):
        """A keyed tip keeps naming its joint; the axis's own keys leave with it."""
        ik = importlib.import_module(f"{ops.__package__}.ik")

        def curves(path: str) -> list:
            return [
                curve for container in ik.own_fcurve_containers(arm6) for curve in container
                if path in curve.data_path
            ]

        arm6.kinema_ik_tip = 2
        arm6.keyframe_insert(data_path="kinema_ik_tip", frame=1)
        _add(ops, ext, arm6, **TRACK)
        arm6.pose.bones["track"].keyframe_insert(data_path="location", index=1, frame=1)
        assert curves('pose.bones["track"]')
        assert [p.co[1] for p in curves("kinema_ik_tip")[0].keyframe_points] == [3.0]

        _remove(ops, arm6, "track")
        assert [p.co[1] for p in curves("kinema_ik_tip")[0].keyframe_points] == [2.0]
        assert not curves('pose.bones["track"]')

    def test_a_tip_on_the_removed_axis_falls_back_to_the_tcp(self, arm6, ops, ext):
        _add(ops, ext, arm6, **TRACK)
        arm6.kinema_ik_tip = 0  # the track itself
        _remove(ops, arm6, "track")
        assert arm6.kinema_ik_tip == -1


class TestSpindleOnTheTool:
    def test_the_tcp_rides_it_and_turns_about_its_own_z(self, arm6, builder, ops, ext):
        from ..conftest import load_addon_module

        chain = load_addon_module("solver.chain")
        before = _tcp_rest(builder, arm6)
        _add(ops, ext, arm6, **SPINDLE)
        assert arm6.data.bones[builder.TCP_BONE].parent.name == "spindle"
        np.testing.assert_allclose(_tcp_rest(builder, arm6), before, atol=1e-6)
        assert [b.name for b in chain.chain_bones(arm6, builder.TCP_BONE)][-1] == "spindle"

        tool = _np4(builder.BONE_TO_TOOL)
        start = _tcp(builder, arm6) @ tool
        _pose(builder, arm6, {"spindle": 0.7})
        turned = _tcp(builder, arm6) @ tool
        np.testing.assert_allclose(turned[:3, 3], start[:3, 3], atol=1e-6)
        spin = np.eye(4)
        spin[:2, :2] = [[math.cos(0.7), -math.sin(0.7)], [math.sin(0.7), math.cos(0.7)]]
        np.testing.assert_allclose(turned, start @ spin, atol=1e-5)

    def test_an_offset_lengthens_the_tool_and_removal_restores_it(
        self, arm6, builder, ops, ext
    ):
        before = _tcp_rest(builder, arm6)
        offset = tuple(arm6.kinema_tcp_offset), tuple(arm6.kinema_tcp_rpy)
        _add(ops, ext, arm6, **{**SPINDLE, "offset_location": (0.0, 0.0, 0.1)})
        tool = before @ _np4(builder.BONE_TO_TOOL)
        moved = _tcp_rest(builder, arm6)[:3, 3] - before[:3, 3]
        np.testing.assert_allclose(moved, 0.1 * tool[:3, 2], atol=1e-6)
        assert tuple(arm6.kinema_tcp_offset) == pytest.approx((0.0, 0.0, 0.1))

        _remove(ops, arm6, "spindle")
        assert arm6.data.bones[builder.TCP_BONE].parent.name == "joint6"
        np.testing.assert_allclose(_tcp_rest(builder, arm6), before, atol=1e-6)
        assert tuple(arm6.kinema_tcp_offset) == pytest.approx(offset[0], abs=1e-6)
        assert tuple(arm6.kinema_tcp_rpy) == pytest.approx(offset[1], abs=1e-6)

    def test_it_needs_a_tcp(self, fixture_dir, clean_scene, builder, ops, ext):
        import bpy

        rig = _import(fixture_dir, builder, "arm6.urdf")
        _activate(rig)
        bpy.ops.object.mode_set(mode="EDIT")
        rig.data.edit_bones.remove(rig.data.edit_bones[builder.TCP_BONE])
        bpy.ops.object.mode_set(mode="OBJECT")
        with pytest.raises(ValueError, match="TCP"):
            _add(ops, ext, rig, **SPINDLE)
        assert "spindle" not in rig.data.bones


class TestPositioner:
    def test_it_moves_nothing_of_the_robot(self, arm6, builder, ops, ext):
        _add(ops, ext, arm6, **POSITIONER)
        assert arm6.data.bones["positioner"].parent.name == builder.ROOT_BONE
        assert _joint_names(builder, arm6)[-1] == "positioner"
        head = np.array(arm6.data.bones["positioner"].head_local)
        np.testing.assert_allclose(head, [1.0, 0.0, 0.0], atol=1e-6)

        before = _tcp(builder, arm6)
        _pose(builder, arm6, {"positioner": 1.0})
        np.testing.assert_allclose(_tcp(builder, arm6), before, atol=1e-9)

    def test_what_is_attached_to_it_turns_with_it(self, arm6, builder, ops, ext):
        import bpy

        attach = importlib.import_module(f"{ops.__package__}.attach")
        _add(ops, ext, arm6, **POSITIONER)
        part = bpy.data.objects.new("part", None)
        bpy.context.scene.collection.objects.link(part)
        copy = attach.attach(arm6, "positioner", part)
        copy.location = (0.2, 0.0, 0.0)
        bpy.context.view_layer.update()
        start = np.array(copy.matrix_world.translation)
        _pose(builder, arm6, {"positioner": math.pi / 2})
        turned = np.array(copy.matrix_world.translation)
        # A quarter turn on a 0.2 m arm about the positioner's axis.
        assert np.linalg.norm(turned - start) == pytest.approx(0.2 * math.sqrt(2), abs=1e-5)

        # Removed, the attachment stays where it is on screen, no longer riding.
        _remove(ops, arm6, "positioner")
        bpy.context.view_layer.update()
        assert copy.parent is None
        np.testing.assert_allclose(np.array(copy.matrix_world.translation), turned, atol=1e-6)


class TestTheSolversModel:
    def _fk_matches(self, builder, rig, urdf, root: str, seed: int) -> None:
        """Every joint's link, rig against URDF, at random joint values."""
        rng = np.random.default_rng(seed)
        joints = builder.joint_bones(rig)
        for _ in range(4):
            q = rng.uniform(-1.0, 1.0, len(joints))
            _pose(builder, rig, dict(zip([pb.name for pb in joints], q, strict=True)))
            urdf.update_cfg({pb.name: float(v) for pb, v in zip(joints, q, strict=True)})
            for pose_bone in joints:
                bone = pose_bone.bone
                correction = np.array(bone[builder.PROP_LINK_CORRECTION]).reshape(4, 4)
                ours = _np4(pose_bone.matrix) @ correction
                theirs = urdf.get_transform(str(bone[builder.PROP_CHILD_LINK]), root)
                np.testing.assert_allclose(theirs, ours, atol=1e-5, err_msg=pose_bone.name)

    def test_with_axes_it_is_the_rig(self, arm6, builder, manager, ops, ext, rig_model):
        for spec in (TRACK, SPINDLE, POSITIONER):
            _add(ops, ext, arm6, **spec)
        urdf = manager._load_source_urdf(arm6)
        assert set(urdf.actuated_joint_names) == set(_joint_names(builder, arm6))
        self._fk_matches(builder, arm6, urdf, rig_model.WORLD_LINK, seed=1)

    def test_without_axes_it_agrees_with_the_description(
        self, fixture_dir, clean_scene, builder, manager, rig_model
    ):
        from ..conftest import load_addon_module

        bridge = load_addon_module("solver.urdf_bridge")
        for name in ("arm6.urdf", "rail6.urdf"):
            rig = _import(fixture_dir, builder, name)
            source = manager._load_source_urdf(rig)
            ours = bridge.urdf_from_model(rig_model.model_from_rig(rig))
            self._fk_matches(builder, rig, source, source.base_link, seed=2)
            self._fk_matches(builder, rig, ours, rig_model.WORLD_LINK, seed=2)

    def test_pyroki_drives_the_track_when_it_is_released(
        self, arm6, builder, handlers, manager, ops, ext
    ):
        """A goal past the arm's reach but within the track's is reached by the track."""
        import bpy
        from mathutils import Matrix

        _add(ops, ext, arm6, **TRACK)
        _pose(builder, arm6, {"joint2": -0.6, "joint3": 1.0, "joint5": 0.4})
        bpy.ops.kinema.add_ik()
        solver = manager.get_solver(arm6)
        assert solver.pyroki_error is None, solver.pyroki_error
        assert "track" in solver.chain.bone_names

        arm6.pose.bones["track"].kinema_ik_hold = False
        ik = arm6.pose.bones[arm6[builder.PROP_IK_BONE]]
        goal = _np4(ik.matrix)
        goal[0, 3] += 1.0
        ik.matrix = Matrix(goal.tolist())
        bpy.context.view_layer.update()
        for _ in range(4):
            handlers.solve_rig(arm6, force=True)
            bpy.context.view_layer.update()

        assert solver.pyroki_error is None, solver.pyroki_error
        reached = _tcp(builder, arm6)
        assert np.linalg.norm(reached[:3, 3] - goal[:3, 3]) < 2e-3
        assert arm6.pose.bones["track"].location[1] > 0.5


class TestTheOperators:
    def test_add_and_remove(self, arm6, builder):
        import bpy

        result = bpy.ops.kinema.add_external_axis(
            preset="CUSTOM", axis_name="slide", mount="AFTER", kind="LINEAR", direction="Z",
            lower_distance=0.0, upper_distance=0.1, speed_distance=0.2, size=0.05,
        )
        assert result == {"FINISHED"}
        bone = arm6.data.bones["slide"]
        assert (bone[builder.PROP_LOWER], bone[builder.PROP_UPPER]) == pytest.approx((0.0, 0.1))
        assert bone[builder.PROP_VELOCITY] == pytest.approx(0.2)
        assert bpy.ops.kinema.remove_external_axis(bone="slide") == {"FINISHED"}
        assert "slide" not in arm6.data.bones

    def test_a_bad_spec_is_refused_with_nothing_changed(self, arm6, builder):
        import bpy

        joints = _joint_names(builder, arm6)
        # Reported as an error, which a Python caller receives as an exception.
        with pytest.raises(RuntimeError, match="lower limit"):
            bpy.ops.kinema.add_external_axis(
                preset="CUSTOM", axis_name="track", kind="LINEAR",
                lower_distance=1.0, upper_distance=-1.0,
            )
        assert _joint_names(builder, arm6) == joints

    def test_pose_mode_is_where_it_leaves_you(self, arm6, ops, ext):
        """The buttons sit in the sidebar an animator poses from."""
        import bpy

        bpy.ops.object.mode_set(mode="POSE")
        try:
            _add(ops, ext, arm6, **SPINDLE)
            assert arm6.mode == "POSE"
            _remove(ops, arm6, "spindle")
            assert arm6.mode == "POSE"
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")

    def test_a_name_in_use_gets_a_suffix(self, arm6, ops, ext):
        assert _add(ops, ext, arm6, **{**POSITIONER, "name": "joint1"}) == "joint1.001"

    def test_the_dialog_and_panel_draw_what_exists(self, arm6, addon, ops, ext):
        """Every property drawn is real, every icon is Blender's, every button works.

        The dialog cannot be opened in the background, and a typo in a draw
        function only shows up as a broken panel -- so both are drawn into a layout
        that records what it was asked for, for every preset, with and without axes.
        """
        import types

        import bpy

        panel = importlib.import_module(f"{addon.__name__}.ui.panel")
        icons = set(
            bpy.types.UILayout.bl_rna.functions["label"].parameters["icon"].enum_items.keys()
        )
        log: list = []

        class Layout:
            use_property_split = enabled = alert = active = False
            alignment, scale_y = "", 1.0

            def row(self, **_):
                return Layout()

            column = box = row

            def separator(self, **_):
                pass

            def label(self, *, text="", icon="NONE"):
                log.append(("icon", icon))

            def prop(self, data, name, **options):
                log.append(("prop", data, name))
                if "icon" in options:
                    log.append(("icon", options["icon"]))

            def operator(self, idname, **options):
                log.append(("operator", idname))
                log.append(("icon", options.get("icon", "NONE")))
                return types.SimpleNamespace()

        dialog_class = ops.KINEMA_OT_add_external_axis
        properties = bpy.ops.kinema.add_external_axis.get_rna_type().properties
        values = {
            p.identifier: (tuple(p.default_array) if getattr(p, "is_array", False) else p.default)
            for p in properties if p.identifier != "rna_type"
        }
        dialog = types.SimpleNamespace(**values)
        dialog.spec = types.MethodType(dialog_class.spec, dialog)

        for preset in [*ext.PRESETS, "CUSTOM"]:
            dialog.preset = preset
            ops._apply_preset(dialog, bpy.context)
            for mount in ext.MOUNTS:
                dialog.mount = mount
                dialog.base_location = (0.0, 0.0, 0.3)  # so the robot-move note draws
                dialog.layout = Layout()
                dialog_class.draw(dialog, bpy.context)
        panel_class = panel.KINEMA_PT_external_axes
        panel_class.draw(types.SimpleNamespace(layout=Layout()), bpy.context)
        for spec in (TRACK, SPINDLE, POSITIONER):
            _add(ops, ext, arm6, **spec)
        panel_class.draw(types.SimpleNamespace(layout=Layout()), bpy.context)

        drawn = {entry[2] for entry in log if entry[0] == "prop" and entry[1] is dialog}
        assert drawn <= set(values), drawn - set(values)
        assert {"base_location", "offset_location", "speed_angle", "lower_distance"} <= drawn
        assert {entry[1] for entry in log if entry[0] == "icon"} <= icons
        for entry in (e for e in log if e[0] == "operator"):
            getattr(bpy.ops.kinema, entry[1].split(".")[1]).get_rna_type()

    def test_presets_fill_the_dialog(self, arm6, ops, ext):
        import bpy

        class Dialog:
            pass

        dialog = Dialog()
        dialog.preset = "TOOL_SPINDLE"
        ops._apply_preset(dialog, bpy.context)
        assert (dialog.mount, dialog.kind, dialog.continuous) == ("AFTER", "ROTARY", True)
        assert dialog.size == pytest.approx(ext.default_size("AFTER", ops.robot_reach(arm6)))
