"""Mimic joints that move (#104). Needs a real ``bpy``.

The claims:

* a mimic joint gets a bone, which a driver moves from the actuated joint it
  follows -- a parallel gripper closes from one slider, a chain of two mimics
  folds to one, a turning mimic turns -- and its link's mesh goes with it;
* the bone is no control: locked, out of the joints the user, IK and the API
  see, and the Tool bone stays on the arm's tool0;
* the rig writes it back to PyRoki as a <mimic>;
* the Motion Check measures it against its own limits.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from ..conftest import requires_bpy

pytestmark = requires_bpy


@pytest.fixture
def builder(addon):
    return importlib.import_module(f"{addon.__name__}.rig.builder")


@pytest.fixture
def gripper(addon, fixture_dir, clean_scene):
    import bpy

    assert "FINISHED" in bpy.ops.kinema.build_robot(
        filepath=str(fixture_dir / "gripper_mimic.urdf")
    )
    return bpy.context.view_layer.objects.active


def _set(rig, **values) -> None:
    """Set joint channels by name, then let the drivers run."""
    import bpy

    for name, value in values.items():
        pose_bone = rig.pose.bones[name]
        prismatic = pose_bone.bone.get("kinema_joint_type") == "prismatic"
        channel = pose_bone.location if prismatic else pose_bone.rotation_euler
        channel[1] = value
    bpy.context.view_layer.update()


def _value(rig, name) -> float:
    pose_bone = rig.pose.bones[name]
    prismatic = pose_bone.bone.get("kinema_joint_type") == "prismatic"
    return float((pose_bone.location if prismatic else pose_bone.rotation_euler)[1])


class TestDriven:
    def test_both_fingers_close_from_one_slider(self, gripper):
        _set(gripper, grip=0.03)
        assert _value(gripper, "grip_mirror") == pytest.approx(-0.03)

    def test_a_chain_of_two_mimics(self, gripper):
        """grip_tip follows grip_mirror, which follows grip: 2 * -grip + 0.01."""
        _set(gripper, grip=0.03)
        assert _value(gripper, "grip_tip") == pytest.approx(-0.05)
        _set(gripper, grip=0.0)
        assert _value(gripper, "grip_tip") == pytest.approx(0.01)

    def test_a_turning_mimic(self, gripper):
        _set(gripper, lift=0.4)
        assert _value(gripper, "knuckle") == pytest.approx(0.5 * 0.4 + 0.1)

    def test_the_fingers_mesh_moves_with_it(self, gripper):
        import bpy

        mesh = next(
            obj for obj in bpy.data.objects
            if obj.type == "MESH" and obj.parent is gripper and obj.parent_bone == "grip_mirror"
        )
        _set(gripper, grip=0.0)
        before = np.array(mesh.matrix_world.translation)
        _set(gripper, grip=0.03)
        after = np.array(mesh.matrix_world.translation)
        np.testing.assert_allclose(after - before, [-0.03, 0.0, 0.0], atol=1e-6)

    def test_with_the_driver_running_where_scripts_may_not(self, gripper):
        """A simple expression: Blender evaluates it without Python."""
        drivers = {
            fcurve.data_path.split('"')[1]: fcurve.driver
            for fcurve in gripper.animation_data.drivers
        }
        assert set(drivers) == {"grip_mirror", "grip_tip", "knuckle"}
        assert all(driver.is_simple_expression for driver in drivers.values())
        assert all(driver.is_valid for driver in drivers.values())


class TestNotAControl:
    def test_locked_and_out_of_the_joints(self, gripper, builder):
        assert [b.name for b in builder.joint_bones(gripper)] == ["lift", "grip"]
        mimics = {b.name: b for b in builder.mimic_bones(gripper)}
        assert set(mimics) == {"grip_mirror", "grip_tip", "knuckle"}
        for pose_bone in mimics.values():
            assert all(pose_bone.lock_location) and all(pose_bone.lock_rotation)
            assert builder.PROP_JOINT_NAME not in pose_bone.bone
        # As its <mimic> says: following another mimic, not folded onto grip.
        tip = mimics["grip_tip"].bone
        assert tip[builder.PROP_MIMIC_OF] == "grip_mirror"
        assert tip[builder.PROP_MIMIC_MULTIPLIER] == pytest.approx(2.0)
        assert tip[builder.PROP_MIMIC_OFFSET] == pytest.approx(0.01)

    def test_in_the_mechanism_collection(self, gripper, builder):
        collection = gripper.data.collections_all[builder.COLLECTION_MECHANISM]
        assert {"grip_mirror", "grip_tip", "knuckle"} <= {b.name for b in collection.bones}

    def test_the_api_lists_only_the_joints_set_by_hand(self, gripper, addon):
        api = importlib.import_module(f"{addon.__name__}.api")
        assert [joint.name for joint in api.joints(gripper)] == ["lift", "grip"]

    def test_the_tool_stays_on_the_arms_tool0(self, gripper, builder):
        tool = gripper.pose.bones[builder.tool_bone_name(gripper)]
        assert tool.parent.name == "lift"
        np.testing.assert_allclose(np.array(tool.bone.head_local), [0.0, 0.0, 0.4], atol=1e-6)


class TestWrittenBack:
    def test_the_rigs_model_has_the_mimics_again(self, gripper, addon):
        rig_model = importlib.import_module(f"{addon.__name__}.solver.rig_model")
        bridge = importlib.import_module(f"{addon.__name__}.solver.urdf_bridge")
        model = rig_model.model_from_rig(gripper)
        joints = {j.name: j for j in model.joints}
        assert joints["grip_mirror"].mimic_joint == "grip"
        assert joints["grip_mirror"].mimic_multiplier == pytest.approx(-1.0)
        assert joints["grip_tip"].mimic_joint == "grip"
        assert joints["grip_tip"].mimic_multiplier == pytest.approx(-2.0)
        assert joints["grip_tip"].mimic_offset == pytest.approx(0.01)
        assert [j.name for j in model.actuated_joints] == ["lift", "grip"]
        assert bridge.urdf_xml(model).count("<mimic ") == 3
        # As PyRoki is handed it.
        urdf = bridge.urdf_from_model(model)
        assert urdf.joint_map["grip_tip"].mimic.joint == "grip"
        assert urdf.joint_map["grip_tip"].mimic.multiplier == pytest.approx(-2.0)


class TestChecked:
    def test_the_motion_check_measures_mimics_against_their_own_limits(self, gripper, addon):
        waypoints = importlib.import_module(f"{addon.__name__}.ops.waypoints")
        joints = {joint.name: joint for joint in waypoints.check_joints(gripper)}
        assert {"grip_mirror", "grip_tip", "knuckle"} <= set(joints)
        assert (joints["grip_mirror"].lower, joints["grip_mirror"].upper) == pytest.approx(
            (-0.04, 0.0)
        )
        assert joints["grip_mirror"].velocity == pytest.approx(0.5)


class TestUnderAnAxis:
    def test_a_mimic_on_the_base_rides_a_new_axis_under_the_robot(
        self, addon, fixture_dir, clean_scene
    ):
        """continuous_and_mimic's mirrored finger hangs off the base, so off Root:
        a track put in under the robot carries it with the joints it stood beside."""
        import bpy

        ops = importlib.import_module(f"{addon.__name__}.ops.external_axes")
        ext = importlib.import_module(f"{addon.__name__}.rig.external_axes")
        assert "FINISHED" in bpy.ops.kinema.build_robot(
            filepath=str(fixture_dir / "continuous_and_mimic.urdf")
        )
        rig = bpy.context.view_layer.objects.active
        assert rig.data.bones["grip_mirror"].parent.name == "Root", "precondition"
        axis = ops.add_axis(
            rig, ext.AxisSpec(name="track", mount="BEFORE", kind="LINEAR", direction="X")
        )
        assert rig.data.bones["grip_mirror"].parent.name == axis
        assert rig.data.bones["grip"].parent.name == axis


class TestFollowsWhereTheLeaderStands:
    def test_a_leader_dragged_past_its_limit_leads_from_the_limit(self, gripper):
        """lift is limited to 1.5: dragged to 2.0 it stands at 1.5, and knuckle
        follows that, 0.5 * 1.5 + 0.1, not 0.5 * 2.0 + 0.1 (clamped to 1.0)."""
        _set(gripper, lift=2.0)
        assert _value(gripper, "knuckle") == pytest.approx(0.85)
        _set(gripper, lift=0.4)
        assert _value(gripper, "knuckle") == pytest.approx(0.3)


class TestUnresolved:
    def test_a_mimic_of_a_joint_that_isnt_there_is_said_at_import(self, addon, clean_scene):
        """Its sibling now moves, so its standing still has to be explained."""
        import numpy as np

        builder = importlib.import_module(f"{addon.__name__}.rig.builder")
        kin = importlib.import_module(f"{addon.__name__}.rig.kinematics")
        joint = kin.JointSpec(
            name="finger", joint_type="prismatic", parent_link="base", child_link="tip",
            origin=np.eye(4), axis=np.array([1.0, 0.0, 0.0]), mimic_joint="typo",
        )
        model = kin.RobotModel(
            name="lost", links={n: kin.LinkSpec(n) for n in ("base", "tip")},
            joints=[joint], root_link="base",
        )
        result = builder.build_rig(model, builder.RigBuildOptions())
        assert any("finger" in w and "'typo'" in w for w in result.warnings), result.warnings


class TestNumpyChain:
    """A mimic between actuated joints: the palletizer's parallelogram. NumPy's
    chain moves it by its leader, so its FK is the rig's and its IK lands."""

    @pytest.fixture
    def palletizer(self, addon, fixture_dir, clean_scene):
        import bpy

        assert "FINISHED" in bpy.ops.kinema.build_robot(
            filepath=str(fixture_dir / "palletizer_mimic.urdf")
        )
        rig = bpy.context.view_layer.objects.active
        rig.kinema_solver_mode = "NUMPY"
        return rig

    def test_its_forward_kinematics_are_the_rigs(self, palletizer, addon, builder):
        chain_mod = importlib.import_module(f"{addon.__name__}.solver.chain")
        tool = builder.tool_bone_name(palletizer)
        chain = chain_mod.chain_from_rig(palletizer, tool)
        assert chain.bone_names == ["swing", "shoulder", "reach", "twist"]
        assert [m.leader for m in chain.mimics] == [1], "level, following shoulder"
        for q in ([0.0, 0.0, 0.0, 0.0], [0.5, 0.7, 0.2, -1.0], [-1.2, -0.9, 0.4, 2.0]):
            _set(palletizer, swing=q[0], shoulder=q[1], reach=q[2], twist=q[3])
            shown = np.array(palletizer.pose.bones[tool].matrix)
            np.testing.assert_allclose(chain.forward(np.array(q)), shown, atol=1e-5)

    def test_numpy_ik_reaches_a_pose_through_the_parallelogram(self, palletizer, builder):
        import bpy

        tool = builder.tool_bone_name(palletizer)
        _set(palletizer, swing=0.4, shoulder=0.6, reach=0.3, twist=0.5)
        goal = palletizer.pose.bones[tool].matrix.copy()
        _set(palletizer, swing=0.0, shoulder=0.0, reach=0.1, twist=0.0)
        assert "FINISHED" in bpy.ops.kinema.add_ik()
        ik = palletizer.pose.bones[palletizer.get(builder.PROP_IK_BONE)]
        ik.matrix = goal
        bpy.context.view_layer.update()
        handlers = importlib.import_module(
            f"{builder.__name__.rsplit('.', 2)[0]}.handlers"
        )
        handlers.solve_rig(palletizer, force=True)
        bpy.context.view_layer.update()
        reached = np.array(palletizer.pose.bones[tool].matrix)
        np.testing.assert_allclose(reached[:3, 3], np.array(goal)[:3, 3], atol=1e-4)


class TestLimitsAndNames:
    def test_load_joint_limits_reaches_a_mimic(self, gripper, builder, tmp_path):
        import bpy

        path = tmp_path / "joint_limits.yaml"
        path.write_text(
            "joint_limits:\n"
            "  grip_mirror:\n"
            "    has_velocity_limits: true\n"
            "    max_velocity: 0.25\n"
            "    has_acceleration_limits: true\n"
            "    max_acceleration: 2.0\n",
            encoding="utf-8",
        )
        assert "FINISHED" in bpy.ops.kinema.load_joint_limits(filepath=str(path))
        bone = gripper.data.bones["grip_mirror"]
        assert bone[builder.PROP_VELOCITY] == pytest.approx(0.25)
        assert bone[builder.PROP_ACCELERATION] == pytest.approx(2.0)

    def test_an_axis_named_as_a_renamed_leader_was_doesnt_take_its_place(
        self, gripper, addon
    ):
        """The grip bone renamed, then an axis on the tool called grip: both record
        the joint name grip, and the mimic still follows the robot's own."""
        ops = importlib.import_module(f"{addon.__name__}.ops.external_axes")
        ext = importlib.import_module(f"{addon.__name__}.rig.external_axes")
        rig_model = importlib.import_module(f"{addon.__name__}.solver.rig_model")
        gripper.data.bones["grip"].name = "grip_control"
        axis = ops.add_axis(
            gripper, ext.AxisSpec(name="grip", mount="AFTER", kind="ROTARY", direction="Z")
        )
        assert gripper.data.bones[axis].get("kinema_joint") == "grip", "precondition"
        joints = {j.name: j for j in rig_model.model_from_rig(gripper).joints}
        assert joints["grip_mirror"].mimic_joint == "grip_control"


def _built(addon, fixture_dir, name, **options):
    """``name`` built straight from its model, with ``options``: the result's
    warnings, and the rig."""
    import yourdfpy

    builder = importlib.import_module(f"{addon.__name__}.rig.builder")
    kin = importlib.import_module(f"{addon.__name__}.rig.kinematics")
    model = kin.model_from_urdf(yourdfpy.URDF.load(str(fixture_dir / name)))
    result = builder.build_rig(model, builder.RigBuildOptions(**options))
    return result, result.armature_object


class TestLimitsAsTheRigHasThem:
    """NumPy's chain holds a mimic, and an off-chain leader, to their limits only
    where the rig does: imported with Enforce Joint Limits off, it doesn't."""

    @pytest.mark.parametrize("enforce", [True, False])
    def test_a_chain_aimed_at_a_fingertip_stands_where_the_rig_does(
        self, addon, fixture_dir, clean_scene, enforce
    ):
        import bpy

        chain_mod = importlib.import_module(f"{addon.__name__}.solver.chain")
        _, rig = _built(addon, fixture_dir, "gripper_mimic.urdf", enforce_limits=enforce)
        _set(rig, grip=0.06)  # past grip's 0.04
        chain = chain_mod.chain_from_rig(rig, "grip_tip")
        assert chain.bone_names == ["lift"]
        mirror = chain.mimics[0]
        assert mirror.leader == -1, "grip is off this chain"
        assert mirror.fixed == pytest.approx(-0.04 if enforce else -0.06)
        bpy.context.view_layer.update()
        np.testing.assert_allclose(
            chain.forward(np.zeros(1)), np.array(rig.pose.bones["grip_tip"].matrix), atol=1e-5
        )

    def test_a_cached_chain_follows_a_leader_moved_since(self, gripper, builder, addon):
        """The solver caches its chain; a fetch stands its held mimics where the
        rig shows them now."""
        import bpy

        manager = importlib.import_module(f"{addon.__name__}.solver.manager")
        chain_mod = importlib.import_module(f"{addon.__name__}.solver.chain")
        assert "FINISHED" in bpy.ops.kinema.add_ik()
        solver = manager.get_solver(gripper)
        # A tip is a joint bone or the tool, and none here stands on a mimic: the
        # cached solver is handed a fingertip's chain, as one on a finger would be.
        solver.chain = chain_mod.chain_from_rig(gripper, "grip_tip")
        _set(gripper, grip=0.03)
        assert solver.chain.mimics[0].fixed == pytest.approx(0.0, abs=1e-6), "built before the move"
        assert manager.get_solver(gripper) is solver, "the cached one"
        assert solver.chain.mimics[0].fixed == pytest.approx(-0.03)
        np.testing.assert_allclose(
            solver.chain.forward(np.zeros(1)),
            np.array(gripper.pose.bones["grip_tip"].matrix),
            atol=1e-5,
        )


class TestRestOutsideItsLimits:
    def test_a_mimic_that_rests_outside_its_limits_is_said_at_import(
        self, addon, fixture_dir, clean_scene, tmp_path
    ):
        """knuckle = 0.5 * lift + 0.1 rests at 0.1; limited to [-0.05, 0.05] it is
        held off its rest pose, which is said, as for a joint the user moves."""
        text = (fixture_dir / "gripper_mimic.urdf").read_text(encoding="utf-8")
        text = text.replace(
            '<limit lower="-1" upper="1" effort="5" velocity="1"/>\n'
            '    <mimic joint="lift"',
            '<limit lower="-0.05" upper="0.05" effort="5" velocity="1"/>\n'
            '    <mimic joint="lift"',
        )
        assert text.count('lower="-0.05"') == 1, "precondition: knuckle's limits narrowed"
        (tmp_path / "narrow.urdf").write_text(text, encoding="utf-8")
        result, _ = _built(addon, tmp_path, "narrow.urdf")
        said = [w for w in result.warnings if w.startswith("knuckle:")]
        assert said and "0.1" in said[0], result.warnings
        result, _ = _built(addon, fixture_dir, "gripper_mimic.urdf")
        assert not [w for w in result.warnings if w.startswith("knuckle:")]


class TestBlendersOwnLimits:
    """Read what Blender shows, rather than a copy of how it gets there."""

    def test_a_limit_partway_down_a_chain_of_mimics_holds(
        self, addon, fixture_dir, clean_scene, tmp_path
    ):
        """grip_mirror narrowed to [-0.02, 0]: at grip 0.03 it is held at -0.02, and
        grip_tip, 2 * grip_mirror + 0.01, follows it to -0.03, not to -0.05."""
        import bpy

        text = (fixture_dir / "gripper_mimic.urdf").read_text(encoding="utf-8")
        text = text.replace(
            '<limit lower="-0.04" upper="0" effort="5" velocity="0.5"/>',
            '<limit lower="-0.02" upper="0" effort="5" velocity="0.5"/>',
        )
        assert text.count('lower="-0.02"') == 1, "precondition: grip_mirror narrowed"
        (tmp_path / "narrow.urdf").write_text(text, encoding="utf-8")
        _, rig = _built(addon, tmp_path, "narrow.urdf")
        _set(rig, grip=0.03)
        chain_mod = importlib.import_module(f"{addon.__name__}.solver.chain")
        shown = chain_mod.displayed_value(rig.pose.bones["grip_tip"], revolute=False)
        assert chain_mod.displayed_value(rig.pose.bones["grip_mirror"], False) == pytest.approx(-0.02)
        assert shown == pytest.approx(-0.03)

    def test_a_constraint_taken_off_after_import_lets_the_mimic_follow(self, gripper):
        """grip's limit removed: dragged to 0.06, it stands there, and so does its
        mirror -- not at the -0.04 the import's limit would have held it to."""
        constraint = gripper.pose.bones["grip"].constraints.get("Kinema Joint Limit")
        gripper.pose.bones["grip"].constraints.remove(constraint)
        _set(gripper, grip=0.06)
        assert _value(gripper, "grip_mirror") == pytest.approx(-0.06)

    @pytest.mark.parametrize(
        "lower, upper, angle",
        [(-1.0, 1.0, 2.5), (-1.0, 1.0, -3.0), (-1.0, 1.0, 2 * np.pi + 0.2),
         (0.5, 1.5, 0.0), (0.5, 1.5, -2.6), (-1.5, 1.5, 0.3)],
    )
    def test_the_chains_clamp_is_blenders_limit_rotation(
        self, gripper, addon, lower, upper, angle
    ):
        """The NumPy chain's clamp_angle, against Blender's Limit Rotation on lift."""
        import bpy

        chain_mod = importlib.import_module(f"{addon.__name__}.solver.chain")
        bone = gripper.pose.bones["lift"]
        constraint = bone.constraints["Kinema Joint Limit"]
        constraint.min_y, constraint.max_y = lower, upper
        bone.rotation_euler[1] = angle
        bpy.context.view_layer.update()
        shown = chain_mod.displayed_value(bone, revolute=True)
        held = chain_mod.clamp_angle(angle, lower, upper)
        assert np.cos(shown) == pytest.approx(np.cos(held), abs=1e-5)
        assert np.sin(shown) == pytest.approx(np.sin(held), abs=1e-5)
