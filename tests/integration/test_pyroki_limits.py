"""PyRoki solves with the limits the rig holds. Needs a real ``bpy``.

The claims worth holding onto:

* an unlimited continuous joint is unlimited to PyRoki too, so a spindle wound
  past half a turn keeps its turns and the arm stays where it is;
* PyRoki's top speeds are the bones', on a rig solved from a bridged URDF too;
* Load Joint Limits reaches a compiled solver without compiling it again.
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
def manager(addon):
    return importlib.import_module(f"{addon.__name__}.solver.manager")


@pytest.fixture
def handlers(addon):
    return importlib.import_module(f"{addon.__name__}.handlers")


@pytest.fixture(autouse=True)
def _free_compiled_solvers(addon, manager):
    """Each test here compiles PyRoki. Drop the kernels, and the rigs' cache
    entries with them, so the next module starts with none of them."""
    yield
    import gc

    manager.invalidate()
    gc.collect()


@pytest.fixture
def arm6(addon, fixture_dir, clean_scene, builder):
    """arm6 in a working pose, with an IK target and PyRoki as its solver."""
    import bpy

    assert "FINISHED" in bpy.ops.kinema.build_robot(
        filepath=str(fixture_dir / "arm6.urdf")
    )
    rig = next(o for o in bpy.data.objects if builder.is_kinema_rig(o))
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    for name, value in {"joint2": -0.6, "joint3": 1.0, "joint5": 0.4}.items():
        rig.pose.bones[name].rotation_euler[1] = value
    bpy.context.view_layer.update()
    bpy.ops.kinema.add_ik()
    rig.kinema_ik_enabled = False
    rig.kinema_solver_mode = "PYROKI"
    return rig


def _speeds(solver) -> dict[str, float]:
    """PyRoki's top speed for each actuated joint, by name."""
    values = np.asarray(solver.robot.joints.velocity_limits).tolist()
    return dict(zip(solver.actuated_names, values, strict=True))


class TestContinuousJoints:
    @pytest.mark.parametrize("wound", [3.6, 7.0])
    def test_an_unheld_spindle_past_half_a_turn_keeps_its_turns(
        self, arm6, builder, manager, handlers, wound
    ):
        """PyRoki held a continuous joint to [-pi, pi].

        Past half a turn the spindle was pulled back, and the wrist turned to
        make up for it: 0.87 rad at 3.6. A turn and more, 7.0, failed
        outright, 46 mm off.
        """
        import bpy

        assert "FINISHED" in bpy.ops.kinema.add_external_axis(
            preset="CUSTOM", axis_name="spindle", mount="AFTER", kind="ROTARY",
            direction="Z", continuous=True, hold=False, placeholder=False,
        )
        solver = manager.get_solver(arm6)
        assert "spindle" in solver.chain.bone_names, "the spindle is not on the IK chain"

        spindle = arm6.pose.bones["spindle"]
        wrist = arm6.pose.bones["joint6"]
        tip = arm6.pose.bones[solver.tip_bone]
        with handlers.suspended():
            spindle.rotation_euler[1] = wound
            bpy.context.view_layer.update()
            # The target where the tool is: the right answer is not to move.
            arm6.pose.bones[solver.ik_bone].matrix = tip.matrix.copy()
            bpy.context.view_layer.update()
            tool = np.array(tip.matrix)
            before = wrist.rotation_euler[1]
            result = solver.solve(arm6, manager.MODE_PYROKI)
            bpy.context.view_layer.update()

        assert result.backend == "PyRoki", "PyRoki did not answer, so this proves nothing"
        assert spindle.rotation_euler[1] == pytest.approx(wound, abs=1e-3)
        assert wrist.rotation_euler[1] == pytest.approx(before, abs=1e-3)
        assert np.linalg.norm(np.array(tip.matrix)[:3, 3] - tool[:3, 3]) < 1e-4


class TestSpeeds:
    def test_a_bridged_rig_solves_with_the_bones_speeds(self, arm6, manager):
        """The URDF written for a rig with external axes said 3.14 for every joint.

        arm6's own URDF says 3.15 and 3.2, and the track was given 0.8 m/s.
        """
        import bpy

        assert "FINISHED" in bpy.ops.kinema.add_external_axis(
            preset="CUSTOM", axis_name="track", mount="BEFORE", kind="LINEAR",
            direction="X", lower_distance=-1.0, upper_distance=1.0,
            speed_distance=0.8, placeholder=False,
        )
        speeds = _speeds(manager.get_solver(arm6).pyroki(arm6))

        assert speeds["track"] == pytest.approx(0.8)
        assert speeds["joint1"] == pytest.approx(3.15)
        assert speeds["joint6"] == pytest.approx(3.2)

    def test_load_joint_limits_reaches_a_compiled_solver_without_compiling(
        self, arm6, manager, tmp_path
    ):
        """The same arrays, new values: the compiled solve takes them as they are."""
        import bpy

        solver = manager.get_solver(arm6)
        assert solver.solve(arm6, manager.MODE_PYROKI).backend == "PyRoki"
        compiled = solver.pyroki(arm6)
        kernels = [entry[0] for entry in compiled._compiled.values()]
        if not all(hasattr(kernel, "_cache_size") for kernel in kernels):
            pytest.skip("this JAX does not report its compile cache")
        sizes = [kernel._cache_size() for kernel in kernels]

        path = tmp_path / "joint_limits.yaml"
        path.write_text(
            "joint_limits:\n  joint1:\n    has_velocity_limits: true\n"
            "    max_velocity: 0.5\n",
            encoding="utf-8",
        )
        assert "FINISHED" in bpy.ops.kinema.load_joint_limits(filepath=str(path))

        # The same solver, not a rebuilt one: a rebuild compiles afresh, and
        # would leave the old kernels' caches untouched for the check below.
        assert solver.pyroki(arm6) is compiled, "the solver was rebuilt"
        assert _speeds(compiled)["joint1"] == pytest.approx(0.5)
        assert solver.solve(arm6, manager.MODE_PYROKI).backend == "PyRoki"
        assert [kernel._cache_size() for kernel in kernels] == sizes, (
            "the new limits cost a compile"
        )
