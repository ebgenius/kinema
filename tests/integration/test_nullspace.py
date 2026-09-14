"""Redundancy an animator can grab: held external axes, and the elbow swivel.

Needs a real ``bpy``. The claims:

* a six-axis arm on a rail gets its rail held when IK is added, and nothing else,
* an arm with nothing to spare, or whose spare joint is an elbow, holds nothing,
* dragging a held rail moves it, the arm follows, and the tool stays put --
  every drag, not just the first,
* released, the rail is the solver's to move again,
* the NumPy fallback and the solution search both respect a hold,
* a swivel rides the shoulder-to-wrist line and is only offered when there is
  an elbow left to swing,
* adding one does not move the arm; turning it swings the elbow round that
  line while the tool stays put -- every turn, keyed or not, wherever the rig
  object stands in the scene.

Only the tests whose claim depends on PyRoki compile it. The rest run on the
NumPy fallback, because a JAX compile per rig is what made this file slow and,
on the development laptop, what ran it out of memory.
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
def handlers(addon):
    return importlib.import_module(f"{addon.__name__}.handlers")


@pytest.fixture
def manager(addon):
    return importlib.import_module(f"{addon.__name__}.solver.manager")


@pytest.fixture
def ik_ops(addon):
    return importlib.import_module(f"{addon.__name__}.ops.ik")


@pytest.fixture
def swivel(addon):
    return importlib.import_module(f"{addon.__name__}.solver.swivel")


@pytest.fixture
def panel(addon):
    return importlib.import_module(f"{addon.__name__}.ui.panel")


@pytest.fixture(autouse=True)
def _free_compiled_solvers(addon, manager):
    """Drop every compiled solver after each test.

    Each test builds its own rig, and each rig compiles its own JAX kernels --
    two once a swivel is added. The manager keeps the last four alive by
    design, which across a file of these was enough to exhaust memory on the
    development laptop. No two tests here share a rig, so nothing is lost.
    """
    yield
    import gc

    manager.invalidate()
    gc.collect()


def _rig_from(fixture_dir, builder, name: str, mode: str = "PYROKI"):
    """Build a rig and add IK, compiling PyRoki only when the test needs it.

    ``add_ik`` warms the solver up in whatever mode the rig is set to, so the
    mode has to be chosen before it runs, not after.
    """
    import bpy

    # Limits on explicitly: they are what a held joint dragged past its range
    # has to be solved against, and a default could change under the tests.
    assert "FINISHED" in bpy.ops.kinema.build_robot(
        filepath=str(fixture_dir / name), enforce_limits=True
    )
    rig = next(o for o in bpy.data.objects if builder.is_kinema_rig(o))
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    rig.kinema_solver_mode = mode
    bpy.ops.kinema.add_ik()
    return rig


@pytest.fixture
def rail6(addon, fixture_dir, clean_scene, builder):
    """arm6 on a linear rail: seven joints, six of them rotary. PyRoki."""
    return _rig_from(fixture_dir, builder, "rail6.urdf")


@pytest.fixture
def rail6_numpy(addon, fixture_dir, clean_scene, builder):
    return _rig_from(fixture_dir, builder, "rail6.urdf", mode="NUMPY")


@pytest.fixture
def arm7(addon, fixture_dir, clean_scene, builder):
    """Seven rotary joints: an elbow left to swing after the tool pose. PyRoki."""
    return _rig_from(fixture_dir, builder, "arm7.urdf")


@pytest.fixture
def arm7_numpy(addon, fixture_dir, clean_scene, builder):
    return _rig_from(fixture_dir, builder, "arm7.urdf", mode="NUMPY")


@pytest.fixture
def arm6_numpy(addon, fixture_dir, clean_scene, builder):
    return _rig_from(fixture_dir, builder, "arm6.urdf", mode="NUMPY")


#: rail, then arm6's six joints. A pose the arm reaches comfortably, so moving
#: the carriage a little leaves the tool still reachable.
REACHABLE = [0.0, 0.4, -0.7, 1.1, 0.2, 0.5, 0.0]
REDUNDANT = [0.3, -0.8, 0.5, 1.0, 0.2, 0.6, 0.0]


def _pose(builder, rig, q) -> None:
    import bpy

    for pose_bone, value in zip(builder.joint_bones(rig), q, strict=True):
        if pose_bone.bone.get(builder.PROP_JOINT_TYPE) == "prismatic":
            pose_bone.location[1] = value
        else:
            pose_bone.rotation_euler[1] = value
    bpy.context.view_layer.update()


def _settle(handlers, rig) -> None:
    """Already solved once and at rest, which is the state a user is in.

    ``solve_rig`` skips a rig whose goal has not changed, and with nothing
    recorded the *first* call always solves -- so a test that moves a control
    before any solve has happened sees the arm move whether or not the control
    was what moved it.
    """
    import bpy

    handlers.solve_rig(rig, force=True)
    bpy.context.view_layer.update()


def _start_at(builder, handlers, rig, q) -> None:
    """Pose with live IK off, aim the target at that pose, then go live.

    Posing with live IK on lets the handler drag the arm back toward wherever
    the target was before it is snapped, and the test then runs from a pose
    nobody chose.
    """
    import bpy

    rig.kinema_ik_enabled = False
    _pose(builder, rig, q)
    bpy.ops.kinema.snap_ik()
    rig.kinema_ik_enabled = True
    _settle(handlers, rig)


def _tool(builder, rig) -> np.ndarray:
    import bpy

    bpy.context.view_layer.update()
    name = rig.get(builder.PROP_TCP_BONE) or builder.TCP_BONE
    return np.array(rig.pose.bones[name].matrix.translation)


def _drag_rail(handlers, rig, distance: float) -> bool:
    """Slide the rail's own bone along its axis, as the viewport does.

    Solved explicitly rather than left to the depsgraph handler, whose time
    budget drops updates on a slow machine and would make this intermittent.
    """
    import bpy

    with handlers.suspended():
        rig.pose.bones["rail"].location[1] += distance
        bpy.context.view_layer.update()
    wrote = handlers.solve_rig(rig)
    bpy.context.view_layer.update()
    return wrote


class TestWhatGetsHeld:
    def test_adding_ik_holds_the_rail_and_nothing_else(self, rail6_numpy, builder):
        held = [pb.name for pb in builder.joint_bones(rail6_numpy) if pb.kinema_ik_hold]
        assert held == ["rail"]

    def test_an_arm_with_nothing_to_spare_holds_nothing(
        self, addon, fixture_dir, clean_scene, builder
    ):
        """arm3 has a slide, but three joints against a six-DoF pose."""
        rig = _rig_from(fixture_dir, builder, "arm3.urdf", mode="NUMPY")
        assert not any(pb.kinema_ik_hold for pb in builder.joint_bones(rig))

    def test_a_seven_axis_arm_holds_nothing(self, arm7_numpy, builder):
        """Its spare joint is rotary: an elbow, not an external axis."""
        assert not any(pb.kinema_ik_hold for pb in builder.joint_bones(arm7_numpy))

    def test_a_joint_already_held_by_hand_counts_as_spare(
        self, rail6_numpy, builder, ik_ops
    ):
        """A joint the user already held is a spare joint already taken out.

        Seven joints with one held by hand leaves six, which is exactly enough.
        Holding the rail as well would leave the arm five to reach a six-DoF pose.
        """
        rig = rail6_numpy
        pose = rig.pose.bones
        pose["rail"].kinema_ik_hold = False
        pose["joint1"].kinema_ik_hold = True

        assert ik_ops.hold_external_axes(rig) == []
        assert not pose["rail"].kinema_ik_hold
        free = [pb.name for pb in builder.joint_bones(rig) if not pb.kinema_ik_hold]
        assert len(free) == 6


class TestDraggingAHeldRail:
    def test_the_rail_moves_the_arm_follows_and_the_tool_stays(
        self, rail6, builder, handlers
    ):
        _start_at(builder, handlers, rail6, REACHABLE)
        tool_before = _tool(builder, rail6).copy()
        arm_before = np.array([pb.rotation_euler[1] for pb in builder.joint_bones(rail6)[1:]])

        assert _drag_rail(handlers, rail6, 0.2), "the drag never reached a solve"

        # Blender stores the channel as float32, so "exactly" means to that.
        assert rail6.pose.bones["rail"].location[1] == pytest.approx(0.2, abs=1e-6)
        arm_after = np.array([pb.rotation_euler[1] for pb in builder.joint_bones(rail6)[1:]])
        assert not np.allclose(arm_after, arm_before, atol=1e-3), "the arm did not follow"
        drift = float(np.linalg.norm(_tool(builder, rail6) - tool_before))
        assert drift < 1e-3, f"the tool gave way ({drift * 1000:.3f} mm)"

    def test_every_drag_re_solves_not_just_the_first(
        self, rail6_numpy, builder, handlers
    ):
        """A guard that forgets an input works once per session, then never.

        On NumPy: what is under test is the handler's comparison, which is the
        same whichever backend answers.
        """
        rig = rail6_numpy
        _start_at(builder, handlers, rig, REACHABLE)
        tool = _tool(builder, rig).copy()

        for step in range(3):
            assert _drag_rail(handlers, rig, 0.08), f"drag {step + 1} did nothing"
            drift = float(np.linalg.norm(_tool(builder, rig) - tool))
            assert drift < 2e-3, f"drag {step + 1}: the tool gave way ({drift * 1000:.3f} mm)"

    def test_the_numpy_fallback_holds_it_too(self, rail6_numpy, builder, handlers):
        rig = rail6_numpy
        _start_at(builder, handlers, rig, REACHABLE)
        tool = _tool(builder, rig).copy()

        assert _drag_rail(handlers, rig, 0.15)
        assert rig.pose.bones["rail"].location[1] == pytest.approx(0.15, abs=1e-6)
        assert float(np.linalg.norm(_tool(builder, rig) - tool)) < 2e-3

    def test_a_rail_dragged_past_its_limit_is_solved_where_it_stands(
        self, rail6, builder, handlers
    ):
        """The solver has to hold the rail where the rig shows it.

        The rail carries the import's limit constraint, so a carriage dragged to
        1.55 m against a 1.5 m limit reads 1.55 in its channel while it stands
        at 1.5. Planned around 1.55, the arm put the displayed tool 5 cm off its
        goal while the solve reported success.
        """
        assert builder.LIMIT_CONSTRAINT in [
            c.name for c in rail6.pose.bones["rail"].constraints
        ], "the rail has no limit constraint, so this proves nothing"

        start = list(REACHABLE)
        start[0] = 1.45
        _start_at(builder, handlers, rail6, start)
        goal = np.array(rail6.pose.bones[rail6.get(builder.PROP_IK_BONE)].matrix.translation)

        assert _drag_rail(handlers, rail6, 0.10)

        # The limit really clamps: the carriage stands at 1.5 however far its
        # channel went. Measured off the bone's evaluated position, not through
        # the code under test -- a muted or misplaced constraint would put the
        # carriage at 1.55 too, the display and the channel would agree, and a
        # solve seeded from the channel would pass without the test proving
        # anything.
        rail = rail6.pose.bones["rail"]
        along = np.array(rail.bone.matrix_local.col[1][:3])
        stands = float(
            np.dot(np.array(rail.matrix.translation) - np.array(rail.bone.head_local), along)
        )
        assert stands == pytest.approx(1.5, abs=1e-4), (
            f"the limit is not clamping: the carriage stands at {stands:.4f} m"
        )

        # The channel keeps what was dragged; the constraint shows the limit.
        assert rail.location[1] == pytest.approx(1.55, abs=1e-6)
        miss = float(np.linalg.norm(_tool(builder, rail6) - goal))
        assert miss < 1e-3, f"the displayed tool is {miss * 1000:.1f} mm off its goal"


class TestReleasing:
    def test_released_the_solver_may_move_the_rail(
        self, rail6_numpy, builder, handlers
    ):
        """The control for the tests above: a far target needs the carriage.

        Held, the rail stays and the tool falls short. Released, the solver
        slides the carriage to reach -- which is what proves the hold was doing
        something rather than the solver simply never touching the rail.
        """
        import bpy
        from mathutils import Vector

        rig = rail6_numpy
        _start_at(builder, handlers, rig, REACHABLE)
        ik = rig.pose.bones[rig.get(builder.PROP_IK_BONE)]

        ik.matrix.translation = ik.matrix.translation + Vector((0.9, 0.0, 0.0))
        bpy.context.view_layer.update()
        handlers.solve_rig(rig, force=True)
        bpy.context.view_layer.update()
        assert rig.pose.bones["rail"].location[1] == pytest.approx(0.0, abs=1e-6)

        rig.pose.bones["rail"].kinema_ik_hold = False
        handlers.solve_rig(rig, force=True)
        bpy.context.view_layer.update()
        assert rig.pose.bones["rail"].location[1] > 0.1, "released, the rail still did not move"


class TestTheSearchRespectsIt:
    def test_every_solution_keeps_the_rail_where_it_stands(
        self, rail6, builder, handlers, ik_ops
    ):
        """On PyRoki, which is where the search spreads the hold onto the full vector."""
        import bpy

        rail6.kinema_ik_enabled = False
        _pose(builder, rail6, REACHABLE)
        rail6.pose.bones["rail"].location[1] = 0.3
        bpy.context.view_layer.update()
        bpy.ops.kinema.snap_ik()

        assert "FINISHED" in bpy.ops.kinema.find_solutions(seeds=20)
        solutions = ik_ops._read_solutions(rail6)
        assert solutions, "the search found nothing, so nothing below was checked"
        for values in solutions:
            assert values[0] == pytest.approx(0.3, abs=1e-6), values


# --------------------------------------------------------------------------
# the swivel
# --------------------------------------------------------------------------
def _elbow_angle(swivel, rig) -> float:
    """The elbow's angle round the shoulder-to-wrist line, from world up.

    Measured off arm7's own joints -- joint2 is the shoulder centre and joint6
    the wrist centre, by its axes -- rather than through anything the swivel
    computes, so the test does not grade the code with the code's own maths.
    """
    import bpy

    bpy.context.view_layer.update()
    pose = rig.pose.bones
    shoulder = np.array(pose["joint2"].matrix.translation)
    wrist = np.array(pose["joint6"].matrix.translation)
    elbow = np.array(pose["joint4"].matrix.translation)
    axis = (wrist - shoulder) / np.linalg.norm(wrist - shoulder)
    centre = shoulder + np.dot(elbow - shoulder, axis) * axis
    return swivel.signed_angle(axis, np.array([0.0, 0.0, 1.0]), elbow - centre)


def _wrapped(angle: float) -> float:
    return float(np.angle(np.exp(1j * angle)))


def _tool_error(builder, rig) -> float:
    """How far the tool is from the goal it is being asked to reach, in metres."""
    goal = np.array(rig.pose.bones[rig.get(builder.PROP_IK_BONE)].matrix.translation)
    return float(np.linalg.norm(_tool(builder, rig) - goal))


def _move_target(handlers, builder, rig, delta) -> None:
    """Move the IK target as a drag does, and let one live update respond."""
    import bpy
    from mathutils import Vector

    ik = rig.pose.bones[rig.get(builder.PROP_IK_BONE)]
    with handlers.suspended():
        ik.matrix.translation = ik.matrix.translation + Vector(delta)
        bpy.context.view_layer.update()
    handlers.solve_rig(rig)
    bpy.context.view_layer.update()


def _turn(handlers, builder, rig, radians: float) -> bool:
    """Turn the swivel as the viewport's rotate tool does, and let IK respond."""
    import bpy

    handle = rig.pose.bones[rig.get(builder.PROP_SWIVEL_BONE)]
    with handlers.suspended():
        handle.rotation_euler[1] += radians
        bpy.context.view_layer.update()
    wrote = handlers.solve_rig(rig)
    bpy.context.view_layer.update()
    return wrote


def _with_swivel(builder, handlers, rig):
    import bpy

    _start_at(builder, handlers, rig, REDUNDANT)
    assert "FINISHED" in bpy.ops.kinema.add_swivel()
    _settle(handlers, rig)


class TestWhereTheSwivelIsOffered:
    def test_a_six_axis_arm_is_refused(self, arm6_numpy):
        import bpy

        with pytest.raises(RuntimeError, match="freedom left over"):
            bpy.ops.kinema.add_swivel()

    def test_a_rail_arm_is_refused_while_its_rail_is_held(self, rail6_numpy):
        """Seven joints, but the spare one is the rail: nothing left to swing."""
        import bpy

        with pytest.raises(RuntimeError, match="freedom left over"):
            bpy.ops.kinema.add_swivel()


class TestTheSwivelPanel:
    def test_holds_past_a_mid_chain_tip_are_not_subtracted(
        self, arm7_numpy, manager, panel
    ):
        """The free count and the chain length must describe the same joints.

        Aim the solver at joint6 and hold joint7, which is past the tip: the
        chain is six joints and none of them is held. Subtracting joint7 anyway
        says five, and a count that can come out short can hide a swivel.
        """
        rig = arm7_numpy
        rig.kinema_ik_tip = 5  # joint6
        solver = manager.get_solver(rig)
        assert solver.chain.dof == 6

        # Nothing held: the chain's six, not all seven joint bones.
        assert panel._free_joint_count(rig) == 6

        rig.pose.bones["joint7"].kinema_ik_hold = True
        assert panel._free_joint_count(rig) == 6

    def test_a_swivel_keeps_its_section_when_a_hold_leaves_nothing_to_swing(
        self, arm7_numpy, builder, handlers, panel
    ):
        """Its Remove button goes wherever the section goes.

        Holding a joint took arm7 to six free joints and the whole section --
        slider, explanation and Remove -- vanished, leaving a ring in the
        viewport that turned and did nothing. The pin is keyframable, so it
        would have blinked in and out over a shot.
        """
        rig = arm7_numpy
        _with_swivel(builder, handlers, rig)
        assert panel._swivel_section(rig) == "active"

        rig.pose.bones["joint1"].kinema_ik_hold = True
        assert panel._swivel_section(rig) == "stalled"

        rig.pose.bones["joint1"].kinema_ik_hold = False
        assert panel._swivel_section(rig) == "active"

    def test_nothing_is_offered_with_nothing_to_swing(self, arm7_numpy, panel):
        rig = arm7_numpy
        assert panel._swivel_section(rig) == "offer"
        rig.pose.bones["joint1"].kinema_ik_hold = True
        assert panel._swivel_section(rig) is None


class TestTheSwivel:
    def test_it_rides_the_shoulder_to_wrist_line(self, arm7_numpy, builder, handlers):
        """On the line when added, and still on it after the arm moves.

        Blender's constraints do this, not a solver, so NumPy is enough.
        """
        import bpy

        rig = arm7_numpy
        _with_swivel(builder, handlers, rig)

        for q in (REDUNDANT, [0.9, -0.4, -0.6, 1.4, -0.3, 0.2, 0.5]):
            rig.kinema_ik_enabled = False
            _pose(builder, rig, q)
            bpy.context.view_layer.update()
            pose = rig.pose.bones
            shoulder = np.array(pose["joint2"].matrix.translation)
            wrist = np.array(pose["joint6"].matrix.translation)
            line = (wrist - shoulder) / np.linalg.norm(wrist - shoulder)

            handle = pose[rig.get(builder.PROP_SWIVEL_BONE)]
            along = np.array(handle.matrix.col[1][:3])
            head = np.array(handle.matrix.translation)
            off_line = (head - shoulder) - np.dot(head - shoulder, line) * line

            assert abs(float(np.dot(along, line))) == pytest.approx(1.0, abs=1e-4)
            assert float(np.linalg.norm(off_line)) < 1e-4

    def test_adding_one_does_not_repose_the_arm(self, arm7, builder, handlers):
        import bpy

        _start_at(builder, handlers, arm7, REDUNDANT)
        before = np.array([pb.rotation_euler[1] for pb in builder.joint_bones(arm7)])

        assert "FINISHED" in bpy.ops.kinema.add_swivel()
        handlers.solve_rig(arm7, force=True)
        bpy.context.view_layer.update()

        after = np.array([pb.rotation_euler[1] for pb in builder.joint_bones(arm7)])
        moved = float(np.max(np.abs(np.degrees(after - before))))
        assert moved < 1.0, f"adding the swivel moved the arm {moved:.2f} degrees"

    def test_turning_it_swings_the_elbow_and_the_tool_stays(
        self, arm7, builder, handlers, swivel
    ):
        """The claim the control rests on.

        The goal sits on the circle the elbow can actually reach, so pulling
        toward it costs the tool nothing. The free-floating target this replaced
        cost 0.55 mm at its default strength; this has to do far better.
        """
        _with_swivel(builder, handlers, arm7)
        tool = _tool(builder, arm7).copy()
        start = _elbow_angle(swivel, arm7)

        assert _turn(handlers, builder, arm7, 0.5), "the turn never reached a solve"

        swung = _wrapped(_elbow_angle(swivel, arm7) - start)
        drift = float(np.linalg.norm(_tool(builder, arm7) - tool))
        assert abs(swung) == pytest.approx(0.5, abs=0.05), f"swung {swung:.3f} rad"
        assert drift < 1e-4, f"the tool gave way ({drift * 1000:.4f} mm)"

    def test_every_turn_re_solves_not_just_the_first(
        self, arm7, builder, handlers, swivel
    ):
        _with_swivel(builder, handlers, arm7)
        for step in range(3):
            before = _elbow_angle(swivel, arm7)
            assert _turn(handlers, builder, arm7, 0.3), f"turn {step + 1} did nothing"
            swung = abs(_wrapped(_elbow_angle(swivel, arm7) - before))
            assert swung > 0.2, f"turn {step + 1} swung only {swung:.3f} rad"

    def test_a_moved_target_is_reached_in_one_update(self, arm7, builder, handlers):
        """The circle is built about the wrist where the tool is going.

        One live update is all a drag gets: the handler only solves again when
        something changes. Built about where the wrist is now, the circle is
        wrong for the goal that update solves toward, and on arm7 a 10 cm move
        left the tool 0.39 mm short. Built about the goal, 0.0002 mm.
        """
        _with_swivel(builder, handlers, arm7)
        _turn(handlers, builder, arm7, 0.6)

        _move_target(handlers, builder, arm7, (0.10, 0.0, 0.0))
        error = _tool_error(builder, arm7)
        assert error < 5e-5, f"one update left the tool {error * 1000:.4f} mm off"

    def test_dragging_the_target_out_of_reach_does_not_flip_the_elbow(
        self, arm7, builder, handlers, swivel
    ):
        """Past reach the elbow stays on the side the swivel put it.

        A goal on the straight line has no side, and arm7 dragged just past its
        reach once came out with the elbow flipped 162 degrees.
        """
        _with_swivel(builder, handlers, arm7)
        _turn(handlers, builder, arm7, 0.6)
        before = _elbow_angle(swivel, arm7)

        _move_target(handlers, builder, arm7, (0.15, -0.10, 0.08))
        handlers.solve_rig(arm7, force=True)
        turned = abs(_wrapped(_elbow_angle(swivel, arm7) - before))
        assert turned < np.pi / 2, f"the elbow flipped {np.degrees(turned):.0f} degrees"

    def test_a_keyed_swivel_plays_back(self, arm7, builder, handlers, swivel):
        import bpy

        _with_swivel(builder, handlers, arm7)
        handle = arm7.pose.bones[arm7.get(builder.PROP_SWIVEL_BONE)]
        base = float(handle.rotation_euler[1])
        handle.keyframe_insert("rotation_euler", index=1, frame=1)
        handle.rotation_euler[1] = base + 0.6
        handle.keyframe_insert("rotation_euler", index=1, frame=10)

        scene = bpy.context.scene
        scene.frame_set(1)
        start = _elbow_angle(swivel, arm7)
        scene.frame_set(10)
        swung = _wrapped(_elbow_angle(swivel, arm7) - start)
        assert abs(swung) == pytest.approx(0.6, abs=0.06), f"swung {swung:.3f} rad"

    def test_a_rig_moved_off_the_origin_swings_the_same(
        self, arm7, builder, handlers, swivel
    ):
        """Everything is compared in armature space; the scene should not matter."""
        import bpy
        from mathutils import Euler

        arm7.location = (1.5, -0.7, 0.4)
        arm7.rotation_euler = Euler((0.3, -0.2, 1.1), "XYZ")
        arm7.scale = (1.3, 1.3, 1.3)
        bpy.context.view_layer.update()

        _with_swivel(builder, handlers, arm7)
        tool = _tool(builder, arm7).copy()
        start = _elbow_angle(swivel, arm7)

        assert _turn(handlers, builder, arm7, 0.5)
        swung = _wrapped(_elbow_angle(swivel, arm7) - start)
        assert abs(swung) == pytest.approx(0.5, abs=0.05), f"swung {swung:.3f} rad"
        assert float(np.linalg.norm(_tool(builder, arm7) - tool)) < 1e-4

    def test_zero_strength_stops_it_swinging(self, arm7, builder, handlers, swivel):
        """On PyRoki: under NumPy the swivel does nothing anyway, so this would pass."""
        _with_swivel(builder, handlers, arm7)
        arm7.kinema_elbow_strength = 0.0
        _settle(handlers, arm7)
        before = _elbow_angle(swivel, arm7)

        _turn(handlers, builder, arm7, 0.5)
        assert abs(_wrapped(_elbow_angle(swivel, arm7) - before)) < 0.02

    def test_the_strength_slider_forgets_the_cached_problem(
        self, arm7_numpy, builder, handlers
    ):
        """What the slider's update callback does, and only that.

        Checked on the cache itself rather than through ``solve_rig``: the
        handler's comparison also includes the strength, so a re-solve would
        happen with or without the callback and could not tell the two apart.
        """
        rig = arm7_numpy
        _with_swivel(builder, handlers, rig)
        assert rig.name in handlers._last_target, "nothing was cached to forget"

        rig.kinema_elbow_strength = 8.0
        assert rig.name not in handlers._last_target, "the slider left the old problem cached"

    def test_a_strength_change_without_the_callback_still_re_solves(
        self, arm7_numpy, builder, handlers, monkeypatch
    ):
        """The handler's comparison includes the strength in its own right.

        Setting the property fires an update callback that forgets the cached
        problem, so a test that sets it that way passes whether or not the
        comparison looks at the strength. A keyed or driven strength changes
        without that callback, so the comparison has to catch it alone.

        The callback's reset is stubbed out for the one assignment, leaving the
        comparison as the only thing that can notice. (Writing
        ``rig["kinema_elbow_strength"]`` does not do it: in Blender 5.2 that
        reaches a separate custom property, and the registered one keeps its
        value.)
        """
        rig = arm7_numpy
        _with_swivel(builder, handlers, rig)
        assert rig.name in handlers._last_target

        monkeypatch.setattr(handlers, "reset", lambda *args, **kwargs: None)
        rig.kinema_elbow_strength = 8.0
        monkeypatch.undo()
        assert rig.kinema_elbow_strength == pytest.approx(8.0)
        assert rig.name in handlers._last_target, "the cache was forgotten, so this proves nothing"

        assert handlers.solve_rig(rig), "the comparison ignored the new strength"

    def test_removing_it_leaves_plain_ik(self, arm7_numpy, builder, handlers):
        import bpy

        rig = arm7_numpy
        _with_swivel(builder, handlers, rig)
        assert "FINISHED" in bpy.ops.kinema.remove_swivel()
        for key in (builder.PROP_SWIVEL_BONE, builder.PROP_SWIVEL_AXIS):
            assert not rig.get(key)
        for name in (builder.SWIVEL_BONE, builder.SWIVEL_AXIS_BONE):
            assert name not in rig.pose.bones
