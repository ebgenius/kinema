"""Checking a generated job, move by move: the part that needs no Blender."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from ..conftest import load_addon_module

check = load_addon_module("rig.motion_check")

FPS = 24.0


def _pose(position=(0.0, 0.0, 0.0), turn=0.0):
    """A pose at ``position``, turned ``turn`` radians about Z."""
    pose = np.eye(4)
    c, s = math.cos(turn), math.sin(turn)
    pose[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    pose[:3, 3] = position
    return pose


def _samples(frames, q, tools):
    return [
        check.Sample(frame=f, q=np.array(values, dtype=float), tool=tool)
        for f, values, tool in zip(frames, q, tools, strict=True)
    ]


def _still(frames, tool=None):
    """Samples of a joint that never moves, with the tool parked."""
    tool = _pose() if tool is None else tool
    return _samples(frames, [[0.0]] * len(frames), [tool] * len(frames))


JOINT = check.Joint("j1", velocity=2.0)


class TestRotations:
    def test_a_quaternion_and_its_negation_are_the_same_orientation(self):
        q = check.quaternion(_pose(turn=2.0)[:3, :3])
        assert check.angle_between(q, -q) == pytest.approx(0.0, abs=1e-7)

    def test_slerp_takes_the_short_way(self):
        """Across the hemisphere boundary, which is where component-wise goes wrong."""
        a = check.quaternion(_pose(turn=math.radians(170))[:3, :3])
        b = -check.quaternion(_pose(turn=math.radians(-170))[:3, :3])
        halfway = check.slerp(a, b, 0.5)
        assert math.degrees(check.angle_between(a, halfway)) == pytest.approx(10.0, abs=1e-6)

    def test_the_angle_is_the_rotation_between(self):
        a = check.quaternion(_pose(turn=0.3)[:3, :3])
        b = check.quaternion(_pose(turn=1.0)[:3, :3])
        assert check.angle_between(a, b) == pytest.approx(0.7)


class TestLinearMoves:
    def test_a_tool_on_the_line_is_not_flagged(self):
        start, end = _pose((0, 0, 0)), _pose((0.2, 0, 0))
        frames = range(1, 11)
        tools = [_pose((0.2 * (f - 1) / 9, 0, 0)) for f in frames]
        result = check.check_move(
            "cut", "LINEAR", 1, 10, start, end,
            _samples(frames, [[0.0]] * 10, tools), [JOINT], FPS,
        )
        assert result.line_error == pytest.approx(0.0, abs=1e-12)
        assert result.twist_error == pytest.approx(0.0, abs=1e-6)
        assert check.problems(result) == []

    def test_distance_is_measured_from_the_segment(self):
        """From the path, not from where the tool should be at that frame."""
        start, end = _pose((0, 0, 0)), _pose((0.2, 0, 0))
        # On the line but a frame behind, and then 3 mm off it.
        tools = [_pose((0.0, 0, 0)), _pose((0.05, 0.003, 0)), _pose((0.2, 0, 0))]
        result = check.check_move(
            "cut", "LINEAR", 1, 3, start, end,
            _samples([1, 2, 3], [[0.0]] * 3, tools), [JOINT], FPS,
        )
        assert result.line_error == pytest.approx(0.003)
        assert "3.0 mm off the line" in check.problems(result)

    def test_a_turn_is_compared_with_the_short_way_round(self):
        start, end = _pose(turn=math.radians(170)), _pose(turn=math.radians(-170))
        # Halfway through a 20 degree turn, the tool should be at 180.
        tools = [start, _pose(turn=math.radians(180)), end]
        good = check.check_move(
            "b", "LINEAR", 1, 3, start, end, _samples([1, 2, 3], [[0.0]] * 3, tools),
            [JOINT], FPS,
        )
        assert good.twist_error == pytest.approx(0.0, abs=1e-6)

        # The long way round passes through 0 instead.
        tools[1] = _pose(turn=0.0)
        bad = check.check_move(
            "b", "LINEAR", 1, 3, start, end, _samples([1, 2, 3], [[0.0]] * 3, tools),
            [JOINT], FPS,
        )
        assert math.degrees(bad.twist_error) == pytest.approx(180.0, abs=1e-4)
        assert any("off" in problem for problem in check.problems(bad))

    def test_a_joint_move_has_no_line_to_keep(self):
        result = check.check_move(
            "pick", "JOINT", 1, 3, _pose(), _pose((1, 0, 0)), _still([1, 2, 3]),
            [JOINT], FPS,
        )
        assert result.line_error < 0.0 and result.twist_error < 0.0
        assert check.problems(result) == []


class TestJoints:
    def test_only_the_moves_own_frames_are_read(self):
        """A job's samples cover every move; each move reads its own."""
        samples = _samples(
            [1, 2, 3, 4], [[0.0], [0.0], [0.0], [5.0]], [_pose()] * 4
        )
        result = check.check_move(
            "a", "JOINT", 1, 3, _pose(), _pose(), samples, [JOINT], FPS
        )
        assert result.jump == 0.0 and result.speed_ratio == 0.0

    def test_speed_is_measured_against_the_joints_limit(self):
        # 0.1 rad a frame at 24 fps is 2.4 rad/s, against a 2 rad/s limit.
        samples = _samples([1, 2, 3], [[0.0], [0.1], [0.2]], [_pose()] * 3)
        result = check.check_move(
            "a", "JOINT", 1, 3, _pose(), _pose(), samples, [JOINT], FPS
        )
        assert result.speed_joint == "j1"
        assert result.speed_ratio == pytest.approx(1.2)
        assert "j1 at 120% of its speed limit" in check.problems(result)

    def test_a_joint_without_a_limit_is_never_over_it(self):
        samples = _samples([1, 2], [[0.0], [0.5]], [_pose()] * 2)
        result = check.check_move(
            "a", "JOINT", 1, 2, _pose(), _pose(), samples,
            [check.Joint("free")], FPS,
        )
        assert result.speed_ratio == 0.0
        assert not any("speed" in problem for problem in check.problems(result))

    def test_a_jump_is_named_with_its_frame(self):
        """A configuration flip: most of a half-turn between two frames."""
        samples = _samples(
            [10, 11, 12], [[0.0], [0.02], [3.0]], [_pose()] * 3
        )
        result = check.check_move(
            "a", "LINEAR", 10, 12, _pose(), _pose(), samples,
            [check.Joint("j6")], FPS,
        )
        assert (result.jump_joint, result.jump_frame) == ("j6", 12)
        assert any("j6 jumps" in problem and "frame 12" in problem
                   for problem in check.problems(result))

    def test_an_ordinary_step_is_not_a_jump(self):
        samples = _samples([1, 2], [[0.0], [0.3]], [_pose()] * 2)
        result = check.check_move(
            "a", "JOINT", 1, 2, _pose(), _pose(), samples, [check.Joint("j")], FPS
        )
        assert result.jump == pytest.approx(0.3)
        assert not any("jumps" in problem for problem in check.problems(result))

    def test_a_prismatic_jump_is_judged_in_metres(self):
        samples = _samples([1, 2], [[0.0], [0.5]], [_pose()] * 2)
        result = check.check_move(
            "a", "JOINT", 1, 2, _pose(), _pose(), samples,
            [check.Joint("rail", prismatic=True)], FPS,
        )
        assert any("rail jumps 500 mm" in problem for problem in check.problems(result))

    def test_a_prismatic_jump_is_not_hidden_by_a_smaller_revolute_one(self):
        """Radians and metres do not compare; each is judged against its own threshold.

        In one frame the wrist steps 0.9 rad, under its threshold, and the
        rail jumps 0.3 m, over its own. Picked by raw size the wrist won and
        the rail's jump went unreported.
        """
        samples = _samples([1, 2], [[0.0, 0.0], [0.9, 0.3]], [_pose()] * 2)
        result = check.check_move(
            "a", "JOINT", 1, 2, _pose(), _pose(), samples,
            [check.Joint("wrist"), check.Joint("rail", prismatic=True)], FPS,
        )
        assert (result.jump_joint, result.jump_prismatic) == ("rail", True)
        assert any("rail jumps 300 mm" in problem for problem in check.problems(result))

    def test_a_joint_on_its_limit_is_flagged(self):
        joint = check.Joint("j5", lower=-2.0, upper=2.0)
        samples = _samples([1, 2, 3], [[1.5], [1.9], [2.0]], [_pose()] * 3)
        result = check.check_move(
            "a", "JOINT", 1, 3, _pose(), _pose(), samples, [joint], FPS
        )
        assert result.limit_joint == "j5"
        assert "j5 at its limit" in check.problems(result)

    def test_a_joint_near_but_not_on_its_limit_is_not(self):
        joint = check.Joint("j5", lower=-2.0, upper=2.0)
        samples = _samples([1, 2], [[1.5], [1.99]], [_pose()] * 2)
        result = check.check_move(
            "a", "JOINT", 1, 2, _pose(), _pose(), samples, [joint], FPS
        )
        assert result.limit_joint == ""


class TestAcceleration:
    JOINT = check.Joint("j1", velocity=100.0, acceleration=10.0)

    @staticmethod
    def _kinked():
        """Still to frame 11, then 0.05 rad a frame: the speed changes at once at 11."""
        frames = list(range(1, 22))
        q = [[0.0 if frame <= 11 else 0.05 * (frame - 11)] for frame in frames]
        return _samples(frames, q, [_pose()] * len(frames))

    def test_a_sudden_change_of_speed_is_flagged(self):
        # From still to 0.05 rad a frame: 0.05 times 24 squared is 28.8 rad/s²,
        # against a 10 rad/s² limit.
        result = check.check_move(
            "b", "JOINT", 11, 21, _pose(), _pose(), self._kinked(), [self.JOINT], FPS
        )
        assert result.accel_joint == "j1"
        assert result.accel_ratio == pytest.approx(2.88)
        assert "j1 at 288% of its acceleration limit" in check.problems(result)

    def test_both_moves_meeting_there_report_it(self):
        """The first move ends still; the frame after it is the next move's."""
        samples = self._kinked()
        into = check.check_move(
            "a", "JOINT", 1, 11, _pose(), _pose(), samples, [self.JOINT], FPS
        )
        out_of = check.check_move(
            "b", "JOINT", 11, 21, _pose(), _pose(), samples, [self.JOINT], FPS
        )
        assert into.accel_ratio == pytest.approx(2.88)
        assert out_of.accel_ratio == pytest.approx(2.88)

    def test_a_steady_speed_is_not_accelerating(self):
        frames = list(range(1, 11))
        samples = _samples(frames, [[0.05 * frame] for frame in frames], [_pose()] * 10)
        result = check.check_move(
            "a", "JOINT", 1, 10, _pose(), _pose(), samples, [self.JOINT], FPS
        )
        assert result.accel_ratio == pytest.approx(0.0, abs=1e-9)
        assert not any("acceleration" in problem for problem in check.problems(result))

    def test_a_joint_without_a_limit_is_never_over_it(self):
        result = check.check_move(
            "b", "JOINT", 11, 21, _pose(), _pose(), self._kinked(),
            [check.Joint("free", velocity=100.0)], FPS,
        )
        assert (result.accel_ratio, result.accel_joint) == (0.0, "")


class TestProblems:
    def test_reads_anything_shaped_like_a_check(self):
        """The copy stored on the rig is a Blender property group, not the dataclass."""

        class Stored:
            line_error = 0.0005
            twist_error = -1.0
            limit_joint = ""
            speed_joint = ""
            speed_ratio = 0.0
            jump_joint = ""
            jump = 0.0
            jump_prismatic = False
            jump_frame = 0

        assert check.problems(Stored()) == ["0.5 mm off the line"]

    def test_a_kept_turn_is_reported_first(self):
        """Set by Generate Motion rather than measured, and the one to act on."""
        result = check.MoveCheck("b", "LINEAR", 1, 21, line_error=0.0005)
        result.turn_note = "joint6 at +73°, taught at -287°: kept as the line arrives"
        assert check.problems(result) == [
            "joint6 at +73°, taught at -287°: kept as the line arrives",
            "0.5 mm off the line",
        ]


geometry = load_addon_module("rig.collision")


def _crossing(frames=(1, 2, 3), between=True, end=3):
    """A capsule of radius 20 mm moving 100 mm a frame along x, over a 10 mm plate
    at x = 150 mm: 50 mm short of it on frame 2, 50 mm past on frame 3.

    ``between`` measures it half-way from frame 2 to 3 too, through the plate.
    """
    capsule = geometry.Capsule("forearm", np.zeros(3), np.array([0.0, 0.0, 0.1]), 0.02)
    plate = geometry.Obstacle(
        "Plate", geometry.KIND_BOX, _pose((0.15, 0.0, 0.05)), np.array([0.005, 0.2, 0.2])
    )

    def at(x, subframe=0.0, frame=0):
        ends = np.array([[[x, 0.0, 0.0], [x, 0.0, 0.1]]])
        return check.Sample(
            frame=frame, q=np.zeros(1), tool=_pose(), segments=ends,
            radii=np.array([0.02]), obstacles=(plate,), subframe=subframe,
        )

    samples = []
    for frame in frames:
        x = 0.1 * (frame - 1)
        sample = at(x, frame=frame)
        if between and frame == 2:
            sample = replace(sample, between=(at(x + 0.05, 0.5, frame),))
        samples.append(sample)
    return check.check_move(
        "B", "JOINT", frames[0], end, _pose(), _pose(), samples, [JOINT], FPS, (capsule,)
    )


class TestBetweenFrames:
    def test_a_capsule_that_moves_less_than_its_radius_is_measured_on_frames_only(self):
        before = np.zeros((1, 2, 3))
        after = before + [0.015, 0.0, 0.0]
        assert check.between_frames(before, after, np.array([0.02])) == 1

    def test_it_is_measured_where_it_has_moved_no_further_than_its_radius(self):
        """100 mm in a frame at 20 mm: five parts, 20 mm each."""
        before = np.zeros((2, 2, 3))
        after = before.copy()
        after[1, 1] = [0.1, 0.0, 0.0]
        assert check.between_frames(before, after, np.array([0.05, 0.02])) == 5

    def test_a_capsule_that_thins_is_measured_by_its_thinner_end(self):
        """100 mm in a frame while the robot's scale takes its radius from 50 to 20 mm."""
        before = np.zeros((1, 2, 3))
        after = before + [0.1, 0.0, 0.0]
        radii = np.array([0.05])
        assert check.between_frames(before, after, radii) == 2
        assert check.between_frames(before, after, radii, radii_after=np.array([0.02])) == 5

    def test_however_fast_at_most_so_many_parts(self):
        before = np.zeros((1, 2, 3))
        after = before + [10.0, 0.0, 0.0]
        assert check.between_frames(before, after, np.array([0.02])) == check.MOST_BETWEEN

    def test_an_obstacle_moving_past_a_still_capsule_is_measured_in_parts_too(self):
        """A plate 100 mm along x in a frame, past a capsule of 20 mm that stays put."""
        ends = np.zeros((1, 2, 3))
        ends[0, 1, 2] = 0.1
        plate = [
            geometry.Obstacle("Plate", geometry.KIND_BOX, _pose((x, 0.0, 0.0)), np.full(3, 0.005))
            for x in (0.05, -0.05)
        ]
        radii = np.array([0.02])
        assert check.between_frames(ends, ends, radii) == 1
        assert check.between_frames(ends, ends, radii, (plate[0],), (plate[1],)) == 5

    def test_an_obstacle_turning_about_a_capsule_moves_against_it(self):
        """Turned a quarter about Z, its frame carries a capsule end 100 mm out 141 mm."""
        ends = np.zeros((1, 2, 3))
        ends[0, :, 0] = 0.1
        box = [
            geometry.Obstacle("Box", geometry.KIND_BOX, _pose(turn=turn), np.full(3, 0.005))
            for turn in (0.0, math.pi / 2)
        ]
        assert check.between_frames(ends, ends, np.array([0.02]), (box[0],), (box[1],)) == 8

    def test_an_obstacle_that_grows_moves_its_faces(self):
        ends = np.zeros((1, 2, 3))
        box = [
            geometry.Obstacle("Box", geometry.KIND_BOX, _pose((0.3, 0.0, 0.0)), np.full(3, size))
            for size in (0.05, 0.15)
        ]
        assert check.between_frames(ends, ends, np.array([0.02]), (box[0],), (box[1],)) == 5

    def test_a_plate_crossed_between_two_frames_is_reported_after_the_first(self):
        found = _crossing()
        assert found.clearance_obstacle == "Plate" and found.clearance < 0.0
        assert (found.clearance_frame, found.clearance_between) == (2, True)
        assert "forearm hits Plate by 25 mm between frames 2 and 3" in check.problems(found)

    def test_measured_on_frames_only_it_isn_t_seen(self):
        found = _crossing(between=False)
        assert found.clearance == pytest.approx(0.025)
        assert not found.clearance_between

    def test_after_a_move_s_last_frame_it_is_the_next_move_s(self):
        found = _crossing(end=2)
        assert found.clearance > 0.0 and not found.clearance_between

    def test_on_a_frame_it_says_so(self):
        found = check.MoveCheck("B", "JOINT", 1, 3, clearance_frame=2)
        assert check.clearance_when(found) == "at frame 2"
