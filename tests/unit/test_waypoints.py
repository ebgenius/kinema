"""Span construction: the part of waypoints that needs no Blender."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ..conftest import load_addon_module

waypoints = load_addon_module("rig.waypoints")


@dataclass
class Point:
    """Stands in for the property group, which needs bpy."""

    name: str
    frame: int
    move: str = waypoints.MOVE_JOINT


class TestOrdering:
    def test_the_timeline_decides_the_order(self):
        """Not the list index: dragging a key in the dope sheet reorders the job."""
        taught = [Point("drop", 60), Point("home", 1), Point("pick", 30)]
        assert [p.name for p in waypoints.ordered(taught)] == ["home", "pick", "drop"]

    def test_ties_keep_the_order_they_were_taught(self):
        """Stable sort, so nothing reshuffles for reasons the user cannot see."""
        taught = [Point("a", 5), Point("b", 5), Point("c", 5)]
        assert [p.name for p in waypoints.ordered(taught)] == ["a", "b", "c"]

    def test_frame_range_spans_first_to_last(self):
        taught = [Point("drop", 60), Point("home", 1), Point("pick", 30)]
        assert waypoints.frame_range(taught) == (1, 60)

    def test_frame_range_needs_a_waypoint(self):
        with pytest.raises(waypoints.WaypointError, match="no waypoints"):
            waypoints.frame_range([])


class TestSpans:
    def test_a_span_arrives_at_its_end_waypoint(self):
        """The move belongs to the waypoint being arrived at.

        MoveL(pick) says how to get *to* pick, which is how robot programs
        read, and it is why the first waypoint needs no move type.
        """
        taught = [
            Point("home", 1, waypoints.MOVE_JOINT),
            Point("approach", 20, waypoints.MOVE_JOINT),
            Point("pick", 40, waypoints.MOVE_LINEAR),
        ]
        got = waypoints.spans(taught)

        assert [(s.start.name, s.end.name, s.move) for s in got] == [
            ("home", "approach", waypoints.MOVE_JOINT),
            ("approach", "pick", waypoints.MOVE_LINEAR),
        ]

    def test_spans_carry_their_frames(self):
        taught = [Point("home", 1), Point("pick", 25)]
        span = waypoints.spans(taught)[0]
        assert (span.start_frame, span.end_frame, span.frames) == (1, 25, 24)

    def test_spans_come_out_in_time_order(self):
        taught = [Point("pick", 40), Point("home", 1), Point("approach", 20)]
        assert [s.end.name for s in waypoints.spans(taught)] == ["approach", "pick"]

    def test_one_waypoint_is_not_a_motion(self):
        with pytest.raises(waypoints.WaypointError, match="At least two"):
            waypoints.spans([Point("home", 1)])

    def test_no_waypoints_is_not_a_motion(self):
        with pytest.raises(waypoints.WaypointError, match="At least two"):
            waypoints.spans([])

    def test_two_waypoints_on_one_frame_are_refused(self):
        """Being in two places at once is a mistake, not something to guess at."""
        taught = [Point("home", 1), Point("pick", 20), Point("drop", 20)]
        with pytest.raises(waypoints.WaypointError, match="both on frame 20"):
            waypoints.spans(taught)

    def test_an_unknown_move_type_is_refused(self):
        taught = [Point("home", 1), Point("pick", 20, "TELEPORT")]
        with pytest.raises(waypoints.WaypointError, match="TELEPORT"):
            waypoints.spans(taught)


@dataclass
class Pose:
    """A waypoint as motions see it: a uid and a name."""

    uid: str
    name: str


@dataclass
class Move:
    """Stands in for a motion's property group."""

    name: str
    start: str
    end: str
    frame_start: int
    frame_end: int
    move: str = waypoints.MOVE_JOINT


POSES = [Pose("h", "home"), Pose("a", "approach"), Pose("p", "pick")]


def _laid(motions):
    return [
        (s.label, s.start.name, s.end.name, s.start_frame, s.end_frame, s.made)
        for s in waypoints.plan(motions, POSES).spans
    ]


class TestPlan:
    def test_motions_run_in_frame_order_not_list_order(self):
        motions = [
            Move("down", "a", "p", 20, 40, waypoints.MOVE_LINEAR),
            Move("in", "h", "a", 1, 20),
        ]
        got = waypoints.plan(motions, POSES).spans
        assert [(s.label, s.move) for s in got] == [
            ("in", waypoints.MOVE_JOINT),
            ("down", waypoints.MOVE_LINEAR),
        ]

    def test_one_waypoint_starts_one_motion_and_ends_another(self):
        """Out to pick and back home, with home taught once."""
        motions = [Move("out", "h", "p", 1, 30), Move("back", "p", "h", 30, 60)]
        assert _laid(motions) == [
            ("out", "home", "pick", 1, 30, waypoints.MADE_MOTION),
            ("back", "pick", "home", 30, 60, waypoints.MADE_MOTION),
        ]

    def test_a_gap_at_the_same_waypoint_is_a_hold(self):
        motions = [Move("out", "h", "p", 1, 30), Move("back", "p", "h", 45, 70)]
        assert _laid(motions)[1] == ("hold at pick", "pick", "pick", 30, 45, waypoints.MADE_HOLD)
        assert not waypoints.plan(motions, POSES).notes

    def test_a_gap_to_another_waypoint_gets_a_joint_move_and_a_note(self):
        motions = [Move("out", "h", "a", 1, 30), Move("down", "p", "h", 45, 70)]
        result = waypoints.plan(motions, POSES)
        assert _laid(motions)[1] == ("to pick", "approach", "pick", 30, 45, waypoints.MADE_BRIDGE)
        assert result.spans[1].move == waypoints.MOVE_JOINT
        assert len(result.notes) == 1 and "frames 30-45" in result.notes[0]

    def test_motions_that_overlap_are_refused(self):
        motions = [Move("out", "h", "p", 1, 30), Move("back", "p", "h", 25, 60)]
        with pytest.raises(waypoints.MotionsOverlap, match="'out'.*'back'.*overlap"):
            waypoints.plan(motions, POSES)

    def test_a_jump_with_no_frames_to_make_it_in_is_refused(self):
        motions = [Move("out", "h", "a", 1, 30), Move("down", "p", "h", 30, 60)]
        with pytest.raises(waypoints.WaypointError, match="leave frames between"):
            waypoints.plan(motions, POSES)

    def test_a_motion_has_to_end_after_it_starts(self):
        with pytest.raises(waypoints.WaypointError, match="not after it starts"):
            waypoints.plan([Move("out", "h", "p", 30, 30)], POSES)

    def test_a_motion_whose_waypoint_is_gone_is_refused(self):
        with pytest.raises(waypoints.WaypointError, match="no end waypoint"):
            waypoints.plan([Move("out", "h", "gone", 1, 30)], POSES)

    def test_no_motions_is_no_job(self):
        with pytest.raises(waypoints.WaypointError, match="no motions"):
            waypoints.plan([], POSES)

    def test_a_motion_can_stay_at_one_waypoint(self):
        """A pause is a motion from a waypoint to itself."""
        assert _laid([Move("wait", "h", "h", 1, 10)]) == [
            ("wait", "home", "home", 1, 10, waypoints.MADE_MOTION)
        ]
