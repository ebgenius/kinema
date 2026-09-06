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
