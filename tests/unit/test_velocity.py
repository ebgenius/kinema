"""Joint speeds and accelerations against their limits: the part that needs no Blender."""

from __future__ import annotations

import math

import pytest

from ..conftest import load_addon_module

velocity = load_addon_module("rig.velocity")


class TestReading:
    def test_ratio_is_speed_over_limit(self):
        assert velocity.Reading("j", 2.0, 3.0).ratio == pytest.approx(1.5)

    def test_over_only_past_the_limit(self):
        assert velocity.Reading("j", 2.0, 2.5).over
        assert not velocity.Reading("j", 2.0, 1.5).over

    def test_exactly_at_the_limit_is_not_over(self):
        """Pose channels are float32; a move keyed at the limit reads back a hair over."""
        assert not velocity.Reading("j", 3.15, 3.15 * (1 + 1e-6)).over

    def test_an_unknown_speed_is_never_over(self):
        reading = velocity.Reading("j", 2.0, None)
        assert reading.ratio is None
        assert not reading.over

    def test_a_reading_is_a_speed_unless_it_says_otherwise(self):
        assert velocity.Reading("j", 2.0, 1.0).kind == velocity.SPEED

    def test_an_acceleration_is_judged_the_same_way(self):
        reading = velocity.Reading("j", 10.0, 12.0, kind=velocity.ACCELERATION)
        assert reading.over
        assert reading.ratio == pytest.approx(1.2)


class TestSpeed:
    def test_one_frame_is_the_change_times_the_frame_rate(self):
        assert velocity.speed(0.5, 0.0, 1, 24.0) == pytest.approx(12.0)

    def test_direction_does_not_matter(self):
        assert velocity.speed(0.0, 0.5, 1, 24.0) == pytest.approx(12.0)

    def test_a_gap_is_averaged(self):
        assert velocity.speed(0.6, 0.0, 3, 30.0) == pytest.approx(6.0)

    def test_stepping_back_measures_the_same_interval(self):
        assert velocity.speed(0.0, 0.5, -1, 24.0) == pytest.approx(12.0)


class TestAcceleration:
    def test_even_frames_give_the_second_difference(self):
        # (0.3 - 2 * 0.1 + 0.0) * 24 * 24
        points = [(10, 0.3), (9, 0.1), (8, 0.0)]
        assert velocity.acceleration(points, 24.0) == pytest.approx(57.6)

    def test_a_steady_speed_has_none(self):
        points = [(10, 0.3), (9, 0.2), (8, 0.1)]
        assert velocity.acceleration(points, 24.0) == pytest.approx(0.0, abs=1e-9)

    def test_slowing_down_reads_like_speeding_up(self):
        speeding = velocity.acceleration([(10, 0.3), (9, 0.1), (8, 0.0)], 24.0)
        slowing = velocity.acceleration([(10, 0.3), (9, 0.2), (8, 0.0)], 24.0)
        assert slowing == pytest.approx(speeding)

    def test_the_order_of_the_points_does_not_matter(self):
        points = [(10, 0.3), (9, 0.1), (8, 0.0)]
        assert velocity.acceleration(points[::-1], 24.0) == pytest.approx(
            velocity.acceleration(points, 24.0)
        )

    def test_a_dropped_frame_is_not_a_spike(self):
        """Uneven frames are fitted, not differenced as though they were even."""
        # Half of 3 rad/s² times t², seen at frames 10, 9 and 7.
        fps = 24.0
        points = [(frame, 0.5 * 3.0 * (frame / fps) ** 2) for frame in (10, 9, 7)]
        assert velocity.acceleration(points, fps) == pytest.approx(3.0)


class TestClamp:
    @pytest.mark.parametrize(
        ("value", "expected"), [(-3.0, -2.0), (0.5, 0.5), (3.0, 2.0)]
    )
    def test_clamps_into_range(self, value, expected):
        assert velocity.clamp(value, -2.0, 2.0) == expected

    def test_a_missing_bound_does_not_clamp(self):
        assert velocity.clamp(9.0, None, None) == 9.0
        assert velocity.clamp(-9.0, -1.0, None) == -1.0


class TestHistory:
    def test_nothing_seen_gives_nothing(self):
        assert velocity.History().before("rig", 10) is None

    def test_the_previous_frame_is_found(self):
        history = velocity.History()
        history.observe("rig", 10, {"j": 0.1})
        history.observe("rig", 11, {"j": 0.2})
        assert history.before("rig", 11) == (10, {"j": 0.1})

    def test_the_frame_not_yet_observed_uses_the_last_one(self):
        """A draw can run before the handler has seen the new frame."""
        history = velocity.History()
        history.observe("rig", 10, {"j": 0.1})
        assert history.before("rig", 11) == (10, {"j": 0.1})

    def test_seeing_the_same_frame_again_keeps_the_one_before(self):
        """Editing the pose on frame 11 is still measured against frame 10."""
        history = velocity.History()
        history.observe("rig", 10, {"j": 0.1})
        history.observe("rig", 11, {"j": 0.2})
        history.observe("rig", 11, {"j": 0.9})
        assert history.before("rig", 11) == (10, {"j": 0.1})

    def test_stepping_back_measures_against_the_frame_just_left(self):
        history = velocity.History()
        history.observe("rig", 9, {"j": 0.0})
        history.observe("rig", 10, {"j": 0.1})
        history.observe("rig", 11, {"j": 0.2})
        history.observe("rig", 10, {"j": 0.1})
        assert history.before("rig", 10) == (11, {"j": 0.2})

    def test_the_nearer_of_two_records_wins(self):
        """Asked about frame 12 before it is seen: frame 11, not an average back to 10."""
        history = velocity.History()
        history.observe("rig", 10, {"j": 0.1})
        history.observe("rig", 11, {"j": 0.2})
        assert history.before("rig", 12) == (11, {"j": 0.2})

    def test_a_small_gap_is_kept(self):
        history = velocity.History()
        history.observe("rig", 10, {"j": 0.1})
        assert history.before("rig", 10 + velocity.MAX_GAP) is not None

    def test_a_jump_is_forgotten(self):
        """An average over a long jump says nothing about any one frame."""
        history = velocity.History()
        history.observe("rig", 10, {"j": 0.1})
        assert history.before("rig", 11 + velocity.MAX_GAP) is None

    def test_rigs_do_not_share_a_history(self):
        history = velocity.History()
        history.observe("a", 10, {"j": 0.1})
        assert history.before("b", 11) is None

    def test_the_values_are_copied(self):
        values = {"j": 0.1}
        history = velocity.History()
        history.observe("rig", 10, values)
        values["j"] = 5.0
        assert history.before("rig", 11)[1] == {"j": 0.1}

    def test_an_acceleration_needs_two_other_frames(self):
        history = velocity.History()
        history.observe("rig", 10, {"j": 0.1})
        history.observe("rig", 11, {"j": 0.2})
        assert history.two_before("rig", 11) is None

    def test_the_two_frames_before_come_nearest_first(self):
        history = velocity.History()
        for frame, value in ((9, 0.0), (10, 0.1), (11, 0.3)):
            history.observe("rig", frame, {"j": value})
        assert history.two_before("rig", 11) == [(10, {"j": 0.1}), (9, {"j": 0.0})]

    def test_the_frame_not_yet_observed_uses_the_last_two(self):
        """A draw can run before the handler has seen the new frame."""
        history = velocity.History()
        history.observe("rig", 9, {"j": 0.0})
        history.observe("rig", 10, {"j": 0.1})
        assert [frame for frame, _ in history.two_before("rig", 11)] == [10, 9]

    def test_seeing_the_same_frame_again_keeps_the_two_before(self):
        """Editing the pose on frame 11 is still measured against frames 10 and 9."""
        history = velocity.History()
        for frame, value in ((9, 0.0), (10, 0.1), (11, 0.3), (11, 0.9)):
            history.observe("rig", frame, {"j": value})
        assert [frame for frame, _ in history.two_before("rig", 11)] == [10, 9]

    def test_a_jump_leaves_no_acceleration(self):
        history = velocity.History()
        history.observe("rig", 9, {"j": 0.0})
        history.observe("rig", 10, {"j": 0.1})
        assert history.two_before("rig", 11 + velocity.MAX_GAP) is None

    def test_forget_one_or_all(self):
        history = velocity.History()
        history.observe("a", 10, {"j": 0.1})
        history.observe("b", 10, {"j": 0.1})
        history.forget("a")
        assert history.before("a", 11) is None
        assert history.before("b", 11) is not None
        history.forget()
        assert history.before("b", 11) is None


class TestFormat:
    def test_unknown_is_a_dash(self):
        assert velocity.format_rate(None, prismatic=False, degrees=True) == "—"

    def test_degrees(self):
        text = velocity.format_rate(math.radians(180), prismatic=False, degrees=True)
        assert text == "180°/s"

    def test_slow_degrees_keep_a_decimal(self):
        text = velocity.format_rate(math.radians(2.5), prismatic=False, degrees=True)
        assert text == "2.5°/s"

    def test_radians(self):
        assert velocity.format_rate(3.15, prismatic=False, degrees=False) == "3.15 rad/s"

    def test_prismatic_is_metres(self):
        assert velocity.format_rate(0.25, prismatic=True, degrees=True) == "0.25 m/s"

    def test_an_acceleration_is_per_second_squared(self):
        text = velocity.format_rate(
            math.radians(180), prismatic=False, degrees=True, order=2
        )
        assert text == "180°/s²"

    def test_a_jerk_is_per_second_cubed(self):
        assert velocity.format_rate(7500.0, prismatic=False, degrees=False, order=3) == (
            "7500 rad/s³"
        )

    def test_a_large_figure_has_no_exponent(self):
        assert velocity.format_rate(1500.0, prismatic=True, degrees=True, order=3) == (
            "1500 m/s³"
        )

    def test_effort_is_a_torque_or_a_force(self):
        assert velocity.format_effort(87.0, prismatic=False) == "87 N·m"
        assert velocity.format_effort(1500.0, prismatic=True) == "1500 N"
        assert velocity.format_effort(None, prismatic=False) == "—"
