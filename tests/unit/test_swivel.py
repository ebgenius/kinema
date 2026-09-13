"""The swivel's geometry: turning an angle into a point the elbow can reach."""

from __future__ import annotations

import numpy as np
import pytest

from ..conftest import load_addon_module

swivel = load_addon_module("solver.swivel")

SHOULDER = np.array([0.0, 0.0, 0.25])
WRIST = np.array([0.45, 0.2, 0.55])
UPPER = 0.4
FORE = 0.35


def _directions(count: int = 36, seed: int = 0):
    return np.random.default_rng(seed).normal(size=(count, 3))


class TestTheGoalIsReachable:
    def test_it_is_an_upper_arm_from_the_shoulder_and_a_forearm_from_the_wrist(self):
        """The property the whole control rests on.

        On a spherical-shoulder, spherical-wrist arm, an elbow can be anywhere
        exactly that far from both centres. A goal that meets both lengths is
        therefore a place the elbow can actually be, so pulling toward it costs
        the tool nothing. A goal that misses either one is the free target this
        replaced.
        """
        for direction in _directions():
            goal = swivel.elbow_goal(SHOULDER, WRIST, UPPER, FORE, direction, WRIST)
            assert np.linalg.norm(goal - SHOULDER) == pytest.approx(UPPER, abs=1e-9)
            assert np.linalg.norm(WRIST - goal) == pytest.approx(FORE, abs=1e-9)

    def test_the_knob_chooses_the_side(self):
        """The goal lies on the side of the axis the knob points to."""
        centre, axis, _ = swivel.elbow_circle(SHOULDER, WRIST, UPPER, FORE)
        for direction in _directions():
            goal = swivel.elbow_goal(SHOULDER, WRIST, UPPER, FORE, direction, WRIST)
            flat = direction - np.dot(direction, axis) * axis
            assert np.dot(goal - centre, flat) > 0.0

    def test_leaning_the_knob_along_the_axis_changes_nothing(self):
        """Only the angle round S-W can move the elbow round its circle."""
        _, axis, _ = swivel.elbow_circle(SHOULDER, WRIST, UPPER, FORE)
        direction = np.array([0.0, 0.0, 1.0])
        plain = swivel.elbow_goal(SHOULDER, WRIST, UPPER, FORE, direction, WRIST)
        leaning = swivel.elbow_goal(
            SHOULDER, WRIST, UPPER, FORE, direction + 3.0 * axis, WRIST
        )
        assert np.allclose(plain, leaning, atol=1e-12)


class TestDegenerateArms:
    def test_out_of_reach_the_goal_keeps_the_knob_side(self):
        """Wrist past full reach: the goal stays just off the line, on the knob's side.

        The true circle there is a point on the straight line, and a goal on the
        line has no side -- a stretched arm solved toward it can come out with
        the elbow flipped. arm7's did, by 162 degrees.
        """
        far = SHOULDER + np.array([2.0, 0.0, 0.0])
        knob = np.array([0.0, 1.0, 0.0])
        centre, axis, radius = swivel.elbow_circle(SHOULDER, far, UPPER, FORE)
        assert np.allclose(centre, SHOULDER + UPPER * axis)
        assert radius == pytest.approx(swivel.STRETCHED_SIDE * UPPER)

        goal = swivel.elbow_goal(SHOULDER, far, UPPER, FORE, knob, far)
        assert np.all(np.isfinite(goal))
        assert np.dot(goal - centre, knob) > 0.0, "the goal is not on the knob's side"
        assert float(np.linalg.norm(goal - centre)) == pytest.approx(radius)

    def test_the_side_floor_leaves_an_ordinary_goal_alone(self):
        """Away from full stretch the real circle is far wider than the floor."""
        _, _, radius = swivel.elbow_circle(SHOULDER, WRIST, UPPER, FORE)
        assert radius > 3 * swivel.STRETCHED_SIDE * UPPER

    def test_a_wrist_closer_than_the_arm_can_fold_keeps_the_goal_within_reach(self):
        """Inside the inner limit the circle's formula divides by a tiny distance.

        Unclamped, a wrist 1 mm from the shoulder put the elbow goal over 18 m
        away -- a pull the tool would pay for. Clamped, the goal is where the
        folded elbow would be.
        """
        near = SHOULDER + np.array([0.001, 0.0, 0.0])
        goal = swivel.elbow_goal(
            SHOULDER, near, UPPER, FORE, np.array([0.0, 1.0, 0.0]), near
        )
        assert np.all(np.isfinite(goal))
        assert float(np.linalg.norm(goal - SHOULDER)) < UPPER * 1.01

    def test_a_knob_along_the_axis_keeps_the_elbow_on_its_side(self):
        """No side to read from the knob, so keep the one the elbow is on."""
        centre, axis, radius = swivel.elbow_circle(SHOULDER, WRIST, UPPER, FORE)
        seed = swivel.elbow_goal(
            SHOULDER, WRIST, UPPER, FORE, np.array([0.3, -1.0, 0.2]), WRIST
        )
        goal = swivel.elbow_goal(SHOULDER, WRIST, UPPER, FORE, axis.copy(), seed)
        assert np.allclose(goal, seed, atol=1e-9)

    def test_nothing_to_go_on_gives_the_centre_not_nan(self):
        centre, axis, _ = swivel.elbow_circle(SHOULDER, WRIST, UPPER, FORE)
        goal = swivel.elbow_goal(SHOULDER, WRIST, UPPER, FORE, axis, centre)
        assert np.allclose(goal, centre)

    def test_coincident_shoulder_and_wrist_do_not_raise(self):
        goal = swivel.elbow_goal(
            SHOULDER, SHOULDER, UPPER, FORE, np.array([1.0, 0.0, 0.0]), WRIST
        )
        assert np.allclose(goal, WRIST)


class TestSignedAngle:
    def test_a_quarter_turn_about_z_is_plus_a_quarter(self):
        angle = swivel.signed_angle(
            np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
        )
        assert angle == pytest.approx(np.pi / 2)

    def test_the_other_way_is_negative(self):
        angle = swivel.signed_angle(
            np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
        )
        assert angle == pytest.approx(-np.pi / 2)

    def test_components_along_the_axis_are_ignored(self):
        axis = np.array([0.0, 0.0, 1.0])
        angle = swivel.signed_angle(
            axis, np.array([1.0, 0.0, 5.0]), np.array([0.0, 1.0, -2.0])
        )
        assert angle == pytest.approx(np.pi / 2)

    def test_rotating_by_the_angle_lands_on_the_target(self):
        """The use it is put to: turn the knob by this and it points at the elbow."""
        rng = np.random.default_rng(3)
        for _ in range(20):
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            start, end = rng.normal(size=3), rng.normal(size=3)
            angle = swivel.signed_angle(axis, start, end)

            # Rodrigues: rotate start's perpendicular part about the axis.
            flat = start - np.dot(start, axis) * axis
            turned = (
                flat * np.cos(angle)
                + np.cross(axis, flat) * np.sin(angle)
            )
            target = end - np.dot(end, axis) * axis
            assert np.allclose(
                turned / np.linalg.norm(turned),
                target / np.linalg.norm(target),
                atol=1e-9,
            )

    def test_no_direction_gives_zero_not_nan(self):
        axis = np.array([0.0, 0.0, 1.0])
        assert swivel.signed_angle(axis, axis, np.array([1.0, 0.0, 0.0])) == 0.0
