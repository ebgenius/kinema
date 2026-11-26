"""How fast a traced tool moved and turned: the part that needs no Blender."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ..conftest import load_addon_module

trace = load_addon_module("rig.trace")

FPS = 30.0


def _turned(axis, angle):
    """A rotation matrix: ``angle`` radians about ``axis``."""
    x, y, z = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def _pose(position=(0.0, 0.0, 0.0), rotation=None):
    pose = np.eye(4)
    pose[:3, :3] = np.eye(3) if rotation is None else rotation
    pose[:3, 3] = position
    return pose


def test_a_steady_line_has_its_speed_and_no_acceleration():
    poses = [_pose((0.01 * frame, 0.02 * frame, 0.0)) for frame in range(6)]
    rates = trace.rates(poses, FPS)
    assert len(rates[trace.SPEED]) == 4
    np.testing.assert_allclose(rates[trace.SPEED], math.hypot(0.01, 0.02) * FPS)
    np.testing.assert_allclose(rates[trace.ACCELERATION], 0.0, atol=1e-9)
    np.testing.assert_allclose(rates[trace.ANGULAR_SPEED], 0.0, atol=1e-9)


def test_speed_is_over_the_frame_before_and_acceleration_from_either_side():
    # x = a t² / 2, sampled a frame apart: the speed over each frame is the mean
    # over it, and the second difference is exactly a.
    a = 2.0
    times = np.arange(5) / FPS
    poses = [_pose((0.5 * a * t * t, 0.0, 0.0)) for t in times]
    rates = trace.rates(poses, FPS)
    np.testing.assert_allclose(rates[trace.SPEED], a * (times[1:-1] - 0.5 / FPS))
    np.testing.assert_allclose(rates[trace.ACCELERATION], a)


def test_turning_at_a_steady_rate_about_a_tilted_axis_has_no_angular_acceleration():
    axis, rate = (1.0, 2.0, -0.5), 1.2
    poses = [_pose(rotation=_turned(axis, rate * frame / FPS)) for frame in range(6)]
    rates = trace.rates(poses, FPS)
    np.testing.assert_allclose(rates[trace.ANGULAR_SPEED], rate)
    np.testing.assert_allclose(rates[trace.ANGULAR_ACCELERATION], 0.0, atol=1e-9)
    np.testing.assert_allclose(rates[trace.SPEED], 0.0, atol=1e-12)


def test_a_turn_that_changes_axis_at_the_same_rate_is_an_angular_acceleration():
    # A quarter-turn a second about Z, then about X: the rate of turn changes as a
    # vector by rate * sqrt(2) over a frame, though its size never does.
    step = 0.25 / FPS
    rotations = [np.eye(3)]
    for axis in [(0, 0, 1)] * 2 + [(1, 0, 0)] * 2:
        rotations.append(_turned(axis, step) @ rotations[-1])
    rates = trace.rates([_pose(rotation=r) for r in rotations], FPS)
    np.testing.assert_allclose(rates[trace.ANGULAR_SPEED], 0.25, rtol=1e-9)
    np.testing.assert_allclose(
        rates[trace.ANGULAR_ACCELERATION], [0.0, 0.25 * math.sqrt(2.0) * FPS, 0.0], atol=1e-9
    )


@pytest.mark.parametrize("angle", [0.1, 2.0, math.pi - 1e-3])
def test_a_rotation_vector_is_its_axis_times_its_angle(angle):
    axis = np.array([0.3, -0.4, 0.5])
    found = trace.rotation_vector(_turned(axis, angle))
    np.testing.assert_allclose(found, axis / np.linalg.norm(axis) * angle, atol=1e-6)


def test_more_than_a_half_turn_is_the_short_way_round():
    found = trace.rotation_vector(_turned((0.0, 0.0, 1.0), 1.5 * math.pi))
    np.testing.assert_allclose(found, (0.0, 0.0, -0.5 * math.pi), atol=1e-9)


def test_a_rate_needs_a_pose_either_side():
    with pytest.raises(ValueError):
        trace.rates([_pose(), _pose()], FPS)
