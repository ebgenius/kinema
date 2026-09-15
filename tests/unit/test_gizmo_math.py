"""Where the IK control's handles put it: slide along an axis, turn about one."""

from __future__ import annotations

import numpy as np
import pytest

from ..conftest import load_addon_module

gizmo_math = load_addon_module("rig.gizmo_math")


def _pose() -> np.ndarray:
    """A control somewhere off the origin, already turned."""
    matrix = np.eye(4)
    matrix[:3, :3] = gizmo_math.rotation([0.3, -0.5, 0.8], 0.9)
    matrix[:3, 3] = [0.4, -0.2, 0.7]
    return matrix


class TestSlide:
    def test_it_moves_only_the_origin_and_only_along_the_axis(self):
        start = _pose()
        moved = gizmo_math.slide(start, [0.0, 0.0, 1.0], 0.25)
        np.testing.assert_allclose(moved[:3, :3], start[:3, :3])
        np.testing.assert_allclose(moved[:3, 3] - start[:3, 3], [0.0, 0.0, 0.25])

    def test_the_distance_is_measured_along_a_unit_axis(self):
        """A handle axis read off a scaled frame is not unit length; the drag still is."""
        moved = gizmo_math.slide(np.eye(4), [0.0, 3.0, 4.0], 1.0)
        np.testing.assert_allclose(moved[:3, 3], [0.0, 0.6, 0.8])

    def test_the_start_is_left_alone(self):
        start = _pose()
        before = start.copy()
        gizmo_math.slide(start, [1.0, 0.0, 0.0], 0.5)
        np.testing.assert_array_equal(start, before)


class TestTurn:
    def test_the_tool_point_stays_where_it_is(self):
        start = _pose()
        turned = gizmo_math.turn(start, [1.0, 1.0, 0.0], 1.2)
        np.testing.assert_allclose(turned[:3, 3], start[:3, 3])

    def test_a_positive_angle_is_right_handed(self):
        """A quarter turn about Z carries X onto Y, not onto -Y."""
        turned = gizmo_math.turn(np.eye(4), [0.0, 0.0, 1.0], np.pi / 2)
        np.testing.assert_allclose(turned[:3, 0], [0.0, 1.0, 0.0], atol=1e-12)

    def test_it_is_about_a_fixed_axis_not_the_frame_s_own(self):
        """The axis is where the handle points, not an axis of the turned frame."""
        start = _pose()
        axis = np.array([0.0, 1.0, 0.0])
        turned = gizmo_math.turn(start, axis, 0.7)
        np.testing.assert_allclose(
            turned[:3, :3], gizmo_math.rotation(axis, 0.7) @ start[:3, :3], atol=1e-12
        )

    def test_turns_about_one_axis_add_up(self):
        start = _pose()
        axis = [0.2, 0.9, -0.4]
        twice = gizmo_math.turn(gizmo_math.turn(start, axis, 0.3), axis, 0.5)
        np.testing.assert_allclose(twice, gizmo_math.turn(start, axis, 0.8), atol=1e-12)

    def test_the_rotation_stays_a_rotation(self):
        turned = gizmo_math.turn(_pose(), [1.0, -2.0, 0.5], 2.4)
        np.testing.assert_allclose(turned[:3, :3] @ turned[:3, :3].T, np.eye(3), atol=1e-12)


class TestAxisBasis:
    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_the_chosen_axis_becomes_z_and_the_frame_stays_right_handed(self, index):
        frame = _pose()
        cycled = gizmo_math.axis_basis(frame, index)
        np.testing.assert_allclose(cycled[:3, 2], frame[:3, index])
        np.testing.assert_allclose(cycled[:3, 3], frame[:3, 3])
        assert np.linalg.det(cycled[:3, :3]) == pytest.approx(1.0)


def test_an_axis_with_no_length_is_refused():
    with pytest.raises(ValueError, match="direction"):
        gizmo_math.slide(np.eye(4), [0.0, 0.0, 0.0], 1.0)
