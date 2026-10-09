"""Collision outside Blender: fitting capsules to points, and the distances the check reports.

The claims checked here:
- every point a capsule set was fitted to is inside one of its capsules;
- a link shaped like an L gets more than one capsule, and a tighter set for it;
- the same points always give the same capsules;
- distances to a box, a sphere and a floor are exact and signed, inside as
  well as out, against a brute-force sample of every point on the segment.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ..conftest import load_addon_module

collision = load_addon_module("rig.collision")


def _rotation(seed):
    random = np.random.default_rng(seed)
    q, _ = np.linalg.qr(random.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _box(centre, rotation, half, name="box"):
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, centre
    return collision.Obstacle(name, collision.KIND_BOX, matrix, np.asarray(half, dtype=float))


def _cylinder_points(radius, length, count=4000, seed=0):
    random = np.random.default_rng(seed)
    angle = random.uniform(0.0, 2.0 * math.pi, count)
    z = random.uniform(-length / 2.0, length / 2.0, count)
    return np.c_[radius * np.cos(angle), radius * np.sin(angle), z]


def _inside_any(points, capsules, tolerance=1e-6):
    """Whether each point is inside at least one capsule."""
    inside = np.zeros(len(points), dtype=bool)
    for a, b, radius in capsules:
        distance = collision._point_segment_distance(points, np.asarray(a), np.asarray(b))
        inside |= distance <= radius + tolerance
    return inside


class TestFitting:
    def test_a_cylinder_gets_one_capsule_its_own_size(self):
        points = _cylinder_points(0.1, 1.0)
        capsules = collision.fit(points)
        assert len(capsules) == 1
        a, b, radius = capsules[0]
        assert radius == pytest.approx(0.1, abs=0.002)
        assert np.linalg.norm(b - a) == pytest.approx(1.0, abs=0.01)
        assert _inside_any(points, capsules).all()

    def test_an_l_gets_several_tighter_capsules(self):
        """One capsule round an L has to swallow the corner between its arms."""
        upright = _cylinder_points(0.05, 1.0, seed=1) + [0.0, 0.0, 0.5]
        arm = _cylinder_points(0.05, 1.0, seed=2)[:, [2, 1, 0]] + [0.5, 0.0, 0.0]
        points = np.r_[upright, arm]
        one = collision.fit(points, max_parts=1)
        several = collision.fit(points)
        assert len(several) > 1
        total = sum(collision.volume(radius, a, b) for a, b, radius in several)
        assert total < 0.5 * collision.volume(one[0][2], one[0][0], one[0][1])
        assert _inside_any(points, several).all()

    def test_the_same_points_give_the_same_capsules(self):
        upright = _cylinder_points(0.05, 1.0, seed=1) + [0.0, 0.0, 0.5]
        arm = _cylinder_points(0.05, 1.0, seed=2)[:, [2, 1, 0]] + [0.5, 0.0, 0.0]
        points = np.r_[upright, arm]
        first, second = collision.fit(points), collision.fit(points)
        assert len(first) == len(second)
        for (a1, b1, r1), (a2, b2, r2) in zip(first, second, strict=True):
            np.testing.assert_allclose(a1, a2)
            np.testing.assert_allclose(b1, b2)
            assert r1 == r2

    def test_a_flat_plate_is_still_bounded(self):
        """No volume, so no hull to take a cylinder of."""
        x, y = np.meshgrid(np.linspace(0.0, 0.3, 20), np.linspace(0.0, 0.1, 8))
        points = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
        capsules = collision.fit(points)
        assert capsules
        assert _inside_any(points, capsules).all()

    def test_a_box_s_faces_are_covered_not_only_its_corners(self):
        """Eight corners and nothing between: the middle of the box must be inside too."""
        side = (-0.035, 0.035)
        corners = np.array([(x, y, z) for x in (0.0, 0.4) for y in side for z in side])
        faces = [
            (0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
            (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3),
        ]
        capsules = collision.fit(collision.surface_points(corners, np.array(faces)))
        middles = np.array(
            [(0.2, 0.035, 0.0), (0.2, -0.035, 0.0), (0.2, 0.0, 0.035), (0.2, 0.0, 0.0)]
        )
        assert _inside_any(middles, capsules, tolerance=0.005).all()

    def test_surface_points_are_the_same_every_time(self):
        corners = np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
        first = collision.surface_points(corners, np.array([(0, 1, 2)]))
        again = collision.surface_points(corners, np.array([(0, 1, 2)]))
        np.testing.assert_array_equal(first, again)
        assert len(first) > 1000

    def test_no_points_no_capsules(self):
        assert collision.fit(np.zeros((0, 3))) == []


class TestDistances:
    @staticmethod
    def _brute(a, b, obstacle, count=400001):
        t = np.linspace(0.0, 1.0, count)[:, None]
        points = a + t * (b - a)
        if obstacle.kind == collision.KIND_BOX:
            matrix = obstacle.matrix
            local = (points - matrix[:3, 3]) @ matrix[:3, :3]
            return float(collision.box_sdf(local, obstacle.size).min())
        raise AssertionError(obstacle.kind)

    @pytest.mark.parametrize("seed", range(12))
    def test_a_segment_to_a_turned_box_is_exact(self, seed):
        """Outside, grazing and through it: the least signed distance along the segment."""
        random = np.random.default_rng(seed)
        box = _box(random.normal(size=3) * 0.2, _rotation(seed), random.uniform(0.05, 0.4, 3))
        a = random.normal(size=3) * 0.6
        b = random.normal(size=3) * 0.6
        found = float(collision.segment_distance(a, b, box))
        assert found == pytest.approx(self._brute(a, b, box), abs=2e-5)

    def test_a_segment_through_a_box_is_negative_by_its_depth(self):
        box = _box(np.zeros(3), np.eye(3), [0.5, 0.5, 0.5])
        a, b = np.array([-1.0, 0.1, 0.0]), np.array([1.0, 0.1, 0.0])
        assert float(collision.segment_distance(a, b, box)) == pytest.approx(-0.4, abs=1e-6)

    def test_a_segment_to_a_sphere(self):
        sphere = collision.Obstacle("ball", collision.KIND_SPHERE, np.eye(4), np.array([0.2]))
        a, b = np.array([-1.0, 0.5, 0.0]), np.array([1.0, 0.5, 0.0])
        assert float(collision.segment_distance(a, b, sphere)) == pytest.approx(0.3)

    def test_a_segment_to_a_floor_goes_negative_below_it(self):
        matrix = np.eye(4)
        matrix[2, 3] = 0.1
        floor = collision.Obstacle("floor", collision.KIND_FLOOR, matrix, np.zeros(3))
        a, b = np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.05])
        assert float(collision.segment_distance(a, b, floor)) == pytest.approx(-0.05)

    def test_clearances_take_off_each_radius_over_every_frame(self):
        box = _box(np.zeros(3), np.eye(3), [0.1, 0.1, 0.1])
        # Two frames, two capsules, each above the box by its height.
        a = np.array([[[0.0, 0.0, 0.5], [0.0, 0.0, 1.0]], [[0.0, 0.0, 0.3], [0.0, 0.0, 2.0]]])
        b = a + [1.0, 0.0, 0.0]
        found = collision.clearances(a, b, np.array([0.1, 0.2]), [box])
        assert found.shape == (2, 2, 1)
        np.testing.assert_allclose(found[..., 0], [[0.3, 0.7], [0.1, 1.7]], atol=1e-6)
