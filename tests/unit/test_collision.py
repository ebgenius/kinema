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
        """Its radius, and its length less a little: a hair wider and the ends trim."""
        points = _cylinder_points(0.1, 1.0)
        capsules = collision.fit(points)
        assert len(capsules) == 1
        a, b, radius = capsules[0]
        assert radius == pytest.approx(0.1, abs=0.002)
        assert np.linalg.norm(b - a) == pytest.approx(1.0, abs=0.025)
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

    @staticmethod
    def _block(half):
        corners = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * half
        faces = [
            (0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
            (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3),
        ]
        return collision.surface_points(corners, np.array(faces))

    @staticmethod
    def _cylinder_rounded(points):
        """The volume of the narrowest cylinder with its ends rounded, as fitted before."""
        import trimesh

        cylinder = trimesh.bounds.minimum_cylinder(trimesh.convex.convex_hull(points))
        radius, height = float(cylinder["radius"]), float(cylinder["height"])
        return math.pi * radius**2 * height + 4.0 / 3.0 * math.pi * radius**3, radius

    def test_a_squat_block_s_capsule_doesn_t_stand_a_radius_past_its_faces(self):
        """A KR120's wrist is about this block. The narrowest cylinder stands on its
        shortest side, and rounding its ends put each a whole radius past a face."""
        half = np.array([0.15, 0.145, 0.105])
        points = self._block(half)
        ((a, b, r),) = collision.fit(points, max_parts=1)
        overhang = np.maximum(np.maximum(a, b) + r - half, -half - (np.minimum(a, b) - r))
        assert overhang.max() < 0.6 * r
        before, _ = self._cylinder_rounded(points)
        assert collision.volume(r, a, b) < 0.85 * before
        assert _inside_any(points, [(a, b, r)]).all()

    def test_a_plate_s_capsule_lies_along_it(self):
        """Its narrowest cylinder stands square to it, and rounded, that is nearly a
        ball 0.5 m across: 0.53 m³. Along the plate's own spread, 0.34."""
        random = np.random.default_rng(7)
        turn, _ = np.linalg.qr(random.normal(size=(3, 3)))
        points = self._block(np.array([0.4, 0.3, 0.03])) @ turn
        ((a, b, r),) = collision.fit(points, max_parts=1)
        assert collision.volume(r, a, b) < 0.4
        assert _inside_any(points, [(a, b, r)]).all()

    @pytest.mark.parametrize("shape", ["block", "long block", "l", "cylinder", "blob"])
    def test_never_bigger_than_the_narrowest_cylinder_rounded(self, shape):
        """Its axis is one of those tried, with a segment trimmed from its height."""
        random = np.random.default_rng(7)
        points = {
            "block": lambda: self._block(np.array([0.15, 0.145, 0.105])),
            "long block": lambda: self._block(np.array([0.5, 0.08, 0.06])),
            "l": lambda: np.r_[
                _cylinder_points(0.05, 1.0, seed=1) + [0.0, 0.0, 0.5],
                _cylinder_points(0.05, 1.0, seed=2)[:, [2, 1, 0]] + [0.5, 0.0, 0.0],
            ],
            "cylinder": lambda: _cylinder_points(0.1, 1.0),
            "blob": lambda: random.normal(size=(2000, 3)) * [0.3, 0.1, 0.2],
        }[shape]()
        # The capsule step alone, on the points the fit hands it: fit() widens
        # afterwards to take in the points thinning left out, either way.
        thinned = collision.thin(points)
        a, b, r = collision._capsule(thinned)
        before, _ = self._cylinder_rounded(thinned)
        assert collision.volume(r, a, b) <= before * (1.0 + 1e-6)

    def test_a_count_asked_for_is_the_count_fitted(self):
        """Fewer than the fit would choose, or more: each holds every point."""
        upright = _cylinder_points(0.05, 1.0, seed=1) + [0.0, 0.0, 0.5]
        arm = _cylinder_points(0.05, 1.0, seed=2)[:, [2, 1, 0]] + [0.5, 0.0, 0.0]
        points = np.r_[upright, arm]
        chosen = len(collision.fit(points))
        for parts in (1, 3, 4):
            assert parts != chosen
            capsules = collision.fit(points, parts=parts)
            assert len(capsules) == parts
            assert _inside_any(points, capsules).all()

    def test_a_count_asked_for_is_the_split_the_fit_would_try(self):
        """The same capsules as the fit chooses, where it chooses that count."""
        upright = _cylinder_points(0.05, 1.0, seed=1) + [0.0, 0.0, 0.5]
        arm = _cylinder_points(0.05, 1.0, seed=2)[:, [2, 1, 0]] + [0.5, 0.0, 0.0]
        points = np.r_[upright, arm]
        chosen = collision.fit(points)
        asked = collision.fit(points, parts=len(chosen))
        for (a1, b1, r1), (a2, b2, r2) in zip(chosen, asked, strict=True):
            np.testing.assert_array_equal(a1, a2)
            np.testing.assert_array_equal(b1, b2)
            assert r1 == r2

    def test_points_too_few_to_split_that_far_get_fewer(self):
        """Four capsules need sixteen points to cluster: six get one."""
        points = np.array(
            [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1),
             (0.1, 0.1, 0.1), (0.2, 0.0, 0.0)]
        )
        capsules = collision.fit(points, parts=4)
        assert len(capsules) == 1
        assert _inside_any(points, capsules).all()

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

    def test_a_disc_s_capsule_is_no_bigger_than_the_ball_round_it(self):
        """Through it, and short: near the ball round it, never across it, as the crude
        fit along its spread lies.

        A ball 0.503 m across holds a disc 0.5 m in radius and 0.1 m thick in less
        than the narrowest cylinder rounded, through it: 0.53 m³ against 0.60.
        """
        random = np.random.default_rng(3)
        angle = random.uniform(0.0, 2.0 * math.pi, 3000)
        radius = np.sqrt(random.uniform(0.0, 1.0, 3000)) * 0.5
        height = random.uniform(-0.05, 0.05, 3000)
        points = np.c_[radius * np.cos(angle), radius * np.sin(angle), height]
        ((a, b, r),) = collision.fit(points, max_parts=1)
        ball = 4.0 / 3.0 * math.pi * math.hypot(0.5, 0.05) ** 3
        assert collision.volume(r, a, b) <= ball * 1.001
        if np.linalg.norm(b - a) > 1e-6:
            assert abs(((b - a) / np.linalg.norm(b - a))[2]) > 0.99
        assert _inside_any(points, [(a, b, r)]).all()

    def test_without_trimesh_fitting_fails_rather_than_degrading(self, monkeypatch):
        """Every capsule would quietly take the crude fit instead."""
        import sys

        monkeypatch.setitem(sys.modules, "trimesh", None)
        with pytest.raises(ImportError):
            collision.fit(_cylinder_points(0.1, 1.0))

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
