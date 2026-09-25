"""External axes without Blender: presets, where things go, and placeholder shapes."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from ..conftest import load_addon_module

ext = load_addon_module("rig.external_axes")
kinematics = load_addon_module("rig.kinematics")


def _random_pose(rng) -> np.ndarray:
    return kinematics.make_transform(rng.uniform(-1.0, 1.0, 3), rng.uniform(-math.pi, math.pi, 3))


class TestPresets:
    def test_every_preset_can_be_built(self):
        for key in ext.PRESETS:
            assert ext.from_preset(key).problems() == [], key

    def test_the_catalogue_covers_every_mount(self):
        mounts = {ext.from_preset(key).mount for key in ext.PRESETS}
        assert mounts == set(ext.MOUNTS)

    def test_a_field_can_be_overridden(self):
        spec = ext.from_preset("LINEAR_TRACK", name="gantry", upper=4.0)
        assert (spec.name, spec.upper, spec.kind) == ("gantry", 4.0, "LINEAR")


class TestSpec:
    def test_joint_types(self):
        assert ext.AxisSpec("a", kind="LINEAR").joint_type == "prismatic"
        assert ext.AxisSpec("a", kind="ROTARY").joint_type == "revolute"
        assert ext.AxisSpec("a", kind="ROTARY", continuous=True).joint_type == "continuous"

    def test_a_continuous_axis_has_no_limits_and_a_linear_one_always_does(self):
        assert ext.AxisSpec("a", kind="ROTARY", continuous=True).limits is None
        # Continuous means nothing for a slide: it still has end stops.
        assert ext.AxisSpec("a", kind="LINEAR", continuous=True).limits == (-1.0, 1.0)

    def test_no_speed_is_no_limit(self):
        assert ext.AxisSpec("a", velocity=0.0).speed is None
        assert ext.AxisSpec("a", velocity=None).speed is None
        assert ext.AxisSpec("a", velocity=0.5).speed == 0.5

    @pytest.mark.parametrize(
        ("changes", "complaint"),
        [
            ({"name": "  "}, "name"),
            ({"lower": 1.0, "upper": 1.0}, "lower limit"),
            ({"size": 0.0}, "size"),
            ({"mount": "SIDEWAYS"}, "mount"),
            ({"direction": "W"}, "direction"),
        ],
    )
    def test_problems_are_named(self, changes, complaint):
        problems = ext.AxisSpec(**{"name": "a", **changes}).problems()
        assert len(problems) == 1 and complaint in problems[0]

    def test_inverted_limits_are_fine_when_there_are_none(self):
        spec = ext.AxisSpec("a", kind="ROTARY", continuous=True, lower=1.0, upper=-1.0)
        assert spec.problems() == []


class TestPlacement:
    def test_by_default_the_robot_stays_where_it_is(self):
        rng = np.random.default_rng(1)
        for _ in range(10):
            base = _random_pose(rng)
            shift = ext.robot_shift(ext.AxisSpec("a"), base)
            np.testing.assert_allclose(shift, np.eye(4), atol=1e-12)

    def test_the_robot_base_lands_at_base_then_offset(self):
        """The shift takes the robot base exactly to base placement then offset."""
        rng = np.random.default_rng(2)
        for _ in range(10):
            robot = _random_pose(rng)
            spec = ext.AxisSpec(
                "a",
                base_location=tuple(rng.uniform(-1, 1, 3)),
                base_rotation=tuple(rng.uniform(-3, 3, 3)),
                offset_location=tuple(rng.uniform(-1, 1, 3)),
                offset_rotation=tuple(rng.uniform(-3, 3, 3)),
            )
            moved = ext.robot_shift(spec, robot) @ robot
            np.testing.assert_allclose(moved, robot @ spec.base @ spec.offset, atol=1e-12)

    def test_an_offset_that_undoes_the_base_moves_nothing(self):
        spec = ext.AxisSpec(
            "a", base_location=(0.0, 0.0, -0.3), offset_location=(0.0, 0.0, 0.3)
        )
        np.testing.assert_allclose(ext.robot_shift(spec, np.eye(4)), np.eye(4), atol=1e-12)

    def test_a_lift_moves_the_robot_up(self):
        spec = ext.AxisSpec("a", offset_location=(0.0, 0.0, 0.2))
        np.testing.assert_allclose(ext.robot_shift(spec, np.eye(4))[:3, 3], [0, 0, 0.2])

    @pytest.mark.parametrize("direction", list(ext.DIRECTIONS))
    def test_the_bone_points_along_the_axis_with_a_square_roll(self, direction):
        rng = np.random.default_rng(3)
        frame = _random_pose(rng)
        axis = np.array(ext.DIRECTIONS[direction])
        head, y, z = ext.bone_placement(frame, axis)
        np.testing.assert_allclose(head, frame[:3, 3])
        np.testing.assert_allclose(y, frame[:3, :3] @ axis, atol=1e-12)
        assert abs(float(np.dot(y, z))) < 1e-12
        assert np.linalg.norm(z) == pytest.approx(1.0)

    def test_a_rail_along_x_keeps_up_up(self):
        _, _, z = ext.bone_placement(np.eye(4), ext.DIRECTIONS["X"])
        np.testing.assert_allclose(z, [0, 0, 1])

    def test_default_sizes_follow_the_robot(self):
        assert ext.default_size("BEFORE", 2.0) == pytest.approx(0.5)
        assert ext.default_size("AFTER", 2.0) < ext.default_size("BEFORE", 2.0)
        assert ext.default_size("BEFORE", 100.0) == 1.0  # clamped
        assert ext.default_size("BEFORE", 0.0) > 0.0


def _is_closed_and_outward(mesh) -> bool:
    """Every edge shared by two faces in opposite directions, every face facing out.

    "Out" is judged against the centroid, which is enough for the convex parts
    these placeholders are built from.
    """
    vertices, faces = (np.array(mesh[0], dtype=float), mesh[1])
    edges = Counter()
    for face in faces:
        for a, b in zip(face, (*face[1:], face[0]), strict=True):
            edges[(a, b)] += 1
    if any(count != 1 or edges[(b, a)] != 1 for (a, b), count in edges.items()):
        return False
    centre = vertices.mean(axis=0)
    for face in faces:
        points = vertices[list(face)]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        if float(np.dot(normal, points.mean(axis=0) - centre)) <= 0.0:
            return False
    return True


class TestPlaceholders:
    def test_a_box_is_closed_and_faces_out(self):
        assert _is_closed_and_outward(ext.box((-1, 2), (-0.5, 0.5), (0, 3)))

    def test_a_cylinder_is_closed_and_faces_out(self):
        assert _is_closed_and_outward(ext.cylinder(0.4, (-0.2, 0.1), segments=12))

    def test_the_rail_covers_the_whole_travel(self):
        spec = ext.AxisSpec("a", kind="LINEAR", lower=-1.2, upper=2.5, size=0.2)
        rail, carriage = ext.placeholder(spec)
        along = [v[1] for v in rail[0]]
        assert min(along) < -1.2 and max(along) > 2.5
        # The carriage is drawn at the joint's zero; the bone carries it along.
        assert max(v[1] for v in carriage[0]) < 0.5 * 0.2 + 1e-9

    def test_what_moves_tops_out_at_the_joint(self):
        """The carried thing sits on the moving part: its top is the joint origin."""
        linear = ext.placeholder(ext.AxisSpec("a", kind="LINEAR", size=0.3))
        rotary = ext.placeholder(ext.AxisSpec("a", kind="ROTARY", size=0.3))
        assert max(v[2] for v in linear[1][0]) == pytest.approx(0.0)
        assert max(v[1] for v in rotary[1][0]) == pytest.approx(0.0)
        # ...and the static part stays below it.
        assert max(v[2] for v in linear[0][0]) < 0.0
        assert max(v[1] for v in rotary[0][0]) < 0.0

    def test_every_part_is_a_valid_mesh(self):
        for key in ext.PRESETS:
            for mesh in ext.placeholder(ext.from_preset(key)):
                vertices, faces = mesh
                assert faces and all(0 <= i < len(vertices) for f in faces for i in f)

    def test_size_scales_everything(self):
        small = ext.placeholder(ext.AxisSpec("a", kind="ROTARY", size=0.1))
        large = ext.placeholder(ext.AxisSpec("a", kind="ROTARY", size=0.2))
        np.testing.assert_allclose(
            np.array(large[1][0]), 2.0 * np.array(small[1][0]), atol=1e-12
        )


class TestStacking:
    def test_a_track_stands_on_what_is_under_its_rail(self):
        spec = ext.from_preset("LINEAR_TRACK")
        drop = ext.footing(spec)
        np.testing.assert_allclose(drop[:3, :3], np.eye(3))
        assert drop[:3, 3] == pytest.approx([0.0, 0.0, -0.45 * spec.size])

    def test_a_rotary_base_stands_on_what_is_under_its_base(self):
        spec = ext.from_preset("ROTARY_BASE")
        assert ext.footing(spec)[2, 3] == pytest.approx(-0.35 * spec.size)

    def test_the_footing_is_the_lowest_point_of_the_shape(self):
        """Taken from the shape, so a lift along Z stands on the bottom of its column."""
        for direction in ext.DIRECTIONS:
            spec = ext.AxisSpec("a", direction=direction, size=0.2)
            static, _ = ext.placeholder(spec)
            bone = ext.bone_matrix(np.eye(4), spec.axis)
            lowest = (bone[:3, :3] @ np.array(static[0]).T)[2].min()
            assert ext.footing(spec)[2, 3] == pytest.approx(min(0.0, lowest)), direction

    def test_placeholder_or_not_the_footing_is_the_same(self):
        spec = ext.from_preset("LINEAR_TRACK")
        bare = ext.from_preset("LINEAR_TRACK", placeholder=False)
        np.testing.assert_allclose(ext.footing(spec), ext.footing(bare))


class TestBoneMatrix:
    @pytest.mark.parametrize("direction", list(ext.DIRECTIONS))
    def test_it_is_the_placement_as_a_right_handed_frame(self, direction):
        rng = np.random.default_rng(len(direction))
        frame = _random_pose(rng)
        axis = ext.DIRECTIONS[direction]
        matrix = ext.bone_matrix(frame, axis)
        head, y_direction, z_reference = ext.bone_placement(frame, axis)
        np.testing.assert_allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-12)
        assert np.linalg.det(matrix[:3, :3]) == pytest.approx(1.0)
        np.testing.assert_allclose(matrix[:3, 3], head)
        np.testing.assert_allclose(matrix[:3, 1], y_direction, atol=1e-12)
        np.testing.assert_allclose(matrix[:3, 2], z_reference, atol=1e-12)


class TestPreviewShapes:
    def test_triangles_cover_the_faces(self):
        vertices, faces = ext.box((0.0, 1.0), (0.0, 2.0), (0.0, 3.0))
        tris = ext.triangles((vertices, faces))
        assert tris.shape == (12, 3, 3)
        area = sum(0.5 * np.linalg.norm(np.cross(b - a, c - a)) for a, b, c in tris)
        assert area == pytest.approx(2 * (1 * 2 + 2 * 3 + 1 * 3))

    def test_edges_are_counted_once(self):
        assert ext.edges(ext.box((0, 1), (0, 1), (0, 1))).shape == (12, 2, 3)

    def test_a_linear_travel_reaches_both_end_stops(self):
        spec = ext.AxisSpec("a", lower=-0.5, upper=1.25, size=0.2)
        lines, labels = ext.travel(spec)
        along = lines[..., 1]
        # The carriage's outline reaches past each stop by half its own length.
        assert along.min() == pytest.approx(-0.5 - 0.35 * 0.2)
        assert along.max() == pytest.approx(1.25 + 0.35 * 0.2)
        assert [text for _, text in labels] == ["-0.50 m", "+1.25 m"]
        assert [point[1] for point, _ in labels] == pytest.approx([-0.5, 1.25])

    def test_a_rotary_travel_is_an_arc_from_stop_to_stop(self):
        spec = ext.AxisSpec("a", kind="ROTARY", lower=0.0, upper=math.pi / 2, size=0.4)
        lines, labels = ext.travel(spec, segments=8)
        arc = lines[:8]
        radius = 0.55 * 0.4
        np.testing.assert_allclose(np.linalg.norm(arc[..., [0, 2]], axis=-1), radius)
        np.testing.assert_allclose(arc[0, 0], [radius, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(arc[-1, 1], [0.0, 0.0, -radius], atol=1e-12)
        assert [text for _, text in labels] == ["+0°", "+90°"]

    def test_a_continuous_one_is_a_full_circle_with_no_stops(self):
        spec = ext.AxisSpec("a", kind="ROTARY", continuous=True)
        lines, labels = ext.travel(spec, segments=12)
        assert labels == []
        assert len(lines) == 12
        np.testing.assert_allclose(lines[0, 0], lines[-1, 1], atol=1e-12)


class TestClustering:
    def _sphere(self, rings: int = 40):
        """A UV sphere of radius 1: many small triangles."""
        points = [(0.0, 0.0, 1.0)]
        for i in range(1, rings):
            theta = math.pi * i / rings
            for j in range(2 * rings):
                phi = math.pi * j / rings
                points.append((math.sin(theta) * math.cos(phi),
                               math.sin(theta) * math.sin(phi), math.cos(theta)))
        points.append((0.0, 0.0, -1.0))
        ring = 2 * rings
        tris = [(0, 1 + j, 1 + (j + 1) % ring) for j in range(ring)]
        for i in range(rings - 2):
            for j in range(ring):
                a, b = 1 + i * ring + j, 1 + i * ring + (j + 1) % ring
                tris += [(a, a + ring, b), (b, a + ring, b + ring)]
        last = len(points) - 1
        tris += [(last, 1 + (rings - 2) * ring + (j + 1) % ring, 1 + (rings - 2) * ring + j)
                 for j in range(ring)]
        return np.array(points), np.array(tris)

    def test_it_has_fewer_triangles_and_keeps_the_shape(self):
        points, tris = self._sphere()
        cell = 0.25
        merged, kept = ext.cluster_triangles(points, tris, cell)
        assert len(kept) < len(tris) / 4
        assert len(merged) < len(points)
        np.testing.assert_allclose(merged.min(axis=0), points.min(axis=0), atol=cell)
        np.testing.assert_allclose(merged.max(axis=0), points.max(axis=0), atol=cell)
        assert kept.max() < len(merged)

    def test_no_triangle_is_collapsed_or_repeated(self):
        points, tris = self._sphere()
        _, kept = ext.cluster_triangles(points, tris, 0.3)
        assert all(len(set(t)) == 3 for t in kept.tolist())
        assert len({tuple(sorted(t)) for t in kept.tolist()}) == len(kept)

    def test_a_fine_grid_changes_nothing(self):
        points, tris = self._sphere(8)
        merged, kept = ext.cluster_triangles(points, tris, 1e-6)
        assert len(kept) == len(tris)
        assert len(merged) == len(points)

    def test_nothing_in_nothing_out(self):
        merged, kept = ext.cluster_triangles(np.empty((0, 3)), np.empty((0, 3)), 0.1)
        assert merged.shape == (0, 3) and kept.shape == (0, 3)
