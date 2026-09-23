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
