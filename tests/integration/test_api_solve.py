"""kinema.api (#102): solving, capsules and obstacles, clearance, Reachable?. Needs a real ``bpy``.

The claims:

* a pose of the tool, in world space, is solved for without moving the rig, and
  the values found stand the tool there -- with the rig moved in the world, its
  Root posed, or a joint held;
* the bones stand at given joint values as the rig would stand them;
* capsules and obstacles read as fitted and added, and obstacles are added and
  removed;
* the clearance at given joint values is the clearance the rig has there;
* Reachable? says whether the tool can stand at a pose, within limits and clear.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from ..conftest import requires_bpy

pytestmark = requires_bpy

#: A pose arm6 can stand in, away from its rest.
POSED = np.array([0.4, -0.5, 1.1, 0.3, 0.6, -0.2])


@pytest.fixture
def api(addon):
    return importlib.import_module(f"{addon.__name__}.api")


@pytest.fixture
def mods(addon):
    return lambda name: importlib.import_module(f"{addon.__name__}.{name}")


def _build(source):
    import bpy

    assert "FINISHED" in bpy.ops.kinema.build_robot(filepath=str(source))
    return bpy.context.view_layer.objects.active


@pytest.fixture
def arm6(api, mods, fixture_dir, clean_scene):
    rig = _build(fixture_dir / "arm6.urdf")
    rig.kinema_solver_mode = mods("solver.manager").MODE_NUMPY  # The same rule; no compile.
    return rig


def _cube(name, location, size=0.05):
    import bpy

    mesh = bpy.data.meshes.new(name)
    corners = [(x * size, y * size, z * size) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    mesh.from_pydata(corners, [], faces)
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.update()
    return obj


def _fit(rig):
    import bpy

    bpy.context.view_layer.objects.active = rig
    assert "FINISHED" in bpy.ops.kinema.fit_capsules()


def _pose_at(api, rig, values):
    """The tool's pose with the joints at ``values``, and the rig put back."""
    before = api.joint_values(rig)
    api.set_joint_values(rig, values)
    pose = api.tool_frame(rig)
    api.set_joint_values(rig, before)
    return pose


class TestBoneMatrices:
    @pytest.mark.parametrize("fixture", ["arm6.urdf", "rail6.urdf"])
    def test_the_bones_stand_where_the_rig_would(
        self, api, mods, fixture_dir, clean_scene, fixture
    ):
        """A prismatic rail too, and Root posed: every bone, not only the tool."""
        import bpy
        from mathutils import Vector

        rig = _build(fixture_dir / fixture)
        rig.pose.bones["Root"].location = Vector((0.1, -0.2, 0.05))
        bpy.context.view_layer.update()
        rng = np.random.default_rng(7)
        values = rng.uniform(-0.6, 0.6, len(api.joints(rig)))
        found = mods("ops.pose").bone_matrices(rig, values)
        api.set_joint_values(rig, values)
        for bone in rig.pose.bones:
            np.testing.assert_allclose(found[bone.name], np.array(bone.matrix), atol=1e-5,
                                       err_msg=bone.name)


class TestSolve:
    def test_a_pose_is_solved_for_without_moving_the_rig(self, api, arm6):
        pose = _pose_at(api, arm6, POSED)
        before, tool_before = api.joint_values(arm6), api.tool_frame(arm6)
        solution = api.solve(arm6, pose)
        assert solution.reached, solution
        np.testing.assert_allclose(api.joint_values(arm6), before)
        np.testing.assert_allclose(api.tool_frame(arm6), tool_before)
        api.set_joint_values(arm6, solution.values)
        np.testing.assert_allclose(api.tool_frame(arm6), pose, atol=1e-4)

    def test_with_the_rig_moved_and_its_root_posed(self, api, arm6):
        """World space in, world space out: the rig's object and Root taken out."""
        import bpy
        from mathutils import Euler, Vector

        arm6.location = Vector((1.0, -2.0, 0.5))
        arm6.rotation_euler = Euler((0.0, 0.0, 0.7))
        arm6.pose.bones["Root"].location = Vector((0.05, 0.0, 0.1))
        bpy.context.view_layer.update()
        pose = _pose_at(api, arm6, POSED)
        solution = api.solve(arm6, pose)
        assert solution.reached, solution
        api.set_joint_values(arm6, solution.values)
        np.testing.assert_allclose(api.tool_frame(arm6), pose, atol=1e-4)

    def test_on_pyroki(self, api, mods, arm6):
        """The link the description's solver aims at, not the marker bone."""
        arm6.kinema_solver_mode = mods("solver.manager").MODE_PYROKI
        pose = _pose_at(api, arm6, POSED)
        solution = api.solve(arm6, pose)
        assert solution.reached, solution
        api.set_joint_values(arm6, solution.values)
        np.testing.assert_allclose(api.tool_frame(arm6), pose, atol=1e-4)

    def test_a_pose_out_of_reach_is_not_reached(self, api, arm6):
        pose = np.eye(4)
        pose[:3, 3] = (5.0, 0.0, 0.0)
        solution = api.solve(arm6, pose)
        assert not solution.reached
        assert solution.position_error > 3.0

    def test_a_held_joint_stays_at_its_seed(self, api, arm6):
        first = api.joints(arm6)[0].bone
        arm6.pose.bones[first].kinema_ik_hold = True
        seed = np.array([0.25, 0.0, 0.5, 0.0, 0.5, 0.0])
        solution = api.solve(arm6, _pose_at(api, arm6, POSED), seed=seed)
        assert solution.values[0] == pytest.approx(0.25)

    def test_what_isnt_a_pose_is_refused(self, api, arm6):
        with pytest.raises(api.Error, match="4x4"):
            api.solve(arm6, np.eye(3))
        with pytest.raises(api.Error, match="has 6 joints"):
            api.solve(arm6, np.eye(4), seed=[0.0] * 4)


class TestObstacles:
    def test_added_read_and_removed(self, api, arm6):
        import bpy

        cube = _cube("Crate", (0.6, 0.4, 0.2))
        api.add_obstacle(cube, shape="SPHERE")
        (found,) = api.obstacles()
        assert (found.name, found.shape) == ("Crate", "SPHERE")
        np.testing.assert_allclose(found.frame[:3, 3], (0.6, 0.4, 0.2), atol=1e-6)
        assert found.size[0] == pytest.approx(np.sqrt(3) * 0.05, rel=1e-3)
        # In the scene only as an obstacle, as one made in the obstacles collection is.
        bpy.context.scene.collection.objects.unlink(cube)
        assert cube.name in bpy.context.scene.objects, "precondition: there as an obstacle"
        api.remove_obstacle(cube)
        assert api.obstacles() == []
        assert cube.name in bpy.context.scene.objects, "kept in the scene"

    def test_a_robot_or_a_bad_shape_is_refused(self, api, arm6):
        with pytest.raises(api.Error, match="is a robot"):
            api.add_obstacle(arm6)
        with pytest.raises(api.Error, match="one of BOX, SPHERE, FLOOR"):
            api.add_obstacle(_cube("Crate", (1, 1, 1)), shape="CONE")


class TestClearance:
    def test_capsules_as_fitted(self, api, arm6):
        assert api.capsules(arm6) == []
        _fit(arm6)
        found = api.capsules(arm6)
        assert found and all(c.bone in arm6.pose.bones and c.radius > 0 for c in found)

    def test_nothing_to_measure_is_infinitely_clear(self, api, arm6):
        assert api.clearance(arm6).distance == float("inf")

    def test_at_given_values_as_the_rig_would_have_it_there(self, api, arm6):
        _fit(arm6)
        api.add_obstacle(_cube("Post", (0.55, 0.25, 0.3)))
        at_values = api.clearance(arm6, POSED)
        here = api.clearance(arm6)
        assert at_values.distance != pytest.approx(here.distance), "precondition: differs"
        api.set_joint_values(arm6, POSED)
        now = api.clearance(arm6)
        assert at_values.distance == pytest.approx(now.distance, abs=1e-6)
        assert (at_values.bone, at_values.obstacle) == (now.bone, now.obstacle)


class TestReachable:
    def test_a_pose_it_can_stand_in(self, api, arm6):
        reach = api.reachable(arm6, _pose_at(api, arm6, POSED))
        assert reach.ok and reach.clearance.distance == float("inf")

    def test_an_obstacle_where_the_arm_would_be(self, api, arm6):
        """On the forearm as it would stand at the pose, not where it stands now."""
        _fit(arm6)
        pose = _pose_at(api, arm6, POSED)
        (capsule, *_) = (c for c in api.capsules(arm6) if c.bone == api.joints(arm6)[2].bone)
        api.set_joint_values(arm6, POSED)
        bone = np.array(arm6.matrix_world) @ np.array(arm6.pose.bones[capsule.bone].matrix)
        middle = bone[:3, :3] @ ((capsule.a + capsule.b) / 2) + bone[:3, 3]
        api.set_joint_values(arm6, np.zeros(6))
        api.add_obstacle(_cube("Crate", tuple(middle), size=0.05))
        assert api.clearance(arm6).distance > 0, "precondition: clear where it stands now"
        reach = api.reachable(arm6, pose)
        assert reach.reachable and reach.within_limits
        assert not reach.clear and not reach.ok
        assert reach.clearance.distance < 0 and reach.clearance.obstacle == "Crate"

    def test_kept_clear_by_the_distance_asked(self, api, arm6):
        _fit(arm6)
        pose = _pose_at(api, arm6, POSED)
        far = pose[:3, 3] + (0.0, 0.0, 0.5)
        api.add_obstacle(_cube("Lamp", tuple(far), size=0.05))
        reach = api.reachable(arm6, pose)
        assert reach.clear, "precondition: clear by some distance"
        assert not api.reachable(arm6, pose, keep_clear=reach.clearance.distance + 0.01).clear

    def test_a_joint_held_past_its_range(self, api, arm6):
        first = api.joints(arm6)[0]
        arm6.pose.bones[first.bone].kinema_ik_hold = True
        seed = np.zeros(6)
        seed[0] = first.upper + 0.2
        reach = api.reachable(arm6, _pose_at(api, arm6, POSED), seed=seed)
        assert not reach.within_limits and not reach.ok


class TestReviewed:
    def test_results_carrying_arrays_compare_and_hash_as_themselves(self, api, arm6):
        _fit(arm6)
        api.add_obstacle(_cube("Crate", (0.6, 0.4, 0.2)))
        first, again = api.obstacles(), api.obstacles()
        assert first != again and first == first  # by identity, not raising
        reach = api.reachable(arm6, _pose_at(api, arm6, POSED))
        assert len({reach, *first, *api.capsules(arm6), api.solve(arm6, np.eye(4))}) >= 3

    @pytest.mark.parametrize("beyond, reached", [(0.0005, True), (0.003, False)])
    def test_reached_within_a_millimetre(self, api, arm6, beyond, reached):
        """Stretched straight out, the tool can't go further: just beyond is as near
        as it gets."""
        pose = api.tool_frame(arm6)
        reach = pose[:3, 3] - np.array(arm6.matrix_world.translation)
        pose[:3, 3] += reach / np.linalg.norm(reach) * beyond
        solution = api.solve(arm6, pose)
        assert solution.position_error > 0.0004, "precondition: out of reach"
        assert solution.reached is reached, solution

    def test_a_floor_has_no_size(self, api, arm6):
        import bpy

        plane = _cube("Floor", (0.0, 0.0, 0.0))
        api.add_obstacle(plane, shape="FLOOR")
        (found,) = api.obstacles()
        assert (found.shape, found.size) == ("FLOOR", ())
        assert plane.name in bpy.context.scene.objects

    def test_a_held_joint_past_its_limit_is_seeded_where_it_shows(self, api, arm6):
        """Its channel past the range, its limit constraint holding it at the end. The
        shoulder's: the base's range is nearly a full turn, and past it is inside it."""
        import bpy

        shoulder = api.joints(arm6)[1]
        bone = arm6.pose.bones[shoulder.bone]
        bone.kinema_ik_hold = True
        bone.rotation_euler[1] = shoulder.upper + 0.3
        bpy.context.view_layer.update()
        solution = api.solve(arm6, _pose_at(api, arm6, POSED))
        assert solution.values[1] == pytest.approx(shoulder.upper, abs=1e-4)
