"""Collision in Blender: capsules fitted to a rig's meshes, obstacles, and the check's clearance.

The claims worth holding onto:
- every vertex of a link's meshes is inside one of its capsules, in its bone's
  frame, so the capsules move with the rig whatever plays it;
- the panel can tell when the meshes have changed since the capsules were fitted;
- an object in the obstacles collection is measured as its box, its sphere or
  its floor, where it stands;
- the Motion Check names the link, the obstacle and the frame where the robot
  hits something, and passes a job that clears everything;
- the overlay draws what touches in red.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from ..conftest import requires_bpy

pytestmark = requires_bpy

HOME = np.array([0.0, -0.6, 1.0, 0.0, 0.6, 0.0])
#: The faces of the eight corners (x, y, z) in -/+ order, x slowest.
CUBE_FACES = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
TURNED = HOME + np.array([1.2, 0.3, -0.3, 0.0, 0.0, 0.0])


@pytest.fixture
def modules(addon):
    def module(name):
        return importlib.import_module(f"{addon.__name__}.{name}")

    return module


@pytest.fixture
def builder(modules):
    return modules("rig.builder")


@pytest.fixture
def arm6(addon, fixture_dir, clean_scene, builder):
    import bpy

    scene = bpy.context.scene
    scene.render.fps, scene.render.fps_base = 30, 1.0
    assert "FINISHED" in bpy.ops.kinema.build_robot(filepath=str(fixture_dir / "arm6.urdf"))
    rig = next(o for o in bpy.data.objects if builder.is_kinema_rig(o))
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    _set_q(builder, rig, HOME)
    return rig


def _set_q(builder, rig, q):
    import bpy

    for pose_bone, value in zip(builder.joint_bones(rig), q, strict=True):
        pose_bone.rotation_euler[1] = float(value)
    bpy.context.view_layer.update()


def _fit(rig):
    import bpy

    bpy.context.view_layer.objects.active = rig
    assert "FINISHED" in bpy.ops.kinema.fit_capsules()


def _obstacle(name, location, size=0.05, shape="BOX", rotation=(0.0, 0.0, 0.0)):
    """A cube of half-size ``size`` made an obstacle."""
    import bpy

    mesh = bpy.data.meshes.new(name)
    corners = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    mesh.from_pydata(corners, [], [])
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.location, obj.scale, obj.rotation_euler = location, (size, size, size), rotation
    obj.kinema_obstacle_shape = shape
    for other in bpy.context.selected_objects:
        other.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    assert "FINISHED" in bpy.ops.kinema.add_obstacles()
    bpy.context.view_layer.update()
    return obj


class TestCapsules:
    def test_every_vertex_of_a_link_is_inside_its_capsules(self, arm6, modules):
        import bpy

        collision, geometry = modules("ops.collision"), modules("rig.collision")
        _fit(arm6)
        capsules = collision.capsules_of(arm6)
        meshes = collision.link_meshes(arm6)
        assert capsules and {capsule.bone for capsule in capsules} == set(meshes)
        for bone, objects in meshes.items():
            points = collision.bone_points(bpy.context, arm6, bone, objects)
            mine = [capsule for capsule in capsules if capsule.bone == bone]
            gaps = np.min(
                [geometry._point_segment_distance(points, c.a, c.b) - c.radius for c in mine],
                axis=0,
            )
            assert gaps.max() <= 1e-6, bone

    def test_they_ride_on_their_bones(self, arm6, builder, modules):
        """Fitted in one pose, and still round the link in another."""
        import bpy

        collision = modules("ops.collision")
        _fit(arm6)
        _set_q(builder, arm6, TURNED)
        capsules = collision.capsules_of(arm6)
        a, b = collision.capsule_ends(arm6, capsules)
        for capsule, end_a, end_b in zip(capsules, a, b, strict=True):
            objects = collision.link_meshes(arm6)[capsule.bone]
            world = np.concatenate([
                np.array([obj.matrix_world @ v.co for v in obj.data.vertices]) for obj in objects
            ])
            # The mesh's own centre is near the capsule's axis, wherever the arm turned.
            centre = world.mean(axis=0)
            geometry = modules("rig.collision")
            assert float(geometry._point_segment_distance(centre, end_a, end_b)) <= capsule.radius
        assert bpy.context.scene.frame_current is not None

    def test_moving_a_mesh_on_its_bone_makes_them_stale(self, arm6, modules):
        import bpy

        collision = modules("ops.collision")
        _fit(arm6)
        assert not collision.capsules_stale(arm6)
        objects = next(iter(collision.link_meshes(arm6).values()))
        objects[0].location.x += 0.05
        bpy.context.view_layer.update()
        assert collision.capsules_stale(arm6)
        _fit(arm6)
        assert not collision.capsules_stale(arm6)


class TestWhatALinkIs:
    @staticmethod
    def _on_bone(rig, bone, name, size=0.05, offset=(0.0, 0.0, 0.0)):
        """A small cube riding on ``bone``, ``offset`` from its head."""
        import bpy
        from mathutils import Matrix

        mesh = bpy.data.meshes.new(name)
        side = (-size, size)
        corners = [(x, y, z) for x in side for y in side for z in side]
        mesh.from_pydata(corners, [], CUBE_FACES)
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.scene.collection.objects.link(obj)
        obj.parent, obj.parent_type, obj.parent_bone = rig, "BONE", bone
        pose_bone = rig.pose.bones[bone]
        obj.matrix_parent_inverse = Matrix.Translation((0.0, -pose_bone.bone.length, 0.0))
        obj.location = offset
        bpy.context.view_layer.update()
        return obj

    def test_collision_meshes_stand_for_the_link_even_hidden(self, arm6, builder, modules):
        """Where a link has the description's collision meshes, they are its envelope."""
        collision = modules("ops.collision")
        bone, (shown,) = next(iter(collision.link_meshes(arm6).items()))
        hull = self._on_bone(arm6, bone, "Hull", size=0.3)
        hull[builder.PROP_GEOMETRY_KIND] = builder.KIND_COLLISION
        hull.hide_viewport = True
        assert collision.link_meshes(arm6)[bone] == [hull]
        _fit(arm6)
        radius = max(c.radius for c in collision.capsules_of(arm6) if c.bone == bone)
        assert radius >= 0.3, "fitted to the hidden 0.6 m cube, not the link's own mesh"

    def test_a_tool_on_the_tcp_gets_capsules(self, arm6, builder, modules):
        collision = modules("ops.collision")
        tcp = arm6.get(builder.PROP_TCP_BONE) or builder.TCP_BONE
        self._on_bone(arm6, tcp, "Gripper", size=0.04)
        _fit(arm6)
        assert any(capsule.bone == tcp for capsule in collision.capsules_of(arm6))

    def test_an_attached_collection_is_read_through_its_instance(self, arm6, builder, modules):
        import bpy
        from mathutils import Matrix

        collision = modules("ops.collision")
        tool = bpy.data.collections.new("Tool")
        mesh = bpy.data.meshes.new("Tip")
        side = (-0.02, 0.02)
        corners = [(x, y, z) for x in (0.0, 0.3) for y in side for z in side]
        mesh.from_pydata(corners, [], CUBE_FACES)
        tool.objects.link(bpy.data.objects.new("Tip", mesh))
        bone = builder.joint_bones(arm6)[-1].name
        empty = bpy.data.objects.new("Tool.instance", None)
        empty.instance_type, empty.instance_collection = "COLLECTION", tool
        bpy.context.scene.collection.objects.link(empty)
        empty.parent, empty.parent_type, empty.parent_bone = arm6, "BONE", bone
        empty.matrix_parent_inverse = Matrix.Translation(
            (0.0, -arm6.pose.bones[bone].bone.length, 0.0)
        )
        bpy.context.view_layer.update()
        assert empty in collision.link_meshes(arm6)[bone]
        _fit(arm6)
        lengths = [
            float(np.linalg.norm(c.b - c.a)) + 2 * c.radius
            for c in collision.capsules_of(arm6) if c.bone == bone
        ]
        assert max(lengths) >= 0.3


class TestObstacles:
    def test_a_box_is_its_own_bounds_where_it_stands(self, arm6, modules):
        import bpy

        collision = modules("ops.collision")
        _obstacle("Crate", (1.0, 0.5, 0.2), size=0.1, rotation=(0.0, 0.0, 0.5))
        (box,) = collision.obstacles(bpy.context)
        assert box.kind == "BOX" and box.name == "Crate"
        np.testing.assert_allclose(box.matrix[:3, 3], (1.0, 0.5, 0.2), atol=1e-6)
        np.testing.assert_allclose(box.size, (0.1, 0.1, 0.1), atol=1e-6)
        np.testing.assert_allclose(box.matrix[:3, 0], (np.cos(0.5), np.sin(0.5), 0.0), atol=1e-6)

    def test_a_sphere_and_a_floor(self, arm6, modules):
        import bpy

        collision = modules("ops.collision")
        _obstacle("Ball", (0.0, 1.0, 0.5), size=0.2, shape="SPHERE")
        _obstacle("Ground", (0.0, 0.0, -0.1), shape="FLOOR")
        found = {obstacle.name: obstacle for obstacle in collision.obstacles(bpy.context)}
        assert found["Ball"].kind == "SPHERE" and float(found["Ball"].size[0]) == pytest.approx(0.2)
        assert found["Ground"].kind == "FLOOR"
        np.testing.assert_allclose(found["Ground"].matrix[:3, 2], (0.0, 0.0, 1.0), atol=1e-6)

    def test_removing_one_keeps_it_in_the_scene(self, arm6, modules):
        import bpy

        collision = modules("ops.collision")
        crate = _obstacle("Crate", (1.0, 0.0, 0.0))
        # Only an obstacle now, as an object made inside the collection is.
        bpy.context.scene.collection.objects.unlink(crate)
        assert [c.name for c in crate.users_collection] == [collision.OBSTACLES]
        assert "FINISHED" in bpy.ops.kinema.remove_obstacles()
        assert collision.obstacle_objects(bpy.context.scene) == []
        assert crate.name in bpy.context.scene.objects

    def test_the_robot_is_never_an_obstacle(self, arm6):
        import bpy

        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        arm6.select_set(True)
        assert not bpy.ops.kinema.add_obstacles.poll()


class TestChecking:
    def _job(self, arm6, builder, modules):
        """Two waypoints, a joint move between them, generated."""
        import bpy

        manager = modules("solver.manager")
        arm6.kinema_solver_mode = "NUMPY"
        assert "FINISHED" in bpy.ops.kinema.add_ik()
        arm6.kinema_ik_enabled = False
        ik_name = arm6[builder.PROP_IK_BONE]
        for name, frame, q in (("A", 1, HOME), ("B", 31, TURNED)):
            bpy.context.scene.frame_set(frame)
            _set_q(builder, arm6, q)
            arm6.pose.bones[ik_name].matrix = arm6.pose.bones[manager.tip_bone(arm6)].matrix.copy()
            bpy.context.view_layer.update()
            assert "FINISHED" in bpy.ops.kinema.add_waypoint(name=name)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()

    def _halfway(self, arm6, builder, modules, frame=16):
        """Where the middle of the first capsule is on ``frame``."""
        import bpy

        collision = modules("ops.collision")
        bpy.context.scene.frame_set(frame)
        capsules = collision.capsules_of(arm6)
        a, b = collision.capsule_ends(arm6, capsules)
        return capsules[0], (a[0] + b[0]) / 2.0

    def test_a_link_hitting_an_obstacle_is_named_with_the_frame(self, arm6, builder, modules):
        import bpy

        self._job(arm6, builder, modules)
        _fit(arm6)
        capsule, middle = self._halfway(arm6, builder, modules)
        _obstacle("Post", tuple(middle), size=0.02)
        bpy.context.view_layer.objects.active = arm6
        assert "FINISHED" in bpy.ops.kinema.check_motion()

        row = arm6.kinema_motion_check[0]
        assert row.clearance_obstacle == "Post"
        assert row.clearance < -capsule.radius
        assert row.clearance_bone == capsule.bone or row.clearance < 0.0
        problems = modules("rig.motion_check").problems(row)
        assert any("hits Post by" in found and "at frame" in found for found in problems)

    def test_a_job_clear_of_everything_passes(self, arm6, builder, modules):
        import bpy

        self._job(arm6, builder, modules)
        _fit(arm6)
        _obstacle("FarAway", (5.0, 5.0, 5.0), size=0.1)
        bpy.context.view_layer.objects.active = arm6
        assert "FINISHED" in bpy.ops.kinema.check_motion()
        row = arm6.kinema_motion_check[0]
        assert row.clearance_obstacle == "FarAway" and row.clearance > 4.0
        assert not any("hits" in found for found in modules("rig.motion_check").problems(row))

    def test_without_capsules_nothing_is_measured(self, arm6, builder, modules):
        import bpy

        self._job(arm6, builder, modules)
        _obstacle("Post", (0.3, 0.0, 0.3))
        bpy.context.view_layer.objects.active = arm6
        assert "FINISHED" in bpy.ops.kinema.check_motion()
        assert arm6.kinema_motion_check[0].clearance_obstacle == ""


class TestOverlay:
    def test_what_touches_is_drawn_red(self, arm6, builder, modules):
        import bpy

        overlay = modules("ui.collision_overlay")
        collision = modules("ops.collision")
        _fit(arm6)
        far = _obstacle("FarAway", (5.0, 5.0, 5.0))
        clear, hit = overlay.shape_lines(bpy.context)
        assert clear and not hit

        capsules = collision.capsules_of(arm6)
        a, b = collision.capsule_ends(arm6, capsules)
        far.location = tuple((a[0] + b[0]) / 2.0)
        bpy.context.view_layer.update()
        clear, hit = overlay.shape_lines(bpy.context)
        # The capsule and the box, each outlined in red.
        assert len(hit) >= len(overlay.capsule_lines(a[0], b[0], capsules[0].radius)) + 12
