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
        collision = modules("ops.collision")
        _fit(arm6)
        _set_q(builder, arm6, TURNED)
        capsules = collision.capsules_of(arm6)
        a, b, _ = collision.capsule_ends(arm6, capsules)
        for capsule, end_a, end_b in zip(capsules, a, b, strict=True):
            objects = collision.link_meshes(arm6)[capsule.bone]
            world = np.concatenate([
                np.array([obj.matrix_world @ v.co for v in obj.data.vertices]) for obj in objects
            ])
            # The mesh's own centre is near the capsule's axis, wherever the arm turned.
            centre = world.mean(axis=0)
            geometry = modules("rig.collision")
            assert float(geometry._point_segment_distance(centre, end_a, end_b)) <= capsule.radius

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

    def test_posing_the_arm_is_no_change(self, arm6, builder, modules):
        """Playing the job moves every mesh, and changes none of them."""
        collision = modules("ops.collision")
        _fit(arm6)
        for q in (TURNED, HOME * 0.5, TURNED * -0.7):
            _set_q(builder, arm6, q)
            assert not collision.capsules_stale(arm6)

    @pytest.mark.parametrize("muted", [False, True])
    def test_the_same_meshes_fit_the_same_in_any_pose(self, arm6, builder, modules, muted):
        """Fitted in the bone's frame: where the arm stands changes nothing, to the bit.

        A muted constraint on a link moves nothing, and changes nothing either.
        """
        collision = modules("ops.collision")
        if muted:
            for objects in collision.link_meshes(arm6).values():
                objects[0].constraints.new("COPY_LOCATION").mute = True
        fitted = []
        for q in (HOME, TURNED):
            _set_q(builder, arm6, q)
            _fit(arm6)
            fitted.append([
                (c.bone, tuple(c.a), tuple(c.b), c.radius) for c in collision.capsules_of(arm6)
            ])
        assert fitted[0] == fitted[1]

    def test_clear_capsules(self, arm6, modules):
        """Nothing fitted, drawn or fingerprinted for the robot afterwards."""
        import bpy

        collision, overlay = modules("ops.collision"), modules("ui.collision_overlay")
        _fit(arm6)
        assert len(overlay.shape_lines(bpy.context)[0])
        assert "FINISHED" in bpy.ops.kinema.clear_capsules()
        assert not len(arm6.kinema_capsules) and arm6.kinema_capsules_source == ""
        assert not collision.capsules_stale(arm6)
        assert not len(overlay.shape_lines(bpy.context)[0])
        assert not bpy.ops.kinema.clear_capsules.poll()

    def test_staleness_is_worked_out_once_per_change(self, arm6, modules, monkeypatch):
        """The panels ask on every redraw, several times: the meshes are fingerprinted once."""
        import bpy

        collision = modules("ops.collision")
        _fit(arm6)
        calls = []
        signature = collision.source_signature
        monkeypatch.setattr(
            collision, "source_signature", lambda rig: calls.append(1) or signature(rig)
        )
        for _ in range(5):
            assert not collision.capsules_stale(arm6)
        assert len(calls) == 1
        objects = next(iter(collision.link_meshes(arm6).values()))
        objects[0].location.x += 0.05
        bpy.context.view_layer.update()
        assert collision.capsules_stale(arm6) and collision.capsules_stale(arm6)
        assert len(calls) == 2

    def test_removing_a_face_makes_them_stale(self, arm6, modules):
        """The same vertices, one face fewer: the surface the fit samples changed."""
        import bmesh
        import bpy

        collision = modules("ops.collision")
        mesh = next(iter(collision.link_meshes(arm6).values()))[0].data
        assert len(mesh.polygons) > 1
        _fit(arm6)
        vertices = len(mesh.vertices)
        shape = bmesh.new()
        shape.from_mesh(mesh)
        shape.faces.ensure_lookup_table()
        bmesh.ops.delete(shape, geom=[shape.faces[0]], context="FACES_ONLY")
        shape.to_mesh(mesh)
        shape.free()
        mesh.update()
        bpy.context.view_layer.update()
        assert len(mesh.vertices) == vertices
        assert collision.capsules_stale(arm6)

    @pytest.mark.parametrize("scale", [(2.0, 2.0, 2.0), (1.0, 1.0, 3.0)])
    def test_they_scale_with_the_robot(self, arm6, modules, scale):
        """As wide as the links they hold: a robot scaled up has thicker links."""
        import bpy

        collision = modules("ops.collision")
        _fit(arm6)
        capsules = collision.capsules_of(arm6)
        arm6.scale = scale
        bpy.context.view_layer.update()
        a, b, radii = collision.capsule_ends(arm6, capsules)
        assert radii == pytest.approx([c.radius * max(scale) for c in capsules])
        # Every vertex of every link still inside its capsules, as scaled.
        geometry = modules("rig.collision")
        for bone, objects in collision.link_meshes(arm6).items():
            world = np.concatenate([
                np.array([obj.matrix_world @ v.co for v in obj.data.vertices]) for obj in objects
            ])
            held = [index for index, capsule in enumerate(capsules) if capsule.bone == bone]
            gaps = np.min([
                geometry._point_segment_distance(world, a[i], b[i]) - radii[i] for i in held
            ], axis=0)
            assert gaps.max() <= 1e-5, bone

    def test_an_undone_reshape_reads_as_fitted_again(self, arm6, modules):
        """Undo puts a mesh back without Blender reporting its geometry changed."""
        import bpy

        collision, handlers = modules("ops.collision"), modules("handlers")
        _fit(arm6)
        mesh = next(iter(collision.link_meshes(arm6).values()))[0].data
        was = mesh.vertices[0].co.x
        mesh.vertices[0].co.x += 0.05
        mesh.update()
        bpy.context.view_layer.update()
        assert collision.capsules_stale(arm6)
        # Back as it was, as undo leaves it: no geometry update, then undo_post.
        mesh.vertices[0].co.x = was
        handlers.on_undone(bpy.context.scene)
        assert not collision.capsules_stale(arm6)

    @pytest.mark.parametrize(
        "what", ["a constraint", "a driver", "keys", "animated shape keys", "animated modifiers"]
    )
    def test_a_link_that_moves_on_its_bone_can_t_be_fitted(self, arm6, modules, what):
        """Capsules ride rigidly on their bone: they can't hold what moves on it."""
        import bpy

        collision = modules("ops.collision")
        _fit(arm6)
        link = next(iter(collision.link_meshes(arm6).values()))[0]
        if what == "a constraint":
            muted = link.constraints.new("COPY_LOCATION")
            muted.mute = True
            bpy.context.view_layer.update()
            # Muted, it moves nothing.
            assert collision.moving_on_bone(arm6) == []
            muted.mute = False
        elif what == "a driver":
            link.driver_add("location", 0).driver.expression = "0.2"
        elif what == "animated shape keys":
            link.shape_key_add(name="Basis")
            link.shape_key_add(name="Bulge").keyframe_insert("value", frame=1)
        elif what == "animated modifiers":
            link.modifiers.new("Twice", "ARRAY")
            link.keyframe_insert('modifiers["Twice"].count', frame=1)
        else:
            link.keyframe_insert("location", frame=1)
        bpy.context.view_layer.update()
        assert collision.moving_on_bone(arm6) == [(link.name, what)]
        assert collision.capsules_stale(arm6)
        count = len(arm6.kinema_capsules)
        bpy.context.view_layer.objects.active = arm6
        with pytest.raises(RuntimeError, match=link.name):
            bpy.ops.kinema.fit_capsules()
        assert len(arm6.kinema_capsules) == count

    def test_a_shape_key_changed_makes_them_stale(self, arm6, modules):
        """The fit reads a link as its shape keys leave it."""
        import bpy

        collision = modules("ops.collision")
        link = next(iter(collision.link_meshes(arm6).values()))[0]
        link.shape_key_add(name="Basis")
        bulge = link.shape_key_add(name="Bulge")
        for point in bulge.data:
            point.co *= 1.5
        bulge.value = 0.0
        bpy.context.view_layer.update()
        _fit(arm6)
        assert not collision.capsules_stale(arm6)
        bulge.value = 1.0
        bpy.context.view_layer.update()
        assert collision.capsules_stale(arm6)

    def test_reshaping_a_mesh_makes_them_stale(self, arm6, modules):
        """As Edit Mode does: the same vertices, as many, one of them moved."""
        import bpy

        collision = modules("ops.collision")
        _fit(arm6)
        mesh = next(iter(collision.link_meshes(arm6).values()))[0].data
        mesh.vertices[0].co.x += 0.05
        mesh.update()
        bpy.context.view_layer.update()
        assert collision.capsules_stale(arm6)


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
        hull[builder.PROP_LINK_NAME] = shown[builder.PROP_LINK_NAME]
        hull.hide_viewport = True
        assert collision.link_meshes(arm6)[bone] == [hull]
        _fit(arm6)
        radius = max(c.radius for c in collision.capsules_of(arm6) if c.bone == bone)
        assert radius >= 0.3, "fitted to the hidden 0.6 m cube, not the link's own mesh"

    def test_collision_meshes_stand_for_their_own_link_only(self, arm6, builder, modules):
        """A link behind a fixed joint rides the same bone, and may have none of its own."""
        collision = modules("ops.collision")
        bone, shown = next(iter(collision.link_meshes(arm6).items()))
        parts = {}
        for name, link, kind in (
            ("A.hull", "link_a", builder.KIND_COLLISION),
            ("A.shown", "link_a", builder.KIND_VISUAL),
            ("B.shown", "link_b", builder.KIND_VISUAL),
        ):
            parts[name] = self._on_bone(arm6, bone, name, size=0.05)
            parts[name][builder.PROP_LINK_NAME] = link
            parts[name][builder.PROP_GEOMETRY_KIND] = kind
        read = collision.link_meshes(arm6)[bone]
        assert parts["A.hull"] in read and parts["B.shown"] in read
        assert parts["A.shown"] not in read
        assert all(obj in read for obj in shown)

    def test_a_longer_bone_makes_them_stale(self, arm6, modules):
        """Its meshes hang from its tail: made longer in Edit Mode, they move with it."""
        import bpy

        collision = modules("ops.collision")
        bone = next(iter(collision.link_meshes(arm6)))
        _fit(arm6)
        bpy.context.view_layer.objects.active = arm6
        bpy.ops.object.mode_set(mode="EDIT")
        try:
            arm6.data.edit_bones[bone].length *= 1.5
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.context.view_layer.update()
        assert collision.capsules_stale(arm6)

    @staticmethod
    def _attached_collection(rig, builder, modifier=False):
        """A tool collection, not in the scene, attached to the last joint as an instance."""
        import bpy
        from mathutils import Matrix

        tool = bpy.data.collections.new("Tool")
        mesh = bpy.data.meshes.new("Tip")
        side = (-0.02, 0.02)
        corners = [(x, y, z) for x in (0.0, 0.3) for y in side for z in side]
        mesh.from_pydata(corners, [], CUBE_FACES)
        tip = bpy.data.objects.new("Tip", mesh)
        tool.objects.link(tip)
        if modifier:
            tip.modifiers.new("Twice", "ARRAY").count = 2
        bone = builder.joint_bones(rig)[-1].name
        empty = bpy.data.objects.new("Tool.instance", None)
        empty.instance_type, empty.instance_collection = "COLLECTION", tool
        bpy.context.scene.collection.objects.link(empty)
        empty.parent, empty.parent_type, empty.parent_bone = rig, "BONE", bone
        empty.matrix_parent_inverse = Matrix.Translation(
            (0.0, -rig.pose.bones[bone].bone.length, 0.0)
        )
        bpy.context.view_layer.update()
        return tool, tip, empty

    def test_a_moving_part_of_an_attached_collection_can_t_be_fitted(
        self, arm6, builder, modules
    ):
        """A finger keyed inside a gripper moves on the bone as much as the gripper would."""
        import bpy

        collision = modules("ops.collision")
        _fit(arm6)
        _tool, tip, empty = self._attached_collection(arm6, builder)
        tip.keyframe_insert("location", frame=1)
        bpy.context.view_layer.update()
        assert collision.moving_on_bone(arm6) == [(empty.name, "keys, on Tip")]
        bpy.context.view_layer.objects.active = arm6
        with pytest.raises(RuntimeError, match="Tip"):
            bpy.ops.kinema.fit_capsules()

    def test_a_part_s_modifiers_changed_makes_them_stale(self, arm6, builder, modules):
        """Shown only through the instance, it's read as the instance shows it."""
        import bpy

        collision = modules("ops.collision")
        _tool, tip, _empty = self._attached_collection(arm6, builder, modifier=True)
        assert not tip.visible_get()
        _fit(arm6)
        assert not collision.capsules_stale(arm6)
        tip.modifiers["Twice"].count = 3
        bpy.context.view_layer.update()
        assert collision.capsules_stale(arm6)

    def test_a_tool_attached_beside_collision_meshes_is_kept(self, arm6, builder, modules):
        """The collision meshes stand for what was imported, not for what was added."""
        collision = modules("ops.collision")
        bone, (shown,) = next(iter(collision.link_meshes(arm6).items()))
        hull = self._on_bone(arm6, bone, "Hull", size=0.3)
        hull[builder.PROP_GEOMETRY_KIND] = builder.KIND_COLLISION
        hull[builder.PROP_LINK_NAME] = shown[builder.PROP_LINK_NAME]
        tool = self._on_bone(arm6, bone, "Gripper", size=0.04, offset=(0.0, 0.0, 0.8))
        assert collision.link_meshes(arm6)[bone] == [hull, tool]
        _fit(arm6)
        signature = collision.source_signature(arm6)
        tool.location.z += 0.1
        assert collision.source_signature(arm6) != signature

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

        # Moving where the collection is instanced from moves the tool.
        assert not collision.capsules_stale(arm6)
        tool.instance_offset = (0.0, 0.0, 0.1)
        bpy.context.view_layer.update()
        assert collision.capsules_stale(arm6)

    def test_a_modifier_following_another_object_makes_them_stale(self, arm6, modules):
        """The link's mesh isn't touched: only what its object evaluates to changes."""
        import bpy

        collision = modules("ops.collision")
        link = next(iter(collision.link_meshes(arm6).values()))[0]
        handle = bpy.data.objects.new("Offset", None)
        bpy.context.scene.collection.objects.link(handle)
        array = link.modifiers.new("Follow", "ARRAY")
        array.use_relative_offset, array.use_object_offset = False, True
        array.offset_object = handle
        handle.location = (0.1, 0.0, 0.0)
        bpy.context.view_layer.update()
        _fit(arm6)
        assert not collision.capsules_stale(arm6)
        handle.location = (0.3, 0.0, 0.0)
        bpy.context.view_layer.update()
        assert collision.capsules_stale(arm6)

    def test_changing_a_link_s_modifiers_makes_them_stale(self, arm6, modules):
        """The fit reads a link as its modifiers leave it."""
        import bpy

        collision = modules("ops.collision")
        link = next(iter(collision.link_meshes(arm6).values()))[0]
        array = link.modifiers.new("Twice", "ARRAY")
        array.count = 2
        bpy.context.view_layer.update()
        _fit(arm6)
        assert not collision.capsules_stale(arm6)
        array.count = 3
        bpy.context.view_layer.update()
        assert collision.capsules_stale(arm6)


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
        # A cube taken for a ball: held whole, to its corners.
        assert found["Ball"].kind == "SPHERE"
        assert float(found["Ball"].size[0]) == pytest.approx(0.2 * np.sqrt(3.0))
        assert found["Ground"].kind == "FLOOR"
        np.testing.assert_allclose(found["Ground"].matrix[:3, 2], (0.0, 0.0, 1.0), atol=1e-6)

    @pytest.mark.parametrize("mirror", [(1.0, 1.0, 1.0), (-1.0, 1.0, 1.0), (1.0, -1.0, 1.0)])
    def test_a_mirrored_floor_is_solid_below_it(self, arm6, modules, mirror):
        """Mirrored in X or Y, it's the same floor: below its origin is inside."""
        import bpy

        collision, geometry = modules("ops.collision"), modules("rig.collision")
        ground = _obstacle("Ground", (0.0, 0.0, 0.0), shape="FLOOR")
        ground.scale = mirror
        bpy.context.view_layer.update()
        (floor,) = collision.obstacles(bpy.context)
        points = np.array([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]])
        found = geometry.clearances(points, points, np.zeros(2), [floor])[:, 0]
        np.testing.assert_allclose(found, [-0.5, 0.5], atol=1e-6)

    @pytest.mark.parametrize("shape", ["BOX", "SPHERE"])
    def test_a_sheared_obstacle_is_inside_what_is_measured(self, arm6, modules, shape):
        """Turned under a parent scaled unevenly, a cube is no longer a box: all of it counts."""
        import bpy

        collision, geometry = modules("ops.collision"), modules("rig.collision")
        parent = bpy.data.objects.new("Stretch", None)
        bpy.context.scene.collection.objects.link(parent)
        parent.scale = (2.0, 1.0, 1.0)
        turned = (0.0, 0.0, 0.785)
        cube = _obstacle("Sheared", (0.0, 0.0, 0.0), size=1.0, shape=shape, rotation=turned)
        cube.parent = parent
        bpy.context.view_layer.update()
        (obstacle,) = collision.obstacles(bpy.context)
        world = np.array([cube.matrix_world @ v.co for v in cube.data.vertices])
        if shape == "BOX":
            local = (world - obstacle.matrix[:3, 3]) @ obstacle.matrix[:3, :3]
            assert geometry.box_sdf(local, obstacle.size).max() <= 1e-6
        else:
            # Its corners, as it stands: further than any ball inside it reaches.
            reach = np.linalg.norm(world - obstacle.matrix[:3, 3], axis=1)
            assert reach.max() <= float(obstacle.size[0]) + 1e-6
            assert reach.max() > 2.4, "precondition: corners stretched past 2"

    def test_a_ball_is_measured_as_the_ball_it_is(self, arm6, modules):
        """Not as the box it fits in, whose corners reach 1.7 times as far."""
        import bpy

        collision = modules("ops.collision")
        directions = np.random.default_rng(1).normal(size=(400, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        mesh = bpy.data.meshes.new("Ball")
        mesh.from_pydata([tuple(d) for d in directions], [], [])
        ball = bpy.data.objects.new("Ball", mesh)
        bpy.context.scene.collection.objects.link(ball)
        ball.scale = (0.3, 0.3, 0.3)
        ball.kinema_obstacle_shape = "SPHERE"
        for other in bpy.context.selected_objects:
            other.select_set(False)
        ball.select_set(True)
        bpy.context.view_layer.objects.active = ball
        assert "FINISHED" in bpy.ops.kinema.add_obstacles()
        bpy.context.view_layer.update()
        (obstacle,) = collision.obstacles(bpy.context)
        assert float(obstacle.size[0]) == pytest.approx(0.3, abs=0.01)

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

    def test_only_the_collection_s_own_objects_are_obstacles(self, arm6, modules):
        """Remove Selected takes them out of it: one only there through another couldn't be."""
        import bpy

        collision = modules("ops.collision")
        crate = _obstacle("Crate", (1.0, 0.0, 0.0))
        fixtures = bpy.data.collections.new("Fixtures")
        collision.obstacle_collection(bpy.context.scene).children.link(fixtures)
        clamp = bpy.data.objects.new("Clamp", None)
        fixtures.objects.link(clamp)
        assert clamp.name in bpy.context.scene.objects
        assert collision.obstacle_objects(bpy.context.scene) == [crate]
        assert collision.nested_objects(bpy.context.scene) == 1
        # Added, it is one, and Remove Selected takes it out again.
        with bpy.context.temp_override(selected_objects=[clamp]):
            assert "FINISHED" in bpy.ops.kinema.add_obstacles()
            assert clamp in collision.obstacle_objects(bpy.context.scene)
            assert "FINISHED" in bpy.ops.kinema.remove_obstacles()
        assert collision.obstacle_objects(bpy.context.scene) == [crate]
        bpy.data.collections.remove(fixtures)

    def test_nothing_riding_on_the_robot_is_an_obstacle(self, arm6, modules):
        """However far down its parents, and however it came to ride there."""
        import bpy

        collision = modules("ops.collision")
        gripper = bpy.data.objects.new("Gripper", None)
        finger = bpy.data.objects.new("Finger", None)
        for obj in (gripper, finger):
            bpy.context.scene.collection.objects.link(obj)
        gripper.parent, finger.parent = arm6, gripper
        with bpy.context.temp_override(selected_objects=[finger]):
            assert not bpy.ops.kinema.add_obstacles.poll()
        crate = _obstacle("Crate", (1.0, 0.0, 0.0))
        assert collision.obstacle_objects(bpy.context.scene) == [crate]
        # Picked up after it was added.
        crate.parent = finger
        assert collision.obstacle_objects(bpy.context.scene) == []

    def test_removing_one_held_by_another_scene_keeps_it_in_this_one(self, arm6, modules):
        """Its other collection is another scene's: unlinked here, it would leave this one."""
        import bpy

        collision = modules("ops.collision")
        scene = bpy.context.scene
        crate = _obstacle("Crate", (1.0, 0.0, 0.0))
        scene.collection.objects.unlink(crate)
        other = bpy.data.scenes.new("Elsewhere")
        try:
            other.collection.objects.link(crate)
            assert len(crate.users_collection) == 2
            with bpy.context.temp_override(selected_objects=[crate]):
                assert "FINISHED" in bpy.ops.kinema.remove_obstacles()
            assert collision.obstacle_objects(scene) == []
            assert crate.name in scene.objects
        finally:
            bpy.data.scenes.remove(other)

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
        """The capsule furthest down the arm, and where its middle is on ``frame``.

        The forearm's: no other link's capsule reaches its middle.
        """
        import bpy

        collision = modules("ops.collision")
        bpy.context.scene.frame_set(frame)
        capsules = collision.capsules_of(arm6)
        order = [pose_bone.name for pose_bone in builder.joint_bones(arm6)]
        index = max(range(len(capsules)), key=lambda i: order.index(capsules[i].bone))
        a, b, _ = collision.capsule_ends(arm6, capsules)
        return capsules[index], (a[index] + b[index]) / 2.0

    @pytest.mark.parametrize("scale", [1.0, 2.0])
    def test_a_link_hitting_an_obstacle_is_named_with_the_frame(
        self, arm6, builder, modules, scale
    ):
        import bpy

        self._job(arm6, builder, modules)
        _fit(arm6)
        # A robot scaled up has links as much thicker: their capsules too.
        arm6.scale = (scale, scale, scale)
        bpy.context.view_layer.update()
        capsule, middle = self._halfway(arm6, builder, modules)
        _obstacle("Post", tuple(middle), size=0.02)
        bpy.context.view_layer.objects.active = arm6
        assert "FINISHED" in bpy.ops.kinema.check_motion()

        row = arm6.kinema_motion_check[0]
        # Deepest where the capsule's axis runs through the post's centre.
        assert (row.clearance_obstacle, row.clearance_bone, row.clearance_frame) == (
            "Post", capsule.bone, 16
        )
        assert row.clearance == pytest.approx(-(capsule.radius * scale + 0.02), abs=1e-4)
        depth = (capsule.radius * scale + 0.02) * 1000
        expected = f"{capsule.bone} hits Post by {depth:.0f} mm at frame 16"
        assert expected in modules("rig.motion_check").problems(row)

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


class TestThePanel:
    def test_the_closest_approach_is_shown_only_for_the_job_checked(self, arm6, builder, modules):
        import bpy

        collision = modules("ops.collision")
        TestChecking()._job(arm6, builder, modules)
        _fit(arm6)
        _obstacle("FarAway", (5.0, 5.0, 5.0), size=0.1)
        bpy.context.view_layer.objects.active = arm6
        assert "FINISHED" in bpy.ops.kinema.check_motion()
        scene = bpy.context.scene
        assert collision.closest_approach(arm6, scene).clearance_obstacle == "FarAway"

        arm6.kinema_waypoints[1].frame = 41
        assert collision.closest_approach(arm6, scene) is None

    def test_a_check_with_nothing_to_measure_is_never_out_of_date(self, arm6, builder, modules):
        """Nor one stored before checks kept a fingerprint of what they measured."""
        import bpy

        collision = modules("ops.collision")
        TestChecking()._job(arm6, builder, modules)
        bpy.context.view_layer.objects.active = arm6
        assert "FINISHED" in bpy.ops.kinema.check_motion()
        assert collision.check_measured_this(arm6, bpy.context.scene)
        arm6.kinema_motion_check_collision = ""
        assert collision.check_measured_this(arm6, bpy.context.scene)

    @pytest.mark.parametrize(
        "change",
        [
            "moved", "removed", "refitted", "scrubbed", "scaled while animated", "constrained",
            "capsules stale", "constraint retuned", "driver retuned", "constraint folded",
            "robot moved", "keys eased", "scrubbed past a driven one",
            "moved with its keys muted", "scrubbed through an NLA strip",
            "an NLA strip's key changed", "capsule changed",
        ],
    )
    def test_the_closest_approach_goes_with_what_was_measured(
        self, arm6, builder, modules, change
    ):
        """Obstacles since moved or removed, or capsules since fitted again, weren't checked."""
        import bpy

        collision = modules("ops.collision")
        TestChecking()._job(arm6, builder, modules)
        _fit(arm6)
        far = _obstacle("FarAway", (5.0, 5.0, 5.0), size=0.1)
        keyed = (
            "scrubbed", "scaled while animated", "keys eased", "moved with its keys muted",
            "scrubbed through an NLA strip", "an NLA strip's key changed",
        )
        if change in keyed:
            # It moves through the job: where the current frame puts it is no change.
            far.keyframe_insert("location", frame=1)
            far.location.x = 6.0
            far.keyframe_insert("location", frame=31)
        if change == "moved with its keys muted":
            # Keyed, but nothing plays them: it stands where it was put.
            for curves in modules("ops.ik").own_fcurve_containers(far):
                for curve in curves:
                    curve.mute = True
        if "NLA" in change:
            # Its keys pushed down into a strip: no active action left.
            data = far.animation_data
            action = data.action
            strip = data.nla_tracks.new().strips.new("Move", 1, action)
            data.action = None
            assert strip.action == action
            assert collision._strip_curves(far)[0][2], "the strip's curves are read"
        if change in ("constraint retuned", "constraint folded"):
            # It follows an empty out of the way: its own location is offset from it.
            anchor = bpy.data.objects.new("Anchor", None)
            bpy.context.scene.collection.objects.link(anchor)
            anchor.location = (6.0, 6.0, 6.0)
            follow = far.constraints.new("COPY_LOCATION")
            follow.target = anchor
        if change == "driver retuned":
            # Its height read off an empty, then off another that stands at the
            # same height on frame 1 and climbs away through the job: only on
            # frame 1 does it stand in the same place.
            anchor = bpy.data.objects.new("Anchor", None)
            climber = bpy.data.objects.new("Climber", None)
            for empty in (anchor, climber):
                bpy.context.scene.collection.objects.link(empty)
                empty.location = (0.0, 0.0, 5.0)
            climber.keyframe_insert("location", index=2, frame=1)
            climber.location.z = 50.0
            climber.keyframe_insert("location", index=2, frame=31)
            driver = far.driver_add("location", 2).driver
            driver.type = "AVERAGE"
            variable = driver.variables.new()
            variable.type = "TRANSFORMS"
            variable.targets[0].id = anchor
            variable.targets[0].transform_type = "LOC_Z"
        if change == "scrubbed past a driven one":
            # It moves through the job by a driver on the frame, not by keys.
            driver = far.driver_add("location", 0).driver
            driver.type, driver.expression = "SCRIPTED", "5 + frame * 0.01"
        bpy.context.view_layer.update()
        bpy.context.view_layer.objects.active = arm6
        assert "FINISHED" in bpy.ops.kinema.check_motion()
        scene = bpy.context.scene
        assert collision.closest_approach(arm6, scene) is not None
        if change == "moved":
            far.location.x = 4.0
        elif change == "removed":
            for obj in bpy.context.selected_objects:
                obj.select_set(False)
            far.select_set(True)
            assert "FINISHED" in bpy.ops.kinema.remove_obstacles()
        elif change == "refitted":
            arm6.kinema_capsules_source = "fitted again"
        elif change == "capsule changed":
            # The meshes as they were fitted to: only the capsule is different.
            arm6.kinema_capsules[0].radius += 0.01
            assert not collision.capsules_stale(arm6), "precondition: the meshes are unchanged"
        elif change == "scaled while animated":
            # Only its location is keyed: its size is set by hand.
            far.scale = (40.0, 40.0, 40.0)
        elif change == "constrained":
            far.constraints.new("COPY_LOCATION").target = arm6
        elif change == "capsules stale":
            mesh_object = next(iter(collision.link_meshes(arm6).values()))[0]
            mesh_object.location.x += 0.05
        elif change == "constraint retuned":
            # Its own location added on: it moves 5 m, nothing else changes.
            far.constraints[0].use_offset = True
        elif change == "driver retuned":
            scene.frame_set(1)
            bpy.context.view_layer.update()
            # Where it stands now is no change: the drivers' settings are.
            assert far.location.z == pytest.approx(5.0)
            assert collision.closest_approach(arm6, scene) is not None
            far.animation_data.drivers[0].driver.variables[0].targets[0].id = climber
            bpy.context.view_layer.update()
            assert far.location.z == pytest.approx(5.0)
        elif change == "constraint folded":
            far.constraints[0].show_expanded = not far.constraints[0].show_expanded
        elif change == "moved with its keys muted":
            far.location.x = 4.0
        elif change == "an NLA strip's key changed":
            # Only where it ends up: on frame 1 it stands where it did.
            scene.frame_set(1)
            bpy.context.view_layer.update()
            assert far.location.x == pytest.approx(5.0)
            assert collision.closest_approach(arm6, scene) is not None
            (_track, _strip, curves), = collision._strip_curves(far)
            x = next(curve for curve in curves if curve.array_index == 0)
            x.keyframe_points[-1].co.y = 7.0
            x.update()
            bpy.context.view_layer.update()
            assert far.location.x == pytest.approx(5.0)
        elif change == "robot moved":
            # Every capsule somewhere else; the job's own fingerprint is its pose.
            arm6.location.x += 1.0
        elif change == "keys eased":
            # The same keys, but it jumps between them instead of gliding.
            for curves in modules("ops.ik").own_fcurve_containers(far):
                for curve in curves:
                    for point in curve.keyframe_points:
                        point.interpolation = "CONSTANT"
        else:
            scene.frame_set(20)
            if change == "scrubbed past a driven one":
                assert far.location.x == pytest.approx(5.2)
            if change == "scrubbed through an NLA strip":
                assert 5.1 < far.location.x < 5.9
        bpy.context.view_layer.update()
        found = collision.closest_approach(arm6, scene)
        stays = (
            "scrubbed", "constraint folded", "scrubbed past a driven one",
            "scrubbed through an NLA strip",
        )
        if change in stays:
            assert found is not None and found.clearance_obstacle == "FarAway"
        else:
            assert found is None

    def test_each_scene_has_obstacles_of_its_own(self, arm6, modules):
        """Collections are the file's: a name alone would share them between scenes."""
        import bpy

        collision = modules("ops.collision")
        first = bpy.context.scene
        crate = _obstacle("Crate", (1.0, 0.0, 0.0))
        firsts = collision.obstacle_collection(first)
        assert firsts.name == collision.OBSTACLES
        # Another scene holding a collection of that name: not its obstacles.
        other = bpy.data.scenes.new("Other")
        other.collection.children.link(firsts)
        assert crate.name in other.objects
        assert collision.obstacle_collection(other) is None
        assert collision.obstacle_objects(other) == []

        post = bpy.data.objects.new("Post", None)
        other.collection.objects.link(post)
        with bpy.context.temp_override(
            scene=other, view_layer=other.view_layers[0], selected_objects=[post]
        ):
            assert "FINISHED" in bpy.ops.kinema.add_obstacles()
        try:
            assert collision.obstacle_collection(other) not in (None, firsts)
            assert collision.obstacle_objects(other) == [post]
            assert collision.obstacle_objects(first) == [crate]
        finally:
            bpy.data.scenes.remove(other)

    def test_moving_the_tcp_outdates_the_check(self, arm6, builder, modules):
        """Set TCP moves where the TCP's bone rests, and the tool's capsules with it."""
        import bpy

        collision, waypoints = modules("ops.collision"), modules("ops.waypoints")
        TestChecking()._job(arm6, builder, modules)
        tcp = arm6.get(builder.PROP_TCP_BONE) or builder.TCP_BONE
        TestWhatALinkIs._on_bone(arm6, tcp, "Gripper", size=0.04)
        _fit(arm6)
        _obstacle("FarAway", (5.0, 5.0, 5.0), size=0.1)
        bpy.context.view_layer.objects.active = arm6
        assert "FINISHED" in bpy.ops.kinema.check_motion()
        scene = bpy.context.scene
        job = waypoints.job_signature(arm6)
        assert collision.check_measured_this(arm6, scene)
        bpy.context.view_layer.objects.active = arm6
        bpy.ops.object.mode_set(mode="EDIT")
        try:
            edit = arm6.data.edit_bones[tcp]
            edit.head.x += 0.05
            edit.tail.x += 0.05
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.context.view_layer.update()
        assert waypoints.job_signature(arm6) == job
        assert not collision.check_measured_this(arm6, scene)
        assert collision.closest_approach(arm6, scene) is None

    def test_resizing_an_empty_obstacle_outdates_the_check(self, arm6, builder, modules):
        """An empty is measured by its display size, which its bounds don't follow."""
        import bpy

        collision = modules("ops.collision")
        TestChecking()._job(arm6, builder, modules)
        _fit(arm6)
        post = bpy.data.objects.new("Post", None)
        bpy.context.scene.collection.objects.link(post)
        post.location = (5.0, 5.0, 5.0)
        with bpy.context.temp_override(selected_objects=[post]):
            assert "FINISHED" in bpy.ops.kinema.add_obstacles()
        bpy.context.view_layer.objects.active = arm6
        assert "FINISHED" in bpy.ops.kinema.check_motion()
        scene = bpy.context.scene
        assert collision.closest_approach(arm6, scene).clearance_obstacle == "Post"
        post.empty_display_size = 3.0
        bpy.context.view_layer.update()
        assert collision.closest_approach(arm6, scene) is None

    def test_stale_capsules_outdate_the_motion_check(self, arm6, builder, modules):
        """Its clearances were measured with capsules that no longer fit the links."""
        import bpy

        collision, panel = modules("ops.collision"), modules("ui.panel")
        TestChecking()._job(arm6, builder, modules)
        _fit(arm6)
        _obstacle("FarAway", (5.0, 5.0, 5.0), size=0.1)
        bpy.context.view_layer.objects.active = arm6
        assert "FINISHED" in bpy.ops.kinema.check_motion()
        scene = bpy.context.scene
        assert panel.check_outdated(arm6, scene) == ""
        next(iter(collision.link_meshes(arm6).values()))[0].location.x += 0.05
        bpy.context.view_layer.update()
        assert collision.check_measured_this(arm6, scene)
        assert "Fit Again" in panel.check_outdated(arm6, scene)


class TestTheRobotStaysInThePanels:
    """Selecting a fixture to make it an obstacle doesn't take the robot's panels away."""

    def _select_only(self, obj):
        import bpy

        for other in bpy.context.selected_objects:
            other.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

    def test_selecting_a_fixture_keeps_the_robot_shown(self, arm6, modules):
        import bpy

        panel = modules("ui.panel")
        self._select_only(arm6)
        assert panel.active_rig(bpy.context) == arm6
        crate = _obstacle("Crate", (1.0, 0.0, 0.0))
        assert bpy.context.object == crate and not arm6.select_get()
        assert panel.active_rig(bpy.context) == arm6
        assert not panel.is_selected(bpy.context, arm6)
        assert panel.KINEMA_PT_collision.poll(bpy.context)
        assert bpy.ops.kinema.fit_capsules.poll()

    def test_the_last_robot_shown_of_two(self, arm6, modules):
        import bpy

        panel = modules("ui.panel")
        second = arm6.copy()
        second.data = arm6.data.copy()
        bpy.context.scene.collection.objects.link(second)
        crate = _obstacle("Crate", (1.0, 0.0, 0.0))
        bpy.context.view_layer.update()
        for shown in (second, arm6):
            self._select_only(shown)
            assert panel.active_rig(bpy.context) == shown
            self._select_only(crate)
            assert panel.active_rig(bpy.context) == shown

    def test_editing_another_object_shows_no_robot(self, arm6, modules):
        import bpy

        panel = modules("ui.panel")
        self._select_only(arm6)
        assert panel.active_rig(bpy.context) == arm6
        crate = _obstacle("Crate", (1.0, 0.0, 0.0))
        bpy.ops.object.mode_set(mode="EDIT")
        try:
            assert bpy.context.mode == "EDIT_MESH"
            assert panel.active_rig(bpy.context) is None
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")
        assert crate.mode == "OBJECT"


class TestOverlay:
    @pytest.mark.parametrize("scale", [(1.0, 1.0, 1.0), (1.0, 1.0, 3.0)])
    def test_outlines_are_placed_not_rebuilt(self, arm6, builder, modules, monkeypatch, scale):
        """One outline, placed on each redraw round each capsule as the check measures it."""
        import bpy

        overlay, collision = modules("ui.collision_overlay"), modules("ops.collision")
        _fit(arm6)
        arm6.scale = scale
        bpy.context.view_layer.update()
        capsules = collision.capsules_of(arm6)
        outlined = []
        original = overlay.capsule_lines
        monkeypatch.setattr(
            overlay, "capsule_lines", lambda *args: outlined.append(1) or original(*args)
        )
        overlay.forget()
        overlay._outline = None
        geometry = modules("rig.collision")
        per_capsule = len(original((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0))
        for q in (HOME, TURNED):
            _set_q(builder, arm6, q)
            clear, _ = overlay.shape_lines(bpy.context)
            assert clear.shape == (per_capsule * len(capsules), 2, 3)
            a, b, radii = collision.capsule_ends(arm6, capsules)
            # Every point drawn on its capsule's surface, wherever the arm is.
            for index in range(len(capsules)):
                points = clear[index * per_capsule:(index + 1) * per_capsule].reshape(-1, 3)
                distance = geometry._point_segment_distance(points, a[index], b[index])
                np.testing.assert_allclose(distance, radii[index], atol=1e-6)
        assert len(outlined) == 1


    def test_what_touches_is_drawn_red(self, arm6, builder, modules):
        import bpy

        overlay = modules("ui.collision_overlay")
        collision = modules("ops.collision")
        _fit(arm6)
        far = _obstacle("FarAway", (5.0, 5.0, 5.0))
        clear, hit = overlay.shape_lines(bpy.context)
        assert len(clear) and not len(hit)

        capsules = collision.capsules_of(arm6)
        a, b, _ = collision.capsule_ends(arm6, capsules)
        far.location = tuple((a[0] + b[0]) / 2.0)
        bpy.context.view_layer.update()
        clear, hit = overlay.shape_lines(bpy.context)
        # The capsule and the box, each outlined in red.
        assert len(hit) >= len(overlay.capsule_lines(a[0], b[0], capsules[0].radius)) + 12


class TestOpeningAFile:
    def test_every_handler_outlives_a_file_being_opened(self, addon, modules):
        """Blender drops a handler that isn't persistent before it opens a file."""
        handlers = modules("handlers")
        for collection, function in handlers._HANDLERS:
            assert function in collection, function.__name__
            # What @persistent leaves: the attribute, set to None.
            assert hasattr(function, "_bpy_persistent"), function.__name__

    def test_the_robot_last_shown_is_forgotten(self, arm6, modules):
        """The next file may have a robot of the same name, in a scene of the same name."""
        import bpy

        panel, handlers = modules("ui.panel"), modules("handlers")
        assert panel.active_rig(bpy.context) == arm6
        assert panel._shown
        handlers.on_load_post()
        assert not panel._shown
