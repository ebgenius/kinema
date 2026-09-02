"""Render the null-space sweep: a redundant arm reconfiguring around a fixed tool.

A 7-DoF arm has a one-dimensional family of configurations that all put the tool
in exactly the same place. Kinema can walk that family because PyRoki treats the
joint limits as constraints and is seeded per solve; Blender's IK returns
whichever single configuration its iterative solver happens to land on, with no
way to ask for another.

The marker sphere is static, placed once at the goal. If the tool leaves it, the
demo is lying.
"""
from __future__ import annotations

import math
import sys

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import robots  # noqa: E402
from _blendfile import save_blend  # noqa: E402

EXT = "bl_ext.user_default.kinema"
ROBOT = "panda_mj_description"
FRAMES = 96
SWING = 1.6          # radians of seed bias, each way
SWEEP_JOINT = 0      # which joint to push; the rest follow to hold the tool

SEED_POSE = [0.0, -0.6, 0.0, -2.2, 0.0, 1.7, 0.8]


def log(m):
    print(f"[nullspace] {m}", flush=True)


def main():
    out_dir = sys.argv[sys.argv.index("--") + 1]

    builder = sys.modules[f"{EXT}.rig.builder"]
    manager = sys.modules[f"{EXT}.solver.manager"]

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    log(f"importing {ROBOT} ...")
    bpy.ops.kinema.build_robot(filepath=robots.resolve(ROBOT))
    rig = next(o for o in bpy.data.objects if builder.is_kinema_rig(o))
    joints = builder.joint_bones(rig)

    # Move the TCP onto the flange before adding IK.
    #
    # The Panda imports with its TCP on 'right_finger', which puts both gripper
    # joints *inside* the IK chain: 9 DoF against a 6-DoF task. The solver then
    # holds the fingertip perfectly while spinning the whole hand around it --
    # the tool pose is fixed by the letter of the metric and visibly rotating on
    # screen. Targeting the flange drops the finger joints out of the chain and
    # leaves exactly the 7 arm joints, which is the redundancy this demo is about.
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="POSE")
    rig.data.bones.active = rig.data.bones["joint7"]
    bpy.ops.kinema.set_tcp()
    bpy.ops.object.mode_set(mode="OBJECT")

    joints = builder.joint_bones(rig)
    arm = joints[:7]
    for pb, v in zip(arm, SEED_POSE):
        pb.rotation_euler[1] = v
    bpy.context.view_layer.update()

    bpy.ops.kinema.add_ik()
    rig.select_set(False)
    rig.kinema_solver_mode = "PYROKI"
    rig.kinema_ik_enabled = False    # baked below
    ik = rig.pose.bones[rig.get(builder.PROP_IK_BONE)]
    solver = manager.get_solver(rig)
    solver.solve(rig, "PYROKI")
    if solver.pyroki_error is not None or solver.last_result.backend != "PyRoki":
        raise RuntimeError(f"not PyRoki: {solver.pyroki_error}")
    log(f"backend={solver.last_result.backend}, {len(joints)} joints "
        f"({len(arm)} arm)")

    tcp = rig.pose.bones[rig.get(builder.PROP_TCP_BONE) or builder.TCP_BONE]
    goal = tcp.matrix.copy()
    goal_world = rig.matrix_world @ goal.translation

    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, FRAMES
    rest = np.array([pb.rotation_euler[1] for pb in joints])

    # The flange's own world orientation, tracked across the sweep. The tool-frame
    # error can read zero while the hand visibly spins, if the task frame sits
    # below a spare joint -- so this is measured directly rather than inferred.
    flange = rig.pose.bones["joint7"]
    flange_ref = flange.matrix.to_3x3().copy()

    log(f"baking {FRAMES} frames ...")
    errors, ori_errors, flange_spin = [], [], []
    for i in range(FRAMES):
        frame = i + 1
        scene.frame_set(frame)
        # Out and back, so the GIF loops without a seam.
        t = math.sin(2.0 * math.pi * i / FRAMES)

        bias = rest.copy()
        bias[SWEEP_JOINT] = rest[SWEEP_JOINT] + t * SWING
        for pb, v in zip(joints, bias):
            pb.rotation_euler[1] = float(v)
        bpy.context.view_layer.update()

        ik.matrix = goal.copy()
        bpy.context.view_layer.update()
        solver.solve(rig, "PYROKI")
        bpy.context.view_layer.update()

        errors.append(((rig.matrix_world @ tcp.matrix.translation)
                       - goal_world).length * 1000.0)
        delta = goal.to_3x3().inverted() @ tcp.matrix.to_3x3()
        ori_errors.append(math.degrees(delta.to_quaternion().angle))
        spin = flange_ref.inverted() @ flange.matrix.to_3x3()
        flange_spin.append(math.degrees(spin.to_quaternion().angle))

        for pb in joints:
            pb.keyframe_insert("rotation_euler", index=1, frame=frame)

    log(f"tool position error: max {max(errors):.4f} mm, "
        f"mean {sum(errors)/len(errors):.4f} mm")
    log(f"tool orientation error: max {max(ori_errors):.4f} deg")
    log(f"flange rotation in world: max {max(flange_spin):.4f} deg "
        f"({'HOLDS STILL' if max(flange_spin) < 0.05 else 'STILL ROTATING'})")

    # --- the marker that proves the tool does not move ---
    marker_mesh = bpy.data.meshes.new("goal")
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=0.038)
    bm.to_mesh(marker_mesh)
    bm.free()
    marker = bpy.data.objects.new("GoalMarker", marker_mesh)
    marker.location = goal_world
    marker.color = (1.0, 0.45, 0.15, 1.0)
    scene.collection.objects.link(marker)

    # --- camera and look ---
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens = 80
    cam = bpy.data.objects.new("Cam", cam_data)
    scene.collection.objects.link(cam)
    # Framed on the arm's working volume (roughly z 0.1-0.8), not on the world
    # origin: the base plinth is not what anyone is looking at.
    cam.location = Vector((1.62, -1.72, 0.92))
    cam.rotation_euler = (math.radians(80), 0.0, math.radians(43))
    scene.camera = cam

    scene.render.engine = "BLENDER_WORKBENCH"
    shading = scene.display.shading
    shading.light = "STUDIO"
    shading.color_type = "OBJECT"
    shading.show_shadows = True
    shading.show_cavity = True
    scene.display.render_aa = "8"
    scene.world = bpy.data.worlds.new("W")
    scene.world.color = (0.05, 0.05, 0.06)
    rig.hide_render = True   # bones would clutter it

    scene.render.resolution_x = 720
    scene.render.resolution_y = 560
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = f"{out_dir}/frame_"

    save_blend("nullspace")

    log(f"rendering to {out_dir} ...")
    bpy.ops.render.render(animation=True)
    log("done")
    return 0


try:
    code = main()
except Exception:
    import traceback
    traceback.print_exc()
    code = 1
log(f"exit {code}")
sys.exit(code)
