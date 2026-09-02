"""Render the multi-turn comparison: Kinema vs Blender's built-in IK.

Both robots live in one scene and are shot by one camera, so the two halves of
the frame are guaranteed to share timing, framing and lighting. The only thing
that differs is which solver drives the arm.

Kinema's motion is baked to keyframes rather than solved live, so what renders
is exactly what tools/demo/sweep.py measured.
"""
from __future__ import annotations

import math
import sys

import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import robots  # noqa: E402
from _blendfile import save_blend  # noqa: E402

EXT = "bl_ext.user_default.kinema"
ROBOT = "ur5e_description"
TURNS = 2.0
FRAMES = 120
SEPARATION = 1.4     # metres between the two robots

SINGULAR_POSE = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.0,
    "elbow_joint": 1.4,
    "wrist_1_joint": -0.4,
    "wrist_2_joint": 0.0,
    "wrist_3_joint": 0.0,
}


def log(m):
    print(f"[render] {m}", flush=True)


def import_rig(enforce_limits: bool):
    builder = sys.modules[f"{EXT}.rig.builder"]
    before = set(bpy.data.objects)
    bpy.ops.kinema.build_robot(
        filepath=robots.resolve(ROBOT), enforce_limits=enforce_limits
    )
    return next(o for o in bpy.data.objects
                if o not in before and builder.is_kinema_rig(o))


def pose_singular(rig):
    for name, angle in SINGULAR_POSE.items():
        if name in rig.pose.bones:
            rig.pose.bones[name].rotation_euler[1] = angle


def main():
    out_dir = sys.argv[sys.argv.index("--") + 1]

    builder = sys.modules[f"{EXT}.rig.builder"]
    manager = sys.modules[f"{EXT}.solver.manager"]

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    log("importing two UR5e rigs ...")
    rig_k = import_rig(enforce_limits=True)
    rig_b = import_rig(enforce_limits=False)
    rig_k.location.x = -SEPARATION / 2
    rig_b.location.x = +SEPARATION / 2
    for rig in (rig_k, rig_b):
        pose_singular(rig)
    bpy.context.view_layer.update()

    # ---------------- Kinema side ----------------
    bpy.context.view_layer.objects.active = rig_k
    rig_k.select_set(True)
    bpy.ops.kinema.add_ik()
    rig_k.select_set(False)
    rig_k.kinema_solver_mode = "PYROKI"
    rig_k.kinema_ik_enabled = False   # baked below; no live solving during render
    ik_bone = rig_k.get(builder.PROP_IK_BONE)
    solver = manager.get_solver(rig_k)
    solver.solve(rig_k, "PYROKI")
    if solver.pyroki_error is not None or solver.last_result.backend != "PyRoki":
        raise RuntimeError(f"not PyRoki: {solver.pyroki_error} "
                           f"{solver.last_result.backend}")
    log(f"kinema backend: {solver.last_result.backend}")

    # ---------------- Blender-native side ----------------
    joints_b = builder.joint_bones(rig_b)
    for pb in joints_b:
        pb.lock_ik_x = pb.lock_ik_z = True
        pb.lock_ik_y = False
        lo, hi = pb.bone.get(builder.PROP_LOWER), pb.bone.get(builder.PROP_UPPER)
        if lo is not None and hi is not None and hi > lo:
            pb.use_ik_limit_y = True
            pb.ik_min_y = float(lo)   # Blender clamps these to +-pi
            pb.ik_max_y = float(hi)
    tcp_b = rig_b.pose.bones[rig_b.get(builder.PROP_TCP_BONE) or builder.TCP_BONE]
    tcp_b.lock_location = (False, False, False)
    tcp_b.lock_rotation = (False, False, False)
    tcp_b.lock_rotation_w = False
    tcp_b.lock_ik_x = tcp_b.lock_ik_y = tcp_b.lock_ik_z = True   # rigid tool
    target = bpy.data.objects.new("IK_Target", None)
    target.empty_display_size = 0.06
    bpy.context.scene.collection.objects.link(target)
    con = tcp_b.constraints.new("IK")
    con.target = target
    con.chain_count = len(joints_b) + 1
    con.use_rotation = True
    rig_b.pose.ik_solver = "LEGACY"
    log(f"native IK: {len(joints_b)} joints, chain={con.chain_count}, "
        f"solver={rig_b.pose.ik_solver}")

    # ---------------- the shared tool path ----------------
    bpy.context.view_layer.update()
    tcp_k = rig_k.pose.bones[rig_k.get(builder.PROP_TCP_BONE) or builder.TCP_BONE]
    base_k = tcp_k.matrix.copy()
    tcp_len = tcp_k.length
    offset = Vector((SEPARATION, 0.0, 0.0))   # rig_b sits here relative to rig_k

    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, FRAMES
    joints_k = builder.joint_bones(rig_k)

    log(f"baking {FRAMES} frames ...")
    for i in range(FRAMES):
        frame = i + 1
        scene.frame_set(frame)
        t = i / (FRAMES - 1)
        goal = base_k @ Matrix.Rotation(t * TURNS * 2.0 * math.pi, 4, "Y")

        # Kinema: solve in armature space, then key the one live channel.
        rig_k.pose.bones[ik_bone].matrix = goal.copy()
        bpy.context.view_layer.update()
        solver.solve(rig_k, "PYROKI")
        for pb in joints_k:
            pb.keyframe_insert("rotation_euler", index=1, frame=frame)

        # Blender: key the empty at the goal's tail, in the other rig's space.
        # Its IK constraint solves at render time, which is the whole point.
        tail = goal @ Vector((0.0, tcp_len, 0.0))
        m = goal.copy()
        m.translation = tail
        target.matrix_world = (Matrix.Translation(rig_k.location + offset)
                               @ m)
        target.keyframe_insert("location", frame=frame)
        target.keyframe_insert("rotation_euler", frame=frame)

    # Interpolation is moot: there is a key on every rendered frame, so nothing
    # is ever interpolated. (Blender 5's slotted actions no longer expose
    # action.fcurves anyway.)

    # ---------------- camera, light, look ----------------
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens = 50
    cam = bpy.data.objects.new("Cam", cam_data)
    scene.collection.objects.link(cam)
    # Level-on and far enough back that both robots and their full reach fit:
    # at 4.2 m a 50 mm lens covers ~3.0 m of width, against ~2.4 m of robot.
    cam.location = Vector((0.0, -4.2, 0.45))
    cam.rotation_euler = (math.radians(90), 0.0, 0.0)
    scene.camera = cam

    scene.render.engine = "BLENDER_WORKBENCH"
    shading = scene.display.shading
    shading.light = "STUDIO"
    shading.color_type = "OBJECT"
    shading.show_shadows = True
    shading.show_cavity = True
    scene.display.render_aa = "8"
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("W")
    scene.world.color = (0.05, 0.05, 0.06)

    # Bones would clutter a demo about where the arm ends up.
    for rig in (rig_k, rig_b):
        rig.hide_render = True
    target.hide_render = True

    scene.render.resolution_x = 960
    scene.render.resolution_y = 540
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = f"{out_dir}/frame_"

    save_blend("ik-comparison")

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
