"""Feasibility probe for two demos PyRoki can do and Blender's IK cannot.

A. Multiple IK branches for ONE target pose (UR5e, 6-DoF). A 6R arm has up to
   eight analytic solutions; seeding the optimiser differently should land on
   several of them, all hitting the same tool pose.

B. Null-space sweep (Panda, 7-DoF). A redundant arm has a one-dimensional family
   of configurations for a fixed tool pose -- the elbow swivel. Sweeping it while
   the tool stays nailed to the target is the demo.

Both use only the public solve path: pose the arm at a seed, then solve. PyRoki's
rest_cost biases toward wherever it started, so the seed selects the branch.
"""
from __future__ import annotations

import math
import sys

import numpy as np
import bpy

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _blendfile import save_blend  # noqa: E402

EXT = "bl_ext.user_default.kinema"


def log(m):
    print(f"[multi] {m}", flush=True)


def build(robot_key, tcp_bone=None):
    builder = sys.modules[f"{EXT}.rig.builder"]
    manager = sys.modules[f"{EXT}.solver.manager"]
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    res = bpy.ops.kinema.build_robot(robot_key=robot_key)
    if "FINISHED" not in res:
        raise RuntimeError(f"import failed for {robot_key}: {res}")
    rig = next(o for o in bpy.data.objects if builder.is_kinema_rig(o))

    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    if tcp_bone is not None:
        bpy.ops.object.mode_set(mode="POSE")
        rig.data.bones.active = rig.data.bones[tcp_bone]
        bpy.ops.kinema.set_tcp()
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.kinema.add_ik()
    rig.select_set(False)
    rig.kinema_solver_mode = "PYROKI"
    rig.kinema_ik_enabled = False        # measurement owns the solving
    solver = manager.get_solver(rig)
    solver.solve(rig, "PYROKI")
    if solver.pyroki_error is not None:
        raise RuntimeError(f"PyRoki unavailable: {solver.pyroki_error}")
    joints = builder.joint_bones(rig)
    log(f"{robot_key}: {len(joints)} joints, backend={solver.last_result.backend}")
    return rig, solver, joints, builder


def q_of(joints):
    return np.array([pb.rotation_euler[1] for pb in joints])


def set_q(joints, q):
    for pb, v in zip(joints, q):
        pb.rotation_euler[1] = float(v)
    bpy.context.view_layer.update()


def tool_error(rig, builder, goal):
    tcp = rig.pose.bones[rig.get(builder.PROP_TCP_BONE) or builder.TCP_BONE]
    return (tcp.matrix.translation - goal.translation).length * 1000.0


# --------------------------------------------------------------- A: branches
def probe_branches(seed_count=40):
    rig, solver, joints, builder = build("ur5e_description")
    ik = rig.pose.bones[rig.get(builder.PROP_IK_BONE)]

    set_q(joints, [0.0, -1.0, 1.4, -0.4, 0.0, 0.0])
    tcp = rig.pose.bones[rig.get(builder.PROP_TCP_BONE) or builder.TCP_BONE]
    goal = tcp.matrix.copy()

    rng = np.random.default_rng(0)
    lo = np.array([pb.bone.get(builder.PROP_LOWER, -math.pi) for pb in joints])
    hi = np.array([pb.bone.get(builder.PROP_UPPER, math.pi) for pb in joints])
    lo = np.clip(lo, -math.pi, math.pi)
    hi = np.clip(hi, -math.pi, math.pi)

    found = []
    for i in range(seed_count):
        set_q(joints, rng.uniform(lo, hi))
        ik.matrix = goal.copy()
        bpy.context.view_layer.update()
        solver.solve(rig, "PYROKI")
        bpy.context.view_layer.update()
        err = tool_error(rig, builder, goal)
        if err > 0.5:            # did not actually reach the target
            continue
        q = q_of(joints)
        if not any(np.max(np.abs(np.angle(np.exp(1j * (q - f))))) < 0.1
                   for f in found):
            found.append(q)

    log(f"A. distinct branches for one UR5e pose: {len(found)} "
        f"(from {seed_count} seeds)")
    for i, q in enumerate(found):
        set_q(joints, q)
        log(f"     branch {i}: err {tool_error(rig, builder, goal):6.4f} mm  "
            f"q = [{', '.join(f'{math.degrees(v):7.1f}' for v in q)}]")
    return len(found)


# ------------------------------------------------------------ B: null space
def probe_nullspace(robot_key="panda_mj_description", steps=24,
                    tcp_bone="joint7"):
    # The Panda imports with its TCP on 'right_finger', which leaves both gripper
    # joints inside the IK chain -- 9 DoF against a 6-DoF task. The solver then
    # holds the fingertip exactly while spinning the hand around it, so a
    # position-only metric reads zero while the flange visibly rotates. Target
    # the flange instead and the chain is the 7 arm joints it should be.
    rig, solver, joints, builder = build(robot_key, tcp_bone=tcp_bone)
    ik = rig.pose.bones[rig.get(builder.PROP_IK_BONE)]
    n = len(joints)
    log(f"B. {robot_key}: TCP on {tcp_bone!r}, {n} actuated joints "
        f"({'redundant' if n > 6 else 'NOT redundant'})")

    rest = q_of(joints)
    tcp = rig.pose.bones[rig.get(builder.PROP_TCP_BONE) or builder.TCP_BONE]
    set_q(joints, rest)
    goal = tcp.matrix.copy()

    tcp = rig.pose.bones[rig.get(builder.PROP_TCP_BONE) or builder.TCP_BONE]
    errs, ori_errs, first_joint = [], [], []
    for i in range(steps):
        t = i / (steps - 1)
        bias = rest.copy()
        bias[0] = rest[0] + (t - 0.5) * 2.0 * 1.6   # swing the base joint
        set_q(joints, bias)
        ik.matrix = goal.copy()
        bpy.context.view_layer.update()
        solver.solve(rig, "PYROKI")
        bpy.context.view_layer.update()
        errs.append(tool_error(rig, builder, goal))
        # Orientation too: position alone reads zero even when the tool spins.
        delta = goal.to_3x3().inverted() @ tcp.matrix.to_3x3()
        ori_errs.append(math.degrees(delta.to_quaternion().angle))
        first_joint.append(q_of(joints)[0])

    span = math.degrees(max(first_joint) - min(first_joint))
    log(f"     tool error over the sweep: max {max(errs):.4f} mm, "
        f"{max(ori_errs):.4f} deg")
    log(f"     joint-0 span while the tool stayed put: {span:.1f} deg")
    return span, max(errs)


def main():
    n = probe_branches()
    save_blend("branches")      # UR5e, left on the last branch found
    span, err = probe_nullspace()
    save_blend("branches-nullspace")
    log("")
    log(f"VERDICT A (multiple branches): "
        f"{'YES' if n >= 2 else 'NO'} -- {n} distinct solutions")
    log(f"VERDICT B (null-space sweep): "
        f"{'YES' if span > 20 and err < 1.0 else 'NO'} -- "
        f"{span:.1f} deg of motion at {err:.3f} mm tool error")
    return 0


try:
    code = main()
except Exception:
    import traceback
    traceback.print_exc()
    code = 1
log(f"exit {code}")
sys.exit(code)
