"""Measure Kinema/PyRoki against Blender's built-in IK on identical UR5e rigs.

Both rigs are built from the same catalog description and sit at the origin, so
"same input" is trivially true: one world-space target matrix per frame is
written to both.

Measurement is identical for both solvers. Blender's IK writes the *evaluated*
pose and never touches rotation_euler, so reading raw channels would report
zeros for the baseline. Instead each joint's local rotation is derived from the
evaluated matrices the same way for both rigs:

    basis = (rest_parent^-1 @ rest_bone)^-1 @ (pose_parent^-1 @ pose_bone)

which is what matrix_basis would hold if nothing else were driving the bone.
"""
from __future__ import annotations

import csv
import json
import math
import sys

import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import robots  # noqa: E402
from _blendfile import save_blend  # noqa: E402

EXT = "bl_ext.user_default.kinema"
ROBOT = "ur5e_description"

# The UR5e wrist singularity: wrist_1 and wrist_3 axes become collinear when
# wrist_2 is zero. Every candidate path is centred on this configuration.
SINGULAR_POSE = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.0,
    "elbow_joint": 1.4,
    "wrist_1_joint": -0.4,
    "wrist_2_joint": 0.0,   # <- the degenerate one
    "wrist_3_joint": 0.0,
}

FRAMES = 60


def log(msg):
    print(f"[sweep] {msg}", flush=True)


# ---------------------------------------------------------------- rig building
def import_rig(enforce_limits: bool):
    before = set(bpy.data.objects)
    res = bpy.ops.kinema.build_robot(
        filepath=robots.resolve(ROBOT), enforce_limits=enforce_limits, create_tcp=True
    )
    if "FINISHED" not in res:
        raise RuntimeError(f"import failed: {res}")
    builder = sys.modules[f"{EXT}.rig.builder"]
    new = [o for o in bpy.data.objects if o not in before and builder.is_kinema_rig(o)]
    if len(new) != 1:
        raise RuntimeError(f"expected one new rig, got {new}")
    return new[0]


def configure_native_ik(rig, ik_target, solver: str, use_limits: bool):
    """Give Blender's own IK the same degrees of freedom Kinema has.

    Transform locks (lock_rotation) do NOT constrain Blender's IK solver -- it
    reads lock_ik_* instead. Without this every 1-DoF joint would be solved as a
    free 3-DoF ball joint, producing motion the robot cannot perform. That would
    flatter Kinema and prove nothing.

    Limits are the part that cannot be made equal. Blender's ik_min_y/ik_max_y
    are hard-clamped to +-pi, while five of the UR5e's six joints travel +-2pi.
    So there are only two honest baselines, and this runs both:

      use_limits=True   the widest range Blender can express (+-180 deg),
                        which is half the robot's real travel
      use_limits=False  no IK limits at all, free to reach any pose including
                        ones the real robot cannot achieve
    """
    builder = sys.modules[f"{EXT}.rig.builder"]
    joints = builder.joint_bones(rig)

    for pb in joints:
        # Only local Y is a real degree of freedom; see builder.py:264.
        pb.lock_ik_x = True
        pb.lock_ik_z = True
        pb.lock_ik_y = False

        lower = pb.bone.get(builder.PROP_LOWER)
        upper = pb.bone.get(builder.PROP_UPPER)
        if use_limits and lower is not None and upper is not None and upper > lower:
            pb.use_ik_limit_y = True
            pb.ik_min_y = float(lower)   # silently clamped to -pi by Blender
            pb.ik_max_y = float(upper)   # silently clamped to +pi
        else:
            pb.use_ik_limit_y = False

        # Kinema's own LIMIT_ROTATION constraints would clip the IK result after
        # the fact, enforcing limits twice in two different places.
        for con in list(pb.constraints):
            if con.name == builder.LIMIT_CONSTRAINT:
                pb.constraints.remove(con)

    tcp_name = rig.get(builder.PROP_TCP_BONE) or builder.TCP_BONE
    tcp = rig.pose.bones[tcp_name]
    tcp.lock_location = (False, False, False)
    tcp.lock_rotation = (False, False, False)
    tcp.lock_rotation_w = False

    # The TCP is a rigid tool offset on the flange, not a joint. Leaving its IK
    # DoF free would hand the baseline 9 degrees of freedom against the robot's
    # real 6 -- unfair in Blender's favour, and physically meaningless.
    tcp.lock_ik_x = tcp.lock_ik_y = tcp.lock_ik_z = True

    con = tcp.constraints.new("IK")
    con.target = ik_target
    con.chain_count = len(joints) + 1   # six joints plus the rigid TCP
    # Blender's IK matches position only unless asked. Kinema solves a full
    # 6-DoF pose goal, so without this the baseline would be solving an easier
    # problem and the comparison would be meaningless.
    con.use_rotation = True
    rig.pose.ik_solver = solver
    return joints, tcp


# ------------------------------------------------------------------ measuring
def local_basis(pb) -> Matrix:
    """The bone's pose relative to its rest, in parent space.

    Equivalent to matrix_basis when nothing else drives the bone, but derived
    from evaluated matrices so it works for IK-driven bones too.
    """
    if pb.parent:
        rest = pb.parent.bone.matrix_local.inverted() @ pb.bone.matrix_local
        pose = pb.parent.matrix.inverted() @ pb.matrix
    else:
        rest = pb.bone.matrix_local
        pose = pb.matrix
    return rest.inverted() @ pose


def joint_angles(joints) -> list[float]:
    return [local_basis(pb).to_euler("YXZ").y for pb in joints]


def wrapped_delta(a: float, b: float) -> float:
    """Shortest angular difference, so a wrap past pi is not a fake 2pi jump."""
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


# ----------------------------------------------------------------- the sweep
def limit_violation(joints, angles) -> float:
    """Worst excursion outside the robot's *real* URDF limits, in radians.

    Measured against the URDF, not against whatever the solver was told, so a
    baseline running without IK limits is scored on the physical robot's range.
    """
    builder = sys.modules[f"{EXT}.rig.builder"]
    worst = 0.0
    for pb, q in zip(joints, angles):
        lo, hi = pb.bone.get(builder.PROP_LOWER), pb.bone.get(builder.PROP_UPPER)
        if lo is None or hi is None or hi <= lo:
            continue
        worst = max(worst, lo - q, q - hi, 0.0)
    return worst


def run_path(rig_k, ik_bone_name, joints_k, baselines, target_empty, path, tag,
             writer, kinema_extra=()):
    """Drive every rig along `path` (world matrices) and record per-frame metrics.

    One target matrix is written to all rigs each frame, so "same input" is not
    an assumption -- it is the only input there is.
    """
    builder = sys.modules[f"{EXT}.rig.builder"]
    manager = sys.modules[f"{EXT}.solver.manager"]
    tcp_k_name = rig_k.get(builder.PROP_TCP_BONE) or builder.TCP_BONE
    solver = manager.get_solver(rig_k)

    tcp_k = rig_k.pose.bones[tcp_k_name]
    tcp_len = tcp_k.length

    # Every path starts from the same configuration. Without this, path N begins
    # wherever path N-1 happened to end, and the numbers depend on dict order.
    for rig in [rig_k] + [e["rig"] for e in kinema_extra] \
            + [b["rig"] for b in baselines]:
        for name, angle in SINGULAR_POSE.items():
            if name in rig.pose.bones:
                rig.pose.bones[name].rotation_euler[1] = angle
    bpy.context.view_layer.update()

    def tail_of(pose_bone) -> Vector:
        """World position of a bone's tail.

        Blender's IK drives the *tail* of the last chain bone, while Kinema
        drives the tool frame at the head. Scoring both at the tail against the
        same goal-derived tail point is what makes the two comparable; scoring
        head-to-target would charge the baseline a fixed offset equal to the
        TCP bone's length for work it was never asked to do.
        """
        return pose_bone.matrix @ Vector((0.0, pose_bone.length, 0.0))

    prev = {"kinema": None,
            **{b["tag"]: None for b in baselines},
            **{e["tag"]: None for e in kinema_extra}}
    rows = []
    for i, goal in enumerate(path):
        goal_tail = goal @ Vector((0.0, tcp_len, 0.0))

        # Kinema is given the tool-frame goal; the baseline's empty is placed at
        # the same goal's tail, with the same rotation, so both are asked for
        # the identical tool pose in the terms each solver expects.
        rig_k.pose.bones[ik_bone_name].matrix = goal.copy()
        for e in kinema_extra:
            e["rig"].pose.bones[e["ik_bone"]].matrix = goal.copy()
        empty_matrix = goal.copy()
        empty_matrix.translation = goal_tail
        target_empty.matrix_world = empty_matrix
        bpy.context.view_layer.update()

        result = solver.solve(rig_k, "PYROKI")
        for e in kinema_extra:
            e["solver"].solve(e["rig"], e["mode"])
        bpy.context.view_layer.update()

        row = {"path": tag, "frame": i,
               "backend": result.backend if result else "none"}
        samples = [("kinema", joints_k, tcp_k)]
        for e in kinema_extra:
            samples.append((e["tag"], e["joints"], e["tcp"]))
        for b in baselines:
            samples.append((b["tag"], b["joints"], b["tcp"]))

        for name, joints, tcp in samples:
            q = joint_angles(joints)
            jump = 0.0
            if prev[name] is not None:
                jump = max(wrapped_delta(a, b) for a, b in zip(q, prev[name]))
            prev[name] = q
            row[f"{name}_err_mm"] = (tail_of(tcp) - goal_tail).length * 1000.0
            row[f"{name}_jump_deg"] = math.degrees(jump)
            row[f"{name}_violation_deg"] = math.degrees(limit_violation(joints, q))

        rows.append(row)
        writer.writerow(row)
    return rows


def summarise(rows, tag, names):
    out = {"path": tag}
    log(f"  {tag}")
    for name in names:
        jump = max(r[f"{name}_jump_deg"] for r in rows)
        err = max(r[f"{name}_err_mm"] for r in rows)
        viol = max(r[f"{name}_violation_deg"] for r in rows)
        log(f"      {name:24s} worst jump {jump:8.2f} deg   worst err {err:9.3f} mm"
            f"   limit violation {viol:7.2f} deg")
        out[name] = dict(max_jump_deg=jump, max_err_mm=err, max_violation_deg=viol)
    return out


def main():
    out_csv = sys.argv[sys.argv.index("--") + 1]
    ik_solver = sys.argv[sys.argv.index("--") + 2]

    builder = sys.modules[f"{EXT}.rig.builder"]
    manager = sys.modules[f"{EXT}.solver.manager"]

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    log(f"importing {ROBOT} x4 (ik_solver={ik_solver}) ...")
    rig_k = import_rig(enforce_limits=True)
    rig_np = import_rig(enforce_limits=True)      # Kinema, NumPy DLS backend
    rig_lim = import_rig(enforce_limits=False)    # native IK, limits on
    rig_free = import_rig(enforce_limits=False)   # native IK, limits off
    log(f"rigs: kinema={rig_k.name} kinema_numpy={rig_np.name} "
        f"native_limited={rig_lim.name} native_unlimited={rig_free.name}")

    # --- put all four in the singular configuration ---
    for rig in (rig_k, rig_np, rig_lim, rig_free):
        for name, angle in SINGULAR_POSE.items():
            if name in rig.pose.bones:
                rig.pose.bones[name].rotation_euler[1] = angle
    bpy.context.view_layer.update()

    # --- Kinema rig ---
    bpy.context.view_layer.objects.active = rig_k
    rig_k.select_set(True)
    bpy.ops.kinema.add_ik()
    rig_k.kinema_solver_mode = "PYROKI"
    # Live IK OFF for measurement. Left on, the depsgraph handler solves the rig
    # again on every view_layer.update(), interleaving with the explicit solves
    # below -- which made results depend on how many other rigs were in the
    # scene and non-reproducible between runs. The measurement must be the only
    # thing that moves the arm.
    rig_k.kinema_ik_enabled = False
    ik_bone_name = rig_k.get(builder.PROP_IK_BONE)

    solver = manager.get_solver(rig_k)
    solver.solve(rig_k, "PYROKI")
    if solver.pyroki_error is not None:
        raise RuntimeError(f"PyRoki unavailable: {solver.pyroki_error}")
    if solver.last_result.backend != "PyRoki":
        raise RuntimeError(f"not really PyRoki: {solver.last_result.backend}")
    log(f"kinema backend confirmed: {solver.last_result.backend}")

    # --- Kinema rig on the NumPy fallback, to re-derive the published claim ---
    bpy.context.view_layer.objects.active = rig_np
    rig_k.select_set(False)
    rig_np.select_set(True)
    bpy.ops.kinema.add_ik()
    rig_np.kinema_solver_mode = "NUMPY"
    rig_np.kinema_ik_enabled = False   # see rig_k above
    kinema_extra = [dict(
        tag="kinema_numpy", rig=rig_np, mode="NUMPY",
        ik_bone=rig_np.get(builder.PROP_IK_BONE),
        joints=builder.joint_bones(rig_np),
        tcp=rig_np.pose.bones[rig_np.get(builder.PROP_TCP_BONE)
                              or builder.TCP_BONE],
        solver=manager.get_solver(rig_np),
    )]
    rig_np.select_set(False)
    probe = kinema_extra[0]["solver"].solve(rig_np, "NUMPY")
    log(f"numpy backend confirmed: {probe.backend}")

    # --- native IK rigs ---
    target = bpy.data.objects.new("IK_Target", None)
    bpy.context.scene.collection.objects.link(target)
    joints_k = builder.joint_bones(rig_k)

    baselines = []
    for rig, tag, use_limits in (
        (rig_lim, "blender_limited", True),
        (rig_free, "blender_unlimited", False),
    ):
        rig.select_set(False)
        joints, tcp = configure_native_ik(rig, target, ik_solver, use_limits)
        baselines.append(dict(tag=tag, rig=rig, joints=joints, tcp=tcp))
        log(f"native IK [{tag}]: {len(joints)} joints locked to Y, "
            f"chain_count={len(joints) + 1}, solver={rig.pose.ik_solver}, "
            f"ik_limits={'on' if use_limits else 'off'}")
    joints_b = baselines[0]["joints"]

    # Fairness audit. Degrees of freedom must match exactly. Limits cannot --
    # Blender clamps to +-pi -- so assert the baseline got the widest range
    # Blender is capable of, and record how much travel that costs.
    mismatches, clamped = [], []
    for pk, pb in zip(joints_k, joints_b):
        if not (pb.lock_ik_x and pb.lock_ik_z and not pb.lock_ik_y):
            mismatches.append(f"{pb.name}: locks wrong")
        lo, hi = pk.bone.get(builder.PROP_LOWER), pk.bone.get(builder.PROP_UPPER)
        if lo is None or not pb.use_ik_limit_y:
            continue
        want_lo, want_hi = max(lo, -math.pi), min(hi, math.pi)
        if abs(pb.ik_min_y - want_lo) > 1e-6 or abs(pb.ik_max_y - want_hi) > 1e-6:
            mismatches.append(
                f"{pb.name}: {pb.ik_min_y:.4f}..{pb.ik_max_y:.4f} "
                f"!= widest-possible {want_lo:.4f}..{want_hi:.4f}")
        if hi - lo > (want_hi - want_lo) + 1e-6:
            clamped.append(f"{pb.name} {math.degrees(hi - lo):.0f}deg"
                           f"->{math.degrees(want_hi - want_lo):.0f}deg")
    if mismatches:
        raise RuntimeError(f"baseline is not fair: {mismatches}")
    log("fairness audit passed: identical DoF; limits at Blender's widest")
    if clamped:
        log(f"  travel lost to Blender's +-pi cap: {', '.join(clamped)}")

    # --- candidate paths, all centred on the singular pose ---
    bpy.context.view_layer.update()
    base = rig_k.pose.bones[rig_k.get(builder.PROP_TCP_BONE)
                            or builder.TCP_BONE].matrix.copy()

    def line(direction, span):
        out = []
        for i in range(FRAMES):
            t = (i / (FRAMES - 1) - 0.5) * 2.0     # -1 .. 1
            m = base.copy()
            m.translation = base.translation + Vector(direction) * (t * span)
            out.append(m)
        return out

    def arc(radius):
        out = []
        for i in range(FRAMES):
            t = i / (FRAMES - 1)
            ang = (t - 0.5) * math.pi
            m = base.copy()
            m.translation = base.translation + Vector(
                (math.sin(ang) * radius, 0.0, (math.cos(ang) - 1.0) * radius))
            out.append(m)
        return out

    def spin(axis: str, turns: float, frames: int = 120):
        """Rotate the goal about its own axis, position held.

        This is the case Blender's IK structurally cannot follow: ik_min/max are
        capped at +-pi, so past half a turn the wrist has nowhere left to go.
        The UR5e's wrist_3 really does travel +-2pi.
        """
        out = []
        for i in range(frames):
            t = i / (frames - 1)
            m = base.copy()
            m @= Matrix.Rotation(t * turns * 2.0 * math.pi, 4, axis)
            out.append(m)
        return out

    paths = {
        "through_singularity_y": line((0, 1, 0), 0.30),
        "through_singularity_x": line((1, 0, 0), 0.30),
        "reach_out_z": line((0, 0, 1), 0.35),
        "arc_over_top": arc(0.22),
        "reach_far_y": line((0, 1, 0), 0.70),
        "spin_tool_y_2turns": spin("Y", 2.0),
        "spin_tool_z_2turns": spin("Z", 2.0),
    }

    names = ["kinema"] + [e["tag"] for e in kinema_extra] \
        + [b["tag"] for b in baselines]
    fields = ["path", "frame", "backend"]
    for n in names:
        fields += [f"{n}_err_mm", f"{n}_jump_deg", f"{n}_violation_deg"]

    log(f"writing {out_csv}")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        summaries = []
        for tag, path in paths.items():
            rows = run_path(rig_k, ik_bone_name, joints_k, baselines, target,
                            path, tag, writer, kinema_extra)
            summaries.append(summarise(rows, tag, names))

    # Saved after the sweep, so the four rigs are left in the state the last
    # path put them in -- open it to inspect the fairness setup by hand.
    save_blend("sweep")

    print("SUMMARY_JSON " + json.dumps(summaries), flush=True)
    return 0


try:
    code = main()
except Exception:
    import traceback
    traceback.print_exc()
    code = 1
log(f"exit {code}")
sys.exit(code)
