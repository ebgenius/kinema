# Demo and IK measurement

Produces `kinema-ik.gif`, the side-by-side comparison used on the extension
listing, and the measurements behind it. Everything here is reproducible; the
point is that the claim in the GIF can be checked rather than taken on trust.

```bash
# measure (writes measurements.csv)
blender --background --python tools/dev_bootstrap.py \
        --python tools/demo/sweep.py -- measurements.csv LEGACY

# render 120 frames
blender --background --python tools/dev_bootstrap.py \
        --python tools/demo/render_demo.py -- frames/

# compose
uv run --with pillow python tools/demo/make_gif.py frames/ kinema-ik.gif 2 720
```

Needs `ur5e_description` in the robot-descriptions cache. `dev_bootstrap.py`
supplies `KINEMA_EXT_ID` and the dev site-packages, as for `dev.py test`.

## What the GIF shows

One tool path — two full turns of the wrist — driving two identical UR5e rigs.
The left is Kinema's PyRoki backend; the right is Blender's own IK constraint.
Blender's wrist snaps ~176° in a single frame, twice. Kinema turns through it
smoothly.

## Making the baseline fair

The comparison is worthless if the baseline is crippled, so:

| Setting | Why it matters |
|---|---|
| `lock_ik_x` / `lock_ik_z` on every joint | Kinema's bones are 1-DoF via *transform* locks, which Blender's IK solver ignores entirely. Without these it solves each joint as a free 3-DoF ball joint and produces motion the robot cannot perform — flattering to Kinema, and meaningless. |
| `use_ik_limit_y`, `ik_min_y` / `ik_max_y` from `kinema_lower` / `kinema_upper` | the same URDF limits Kinema enforces, in the field Blender's solver actually reads |
| Kinema's `LIMIT_ROTATION` constraints removed on the baseline rig | otherwise limits are enforced twice, in two different places, and the tool misses for the wrong reason |
| TCP's IK DoF fully locked | the TCP is a rigid tool offset, not a joint. Leaving it free hands the baseline 9 DoF against the robot's real 6. |
| `use_rotation = True` on the IK constraint | Kinema solves a full 6-DoF pose goal. Position-only would be an easier problem. |
| both scored at the last bone's **tail** | Blender's IK drives the tail; Kinema drives the tool frame at the head. Scoring head-to-target charges the baseline a fixed offset — the TCP bone's length — for work it was never asked to do. |
| live IK **off**, motion baked | left on, Kinema's depsgraph handler re-solves on every `view_layer.update()`, interleaving with the explicit solves. That made results depend on how many rigs were in the scene and non-reproducible between runs. |

`sweep.py` asserts the DoF and limits actually match before it measures anything,
and asserts Kinema is really running PyRoki — `manager.solve` silently falls back
to NumPy when PyRoki throws, so a "PyRoki" rig can quietly be a NumPy one.

Two runs of `sweep.py` produce byte-identical CSVs. If they ever stop doing so,
something is solving that should not be.

## Results

From `measurements.csv`, UR5e, Blender's `LEGACY` IK solver, worst case per path:

| Path | Kinema max jump | Blender max jump | Kinema max err | Blender max err |
|---|---|---|---|---|
| through wrist singularity (Y) | 2.01° | 2.01° | 0.00 mm | 0.05 mm |
| through wrist singularity (X) | 8.02° | 8.46° | 0.00 mm | 0.09 mm |
| reach out | 2.08° | 2.18° | 0.00 mm | 0.05 mm |
| arc over top | 2.09° | 2.14° | 0.00 mm | 0.09 mm |
| **two-turn tool spin** | **7.31°** | **176.46°** | **0.00 mm** | 0.08 mm |

**On ordinary motion the two are equivalent.** Properly configured, Blender's
built-in IK tracks as accurately as Kinema and just as smoothly. That is worth
stating plainly rather than hiding: the GIF is not a claim that Kinema's IK is
better everywhere.

**The difference is multi-turn rotation, and it is structural.** Blender's
`ik_min_y` / `ik_max_y` are hard-clamped to ±π. Five of the UR5e's six joints
travel ±2π, so the baseline gets half the real range no matter how it is
configured:

```
shoulder_pan 720°->360°   shoulder_lift 720°->360°   wrist_1 720°->360°
wrist_2      720°->360°   wrist_3       720°->360°
```

Past half a turn the wrist has nowhere to go and flips. The two flips land at
frames 30 and 90 of 120 — exactly the half-turn and turn-and-a-half marks, which
is what makes this a property of the representation rather than a tuned result.
Enabling or disabling Blender's IK limits changes nothing here, and both variants
are measured.

## Things this does not show

- **`spin_tool_z_2turns`** is in the sweep and is *not* in the GIF: Kinema also
  degrades there (124 mm worst error). Included in `measurements.csv` so the bad
  case is on the record.
- **`reach_far_y`** puts the target outside the workspace. Both solvers miss;
  nobody wins.
- Only Blender's `LEGACY` solver was used. `ITASC` has its own damping parameters
  and has not been measured.
- One robot, one seed pose. These are not general claims about either solver.
