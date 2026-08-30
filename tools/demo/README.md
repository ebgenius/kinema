# Demo and IK measurement

Two GIFs for the extension listing, and the measurements behind them. Everything
here is reproducible; the point is that the claims can be checked rather than
taken on trust.

| Asset | Shows | Size |
|---|---|---|
| `kinema-ik.gif` | UR5e, side by side against Blender's built-in IK, on a two-turn tool spin | 0.44 MB |
| `kinema-nullspace.gif` | Franka Panda, 7 DoF, reconfiguring 90° around a fixed tool pose | 1.03 MB |

Both are quantised against a single shared palette with dithering off, which is
what keeps them small — see the note in `make_gif.py`. Palette size is the last
argument to either composer.

```bash
# 1. the comparison
blender --background --python tools/dev_bootstrap.py \
        --python tools/demo/sweep.py -- measurements.csv LEGACY
blender --background --python tools/dev_bootstrap.py \
        --python tools/demo/render_demo.py -- frames/
uv run --with pillow python tools/demo/make_gif.py frames/ kinema-ik.gif 2 720

# 2. the null-space sweep
blender --background --python tools/dev_bootstrap.py \
        --python tools/demo/nullspace_demo.py -- nsframes/
uv run --with pillow python tools/demo/make_gif_single.py nsframes/ \
        kinema-nullspace.gif 2 620 "Same tool pose, 90 degrees of elbow swivel" \
        "tool held to 0.0005 mm - Franka Panda, 7 DoF"

# 3. feasibility numbers for both (branch count, null-space span)
#    optional trailing argument is the seed count, default 40
blender --background --python tools/dev_bootstrap.py \
        --python tools/demo/branches.py -- 40
```

Needs `ur5e_description` and `panda_mj_description` in the robot-descriptions
cache. `dev_bootstrap.py` supplies `KINEMA_EXT_ID` and the dev site-packages, as
for `dev.py test`.

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

## Solution branches and the null space

`branches.py` measures two things Blender's IK cannot express at all. It returns
whichever single configuration its iterative solver lands on; there is no way to
ask it for a different one.

**Multiple branches for one pose.** Seeding PyRoki from 40 random configurations
and keeping the distinct results finds **at least 15 different solutions** for a
single UR5e tool pose, every one converging to ~0.0001 mm. A 6R arm has at most
eight *analytic* solutions — the extra ones are multi-turn variants, sitting at
joint angles like −241° and +205°. Those exist only because Kinema carries the
real ±360° limits, and are exactly the configurations Blender's ±180° IK cannot
represent. Same root cause as the wrist flip above.

The count is a **lower bound, not an enumeration** — it is whatever those seeds
happened to reach, and more seeds reach more:

| Seeds | Distinct solutions |
|---|---|
| 40 | 15 |
| 120 | 34 |

So quote it as "at least 15", never "15 solutions". `branches.py` takes the seed
count as a trailing argument if you want to push it further.

**Null-space sweep.** A 7-DoF arm has a one-dimensional family of configurations
for any reachable tool pose. Biasing the seed along it and re-solving walks that
family: on the Panda, **91.4° of arm motion at 0.0019 mm and 0.0000°**.
`nullspace_demo.py` renders it, with a static marker sphere placed once at the
goal — if the tool drifts off the marker, the demo is lying.

Both use only the public solve path (pose the arm at a seed, then solve). PyRoki's
`rest_cost` biases toward the seed, so the seed is what selects the branch. No
add-on code was added for either.

### Put the TCP on the flange, and measure orientation

The Panda imports with its **TCP on `right_finger`**, which leaves both gripper
joints *inside* the IK chain: 9 DoF against a 6-DoF task. The solver then holds
the fingertip exactly while spinning the whole hand around it — and a
position-only metric reports 0.0005 mm while the flange visibly rotates on
screen. Two mistakes compounding: the wrong task frame, and a metric too weak to
notice.

Both scripts now move the TCP to `joint7` with `kinema.set_tcp` before adding IK,
leaving exactly the seven arm joints, and both measure **position and
orientation**. `nullspace_demo.py` additionally tracks the flange's own world
rotation across the sweep, which is the thing a viewer actually watches:

```
tool position error:      max 0.0005 mm
tool orientation error:   max 0.0000 deg
flange rotation in world: max 0.0000 deg   (HOLDS STILL)
```

## Saved scenes

Each script writes the scene it built to `tools/demo/blend/`, so you can open the
exact result of a headless run and scrub it rather than trusting a log line:
`sweep.blend`, `ik-comparison.blend`, `nullspace.blend`, `branches.blend`,
`branches-nullspace.blend`.

They are **gitignored** — tens of megabytes of packed robot meshes each, and all
regenerable by re-running the script. Set `KINEMA_DEMO_BLEND_DIR` to write them
somewhere else. Saving is best-effort and never fails a run.

## Things this does not show

- **`spin_tool_z_2turns`** is in the sweep and is *not* in the GIF: Kinema also
  degrades there (124 mm worst error). Included in `measurements.csv` so the bad
  case is on the record.
- **`reach_far_y`** puts the target outside the workspace. Both solvers miss;
  nobody wins.
- Only Blender's `LEGACY` solver was used. `ITASC` has its own damping parameters
  and has not been measured.
- One robot, one seed pose. These are not general claims about either solver.
