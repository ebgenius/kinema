# Troubleshooting

## Solver unavailable

**Symptom:** the [Solver panel](reference/sidebar.md#solver) reads **PyRoki unavailable**,
with an error line underneath and **Using NumPy fallback**.

**What it means:** the full solver stack failed to import. Kinema still works — IK is
being solved by the simpler NumPy backend — but you have lost the singularity- and
limit-aware solving that is the reason to use Kinema.

**Diagnose it.** The error line is the actual Python import failure. Read it. Then open
**Edit → Preferences → Add-ons → Kinema** for the full dependency report: ten components,
each `ok` with a version or `missing` with the reason.

### Cause 1: the wrong build for your platform

By far the most common. Kinema bundles a compiled maths library, so each platform gets its
own zip. Installing the Linux zip on Windows, or an Intel build on Apple Silicon, produces
exactly this failure.

Errors that point here mention `jaxlib`, `jax`, or a version mismatch between them:

```
RuntimeError: jaxlib is version 0.7.0, but this version of jax
requires version >= 0.11.1.
```

**Fix:** remove the extension, download the zip matching your OS from the
[releases page](https://github.com/ebgenius/kinema/releases), and install that. See
[Install](getting-started/install.md).

### Cause 2: a partial install

On Windows, an install interrupted by the long-path limit leaves some components on disk
and some missing. The report will show `missing` against several entries at once.

**Fix:** [enable long paths](#windows-winerror-206), remove the extension, and reinstall.

### Cause 3: it is still loading

Right after Blender starts, the panel may say **Solver loading…**. The stack imports on a
background thread and takes a few seconds.

**Fix:** wait, then click **Re-check**.

### Still stuck?

Click **Re-check** first — it re-runs the whole dependency check. If the report still shows
failures, that screenshot plus the error line is exactly what to put in a
[bug report](#reporting-a-bug).

## Windows: WinError 206

**Symptom:** installing the extension fails partway through, with `WinError 206` or a
message about a path being too long.

**Cause:** the bundled solver unpacks into deeply nested directories — paths like
`jax/_src/internal_test_util/export_back_compat_test_data/…`. Windows limits paths to 260
characters unless long-path support is switched on.

**Fix**, either one:

- Enable long paths system-wide. Set this registry value to `1`:

    ```
    HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled
    ```

    Then restart, remove any partial install, and reinstall the extension.

- Or keep Blender's configuration directory shallow — a path near the root of a drive
  rather than nested inside several folders of user data.

## Blender freezes for about 15 seconds

**Symptom:** clicking **Add IK Target** locks Blender up, with a wait cursor.

**This is expected, once per robot.** The solver compiles itself for this specific robot's
structure the first time it runs. Kinema pays that cost deliberately at the moment you add
the target, rather than letting it land on your first drag of the control.

Subsequent solves take milliseconds. Loading a different robot compiles again.

See [why the first solve is slow](concepts/ik.md#why-the-first-solve-is-slow).

## Over budget; live updates paused

**Symptom:** the IK panel shows a warning and the rig stops following the target.

**Cause:** a solve took longer than the solve budget (33 ms by default), so live updates
paused to keep the viewport usable.

**Fix**, in order of preference:

1. Raise **Solve Budget** in [Preferences](reference/preferences.md#solve-budget-ms)
2. Switch the rig's **Solver** to NumPy — faster, less robust
3. Keep working with live updates off and [bake](tutorials/bake.md) at the end; baking
   ignores the budget entirely

Very high-DoF rigs — a fifty-joint humanoid — hit this legitimately. A six-jointed arm
should not, and if it does, check whether PyRoki is actually loaded.

## The robot imported but I cannot see it

**Cause:** the meshes did not resolve. The rig is correct — joints, limits and IK all work
— but there is no visible geometry.

URDF files reference meshes by path, usually with a `package://` URL. Kinema resolves
these by searching the description file's own directory tree, which needs the tree to be
there.

**Fix:** make sure you have the **whole robot package**, not just the `.urdf` file. The
`meshes/` folder must be present alongside it. Downloading a single description file from
a repository web page is the usual mistake.

## The parts pile up at the origin, then snap into place

**Not a bug.** Meshes are loaded one at a time and parented to their bones before any of them
is positioned, so for a moment the robot is a heap at the world origin. It resolves itself on
the next redraw.

## A second import is refused while one is running

**Not a bug.** Kinema imports one robot at a time. Wait for the first to finish, or press
`Esc` to cancel it.

## There are fewer bones than I expected

**Not a bug.** Two bone collections are hidden on a new rig:

- `Kinema/IK` — until you add an IK target
- `Kinema/Mechanism` — internal bones that are not controls

Unhide them in the Properties editor under Object Data → Bone Collections. See
[bone collections](concepts/links-and-joints.md#bone-collections).

Also worth knowing: in a robot rig, **bones are joints, not limbs**. A six-jointed arm has
six control bones, not one per visible segment.

## Move TCP to Active Bone raises an AttributeError

**Symptom:** on **0.1.0 only**, clicking **Move TCP to Active Bone** (or **Create TCP from
Active Bone**) prints a traceback ending in:

```
AttributeError: 'Bone' object has no attribute 'select'
```

**Cause:** the button acted on the bone that was *active in Pose Mode*, and did not cope with
there being none. A freshly imported robot has no bone selected, so clicking it straight after
an import hit this every time.

**Fix:** update to 0.2.0, where the TCP's bone is chosen from a **Parent Bone** field in the
panel and no mode change is needed. On 0.1.0, enter Pose Mode and click a bone first.

## A link mesh ended up in the wrong place

**Symptom:** part of the robot is floating off on its own, or rotated, and **Rest Pose** does
not bring it back.

**Cause:** the robot's visual meshes are separate objects parented to the bones. Grabbing one
by accident moves it, and *Rest Pose* returns the **joints** — a mesh is not a joint, so it
stays where it was left.

**Fix:** **Reset Meshes**, in the Joints (FK) panel beside *Rest Pose*. It puts every link
mesh back where the importer placed it, and leaves anything you attached yourself alone.

From 0.2.0 the meshes are also locked, so this is harder to do by accident. A rig imported
with 0.1.0 has no record of where its meshes belong and cannot be repaired — re-import it.

## Add IK Target is refused

**Symptom:** "This rig has no TCP; create one first".

**Cause:** the IK target is created at the [tool centre point](concepts/tcp.md), so there
has to be one. This happens if you imported with **Create TCP** turned off.

**Fix:** enter Pose Mode, select the bone at the end of the chain, and click **Create TCP
from Active Bone** in the Tool Centre Point panel.

## The arm stops short of the target

**Cause:** the target is unreachable — outside the robot's workspace, or blocked by a joint
limit. The solve does not converge and the panel says so.

**Fix:** move the target inside the robot's reach, or move the rig's root bone to bring the
robot closer. If you believe the limits are wrong, you can re-import with **Enforce Joint
Limits** off — but then you are animating poses the real machine cannot achieve.

See [reach](concepts/ik.md#reach).

## The IK target and the robot disagree

**Cause:** they have drifted apart — you posed the arm with FK while the solver was Off, or
a solve failed to converge.

**Fix:** **Snap to Tool**, which puts the target back on the tool.

## MTL errors in the console

**Symptom:** importing an MJCF robot prints errors about missing `.mtl` files.

**Harmless.** MuJoCo robots often ship OBJ meshes without their material companion files,
and Blender's OBJ importer complains. MJCF carries its own colours, so nothing is actually
missing. Ignore it.

## A robot will not import at all

**Ball joints are rejected.** A 3-DoF spherical joint has no honest single-axis bone
equivalent. Exactly one catalog robot — **Cassie** — is affected.

For a local file, the other common causes are a xacro missing a required argument (the
error names it), or a file that is not actually a robot description.

## The catalog will not download

Importing from the catalog fetches the robot the first time. Failures usually mean no
network, a firewall blocking the host, or an upstream repository that has moved.

The cache is safe to delete — the next import re-fetches. See
[Robot catalog](reference/catalog.md#downloads-and-the-cache).

## The baked animation looks wrong

**Cause:** frames where the solve did not converge. Kinema reports the count of failed
frames when the bake finishes.

**Fix:** go to those frames, move the IK target into reach, and bake again.

If the motion is wrong specifically *between* keys, you may have baked with **Step** above
1. Interpolating between two solved poses does not always follow a path the robot can
take. Bake every frame.

## Reporting a bug

Please include:

1. Your OS and which zip you installed
2. Your Blender version
3. A screenshot of the dependency report in
   [Preferences](reference/preferences.md#dependency-report)
4. The robot involved — catalog name, or the format if it is your own file
5. Console output with **Debug Logging** enabled
   ([how to open the console](reference/preferences.md#debug-logging))

[Open an issue](https://github.com/ebgenius/kinema/issues).
