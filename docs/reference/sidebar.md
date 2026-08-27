# The Kinema sidebar

Everything Kinema does lives in the 3D viewport sidebar. Press <kbd>N</kbd> and click the
**Kinema** tab.

There are five panels. The last four only show anything useful when a Kinema rig is
selected.

> 📷 *Screenshot: the full Kinema sidebar with all panels expanded.*

## Kinema

The top-level panel. Always available.

| Control | What it does |
|---|---|
| **Import from Catalog…** | Open the [186-robot picker](catalog.md) |
| **Import URDF File…** | [Import a local file](../tutorials/import-your-own.md) — URDF, xacro or MJCF |

With a Kinema rig selected it also shows the robot's name and its degree-of-freedom count
— `6 DoF` for a typical industrial arm.

With nothing selected it says **No robot selected**. That is the empty state, not an
error.

## Joints (FK)

One slider per movable joint, labelled with the joint's real name from the robot's
description file. Each is clamped to that joint's true range of motion.

| Control | What it does |
|---|---|
| **Rest Pose** | Return every joint to zero — the robot's documented home configuration |
| **Key All** | Insert a keyframe on every joint channel at the current frame |

A rig with no movable joints — a fixed sensor mount, say — says so instead.

See [Pose a robot by hand](../tutorials/pose-fk.md).

## Tool Centre Point

The [TCP](../concepts/tcp.md) is the working point of the robot: the tip of the tool,
the thing IK aims.

With a TCP on the rig, the panel shows which link it is attached to and its live X/Y/Z
position, plus:

| Control | What it does |
|---|---|
| **Move TCP to Active Bone** | Relocate the TCP onto the selected bone |

With no TCP, the panel offers **Create TCP from Active Bone** instead.

Both require a bone to be selected in Pose Mode, and both refuse if you select the TCP
itself.

## Inverse Kinematics

Before you add a target, this panel has one button:

| Control | What it does |
|---|---|
| **Add IK Target** | Create a keyframable control at the TCP |

Adding a target compiles the solver for this robot — roughly 15 seconds, once. It requires
a TCP.

Once a target exists:

| Control | What it does |
|---|---|
| **Live IK** | Toggle continuous solving as the target moves. Enabled when the target is created. |
| **Solver** | PyRoki, NumPy or Off — see [the two solvers](../concepts/ik.md#the-two-solvers) |
| **Snap to Tool** | Move the target back onto the tool's current position |
| **✕** | Remove the IK target; the rig returns to plain FK |
| **Bake to Keyframes** | [Solve every frame and key the joints](../tutorials/bake.md) |

Below sits a readout box:

- **Last solve: N ms** — with a tick if within the solve budget, a warning if not
- **Over budget; live updates paused** — the guard described in
  [the solve budget](../concepts/ik.md#the-solve-budget)
- A one-line summary of how the last solve converged
- **PyRoki unavailable for this rig** plus the reason, if the good solver could not be
  used for this particular robot

See [Animate with an IK target](../tutorials/animate-ik.md).

## Solver

Global solver status, independent of which robot is selected. This is the panel to check
when something is wrong.

| It says | It means |
|---|---|
| ✓ **PyRoki ready** | The full solver is loaded. This is what you want. |
| **Solver loading…** / Using NumPy fallback | Still importing in the background. Wait a moment. |
| **PyRoki unavailable** + an error + Using NumPy fallback | The solver stack failed to load. The add-on still works, with the simpler solver. |

| Control | What it does |
|---|---|
| **Re-check** | Re-run the dependency check and refresh this panel |

The error line is the actual import failure, truncated to fit. It is the most useful piece
of information you can give in a bug report — see
[Solver unavailable](../troubleshooting.md#solver-unavailable).

## Menu entries

Kinema also adds two entries outside the sidebar, under **File → Import**:

| Entry | What it does |
|---|---|
| **Robot URDF (.urdf/.xacro)** | Same as Import URDF File… |
| **COLLADA (.dae)** | Import a `.dae` mesh — Blender 5.0 removed its own importer |
