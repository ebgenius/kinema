# Forward and inverse kinematics

Two directions of the same problem.

**Forward kinematics (FK)** — you know the joint angles, you want the tool position. This
is easy: apply each rotation down the chain and see where you end up. There is exactly one
answer, always.

**Inverse kinematics (IK)** — you know where you want the tool, you want the joint angles.
This is hard. There may be no answer, one answer, several answers, or infinitely many.

Blender animators use both constantly, usually without naming them: posing a bone chain by
hand is FK, dragging an IK handle is IK.

## Why robots need a different solver

Blender ships an IK solver, and it is good at what it was built for — character limbs,
where "looks right" is the acceptance test. Robots break three of its assumptions.

**Joint limits are hard, not soft.** A character's elbow bending slightly too far is a
rendering artefact. A robot joint driven past its stop is a pose the machine physically
cannot achieve, which makes the animation wrong in a way a client will notice.

**Singularities are real geometry.** At certain configurations a robot loses the ability
to move in some direction, and near them small tool movements demand enormous joint
movements. A character rig glosses over this. A robot rig that glosses over it produces
motion the machine cannot perform.

**Joints count turns.** A wrist at +400° differs from one at +40°, and unwinding matters
when a cable is attached. Solvers that wrap angles into a single turn throw that away.

Kinema uses [PyRoki](https://github.com/chungmin99/pyroki), a solver built for robots,
which treats all three correctly.

## Singularities

A **singularity** is a configuration where the robot loses a degree of freedom — where two
joint axes line up and, between them, can no longer produce motion in some direction.

The clearest example is a **fully extended arm**. Reach your own arm straight out and try
to move your hand further away without moving your shoulder. You cannot: at full extension
the arm has run out of that direction entirely.

Approaching a singularity, the maths gets violent. To move the tool a millimetre, joints
must swing through large angles, and as you get closer the required joint speed heads for
infinity. On a real machine this is an emergency stop. In an animation it is a wrist that
snaps round in a single frame.

Common singularities on a 6-DoF arm:

- **Full extension** — arm straight out, at the edge of its reach
- **Wrist alignment** — two wrist axes collinear, so they fight over the same rotation
- **Shoulder alignment** — the tool directly above or below the shoulder axis

What to do about them:

- **Route around.** Keep the tool path away from the edge of reach. Most singularities sit
  at the boundary of the workspace; staying comfortably inside it avoids them.
- **Watch for the tell.** Sudden fast joint motion for slow tool motion means you are near
  one.
- **Keyframe through.** If the shot must pass through, key on both sides and accept that
  the transit is fast — real robots do exactly this.

PyRoki handles singularities gracefully rather than blowing up: it prefers a small,
sensible joint motion over an exact solution when the exact one would require an absurd
one.

## Joint limits

Every revolute joint has a range. Kinema applies these on import — the
[FK sliders](../tutorials/pose-fk.md) stop where the machine stops, and IK will not solve
through a limit.

This means an IK target you can see is not necessarily a target the robot can reach. When
it cannot, the tool stops short and the panel reports that the solve did not converge.
That is the rig telling you something true.

You can disable limits at import time. Do that only when you know the file's limits are
wrong — with them off, you can animate poses the real machine cannot achieve.

## Reach

A robot's **workspace** is the set of points its tool can get to. It is not a sphere: it
has a hole in the middle (the robot cannot fold into itself), a hard outer boundary, and
dead zones behind the base.

Targets outside the workspace do not converge. The solver gets as close as it can and
stops. If the tool consistently stops short, the target is out of reach — move it, or move
the robot's root closer.

## The two solvers

| | PyRoki | NumPy |
|---|---|---|
| Joint limits | Enforced | Enforced |
| Singularities | Handled robustly | Damped, less gracefully |
| Multi-turn joints | Tracked | Tracked |
| First solve | ~15 s compile | Instant |
| Warm solve | ~5–20 ms | ~1 ms |
| Availability | Needs the bundled solver stack | Always |

**Use PyRoki.** It is the default and it is the reason Kinema exists.

**NumPy** is a damped least-squares fallback — a classical method that is fast and
adequate away from singularities, and noticeably worse near them. It exists so that the
add-on still functions when the full stack cannot load, and as an option on very large
rigs where PyRoki's per-solve cost is too high for live interaction.

If the Solver panel says PyRoki is unavailable, that is a problem to fix rather than a
choice you made — see [Troubleshooting](../troubleshooting.md#solver-unavailable).

## Why the first solve is slow

The first solve for a given robot takes roughly **15 seconds**. Every solve after that
takes a few milliseconds. This is expected, and it is not a performance bug.

PyRoki is built on JAX, which compiles numerical code into optimised machine code the
first time it runs — a technique called just-in-time compilation. The compilation is
specific to the shape of the problem, which means specific to *this robot*: its joint
count, its structure, its limits.

So the cost is paid once per robot, and Kinema pays it deliberately at the moment you
click **Add IK Target**, behind a wait cursor. The alternative would be a 15-second freeze
the first time you drag the control, which is worse.

Two things affect when you feel it:

- **Preload Solver in Background** (on by default) imports the solver stack on a worker
  thread at Blender startup, removing part of the delay from your first interaction.
- Loading a *second* robot of a different structure compiles again. Same robot, same
  session, no recompile.

## The solve budget

Kinema measures every solve. If one exceeds the budget — 33 ms by default — live updates
pause and the panel says **Over budget; live updates paused**.

This is a deliberate guard. A solve slower than a frame makes the viewport feel broken,
and a fifty-joint humanoid is a genuinely bigger problem than a six-jointed arm.

Options when you hit it: raise the budget in
[Preferences](../reference/preferences.md), switch to the NumPy solver, or keep working
with live updates off and [bake](../tutorials/bake.md) at the end. Baking is unaffected by
the budget — it solves every frame however long that takes.
