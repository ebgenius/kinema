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

## Several ways to reach one pose

A six-axis arm can usually put its tool in one place, facing one way, up to eight different
ways: elbow up or elbow down, wrist flipped or not, base facing the target or swung round
behind. These are the robot's **configurations**. The solver lands on whichever is nearest
where the arm already was, which is usually what you want. But it's how a shot ends up with
the elbow on the wrong side of the robot.

**Find Solutions**, in the [Inverse Kinematics panel](../reference/sidebar.md#configuration),
looks for the others. PyRoki converges to the solution nearest its starting point rather
than listing them all, so the others are *found*, not derived:

- start the solver from configurations scattered across the joint limits;
- keep the ones that reach the goal;
- fold away duplicates.

What it shows is a lower bound. Flipping between two configurations is two keyed poses,
which is what a robot programmer would do anyway.

## Spare joints: rails and the elbow

A tool pose has six numbers, so a chain of six joints has, in general, a finite set of ways
to reach it. With more joints than that, the arm can reach one pose a continuous family of
ways. What the spare joint *is* decides the right control, so there are two.

**Held joints.** Most redundant robots in a cell are a six-axis arm on a rail or a gantry.
The extra axis isn't something to solve for: an operator places it, and the arm reaches
from wherever it stands. So a joint can be **held**:

- click the pin beside its slider in [Joints (FK)](../reference/sidebar.md#joints-fk);
- move it with that slider, or by dragging its bone;
- IK then solves the rest of the chain with the tool where it is.

**Add IK Target** holds slides for you. While a chain has more than six unheld joints, its
prismatic joints are held, base first. So an arm on a rail gets its rail held. A turntable
can't be told apart from a seventh arm joint, so pin that one yourself. The pin is
keyframable.

**The elbow swivel.** On a seven-axis arm the spare freedom *is* the elbow. With the tool
held still, the elbow can swing round the line from shoulder to wrist. **Add Elbow Swivel**
puts a ring on that line, and turning the ring swings the elbow. It's the same idea as the
control that steers a character's elbow once IK has placed the hand. The ring's one
rotation channel is the elbow's angle, so it keys as a single curve.

The swivel aims the elbow at a point it can actually reach, on the circle it sweeps. So the
tool doesn't pay for it: on a seven-axis test arm the tool holds within 0.0003 mm while the
elbow swings. It needs the PyRoki solver, and it's offered only while more than six joints
are left to IK.

## Joint speed

Every motor has a top speed, and most robot descriptions record it for each joint. Blender
doesn't know that. It will play back a joint keyed through half a turn in two frames as
smoothly as one given two seconds, and the real robot can't do the first.

Kinema checks the frame on screen against those limits. A joint moving faster than its
limit is flagged in red on its slider and on its dial in the viewport. See
[Velocity Limits](../reference/sidebar.md#velocity-limits) for where the limits come from
and how speed is measured. It is a warning, not a limit: fix it by retiming the move.

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

- **The first time in a session there is a second cost**, before the compile: importing the
  solver stack itself, 2–5 seconds. It happens on first use rather than at Blender startup,
  so a session where you never touch a robot never pays it.
- Loading a *second* robot of a different structure compiles again. Same robot, same
  session, no recompile.
- **Adding an elbow swivel** compiles once more, because an elbow goal makes a different
  problem. Holding or releasing a joint costs nothing: that cost is always part of the
  compiled problem.

## The solve budget

Kinema measures every solve. If one exceeds the budget — 33 ms by default — live updates
pause and the panel says **Over budget; live updates paused**.

This is a deliberate guard. A solve slower than a frame makes the viewport feel broken,
and a fifty-joint humanoid is a genuinely bigger problem than a six-jointed arm.

Options when you hit it: raise the budget in
[Preferences](../reference/preferences.md), switch to the NumPy solver, or keep working
with live updates off and [bake](../tutorials/bake.md) at the end. Baking is unaffected by
the budget — it solves every frame however long that takes.
