# Preferences

**Edit → Preferences → Add-ons → Kinema**, then expand the entry.

> 📷 *Screenshot: the Kinema add-on preferences.*

## Default Solver

Which IK backend new rigs start with. Existing rigs keep whatever they were set to — this
is the default for the *next* one.

| Value | Behaviour |
|---|---|
| **PyRoki** (default) | Limit- and singularity-aware solver. Recommended. |
| **NumPy** | Lightweight damped least squares. No compile step, less robust near singularities. |
| **Off** | New rigs get no IK; they behave as plain FK armatures. |

Setting this to **Off** is reasonable if you mostly do FK work and do not want the solver
compiling on rigs you were not going to solve. You can still add IK per rig afterwards.

See [the two solvers](../concepts/ik.md#the-two-solvers).

## Preload Solver in Background

*Default: on.*

Imports the solver stack on a worker thread when Blender starts, rather than waiting until
you first need it.

It costs a few seconds of background work at startup and removes the corresponding pause
from your first interaction. It does **not** eliminate the per-robot compile — that is a
separate cost, paid when you add an IK target. See
[why the first solve is slow](../concepts/ik.md#why-the-first-solve-is-slow).

Turn it off if you use Blender for non-robot work most of the time and would rather not
pay the startup cost.

## Solve Budget (ms)

*Default: 33. Range: 4–1000.*

If a live solve takes longer than this, Kinema pauses live updates and says so in the IK
panel.

33 ms is roughly one frame at 30 fps — the point past which dragging the IK target stops
feeling connected to the viewport.

Raise it for large rigs where you would rather have slow live feedback than none. Lower it
if you want the guard to trip sooner. Baking ignores this entirely: it solves every frame
however long that takes.

See [the solve budget](../concepts/ik.md#the-solve-budget).

## Robot Cache

*Default: blank.*

Where downloaded robot descriptions are stored. Blank means the standard location used by
the underlying `robot_descriptions` library.

Point this somewhere specific if you want the cache on a fast drive, want it shared with
existing tooling, or want it somewhere you can clear easily. Robots already downloaded to
the old location are not moved — they will simply be fetched again into the new one on
first use.

See [Robot catalog](catalog.md).

## Debug Logging

*Default: off.*

Prints solver diagnostics to the system console.

Normally the solver's internal logging is silenced, because it emits several lines on
every single solve — which, with live IK running, is several lines per frame.

Turn this on when investigating a problem or filing a bug, and open the console first:

- **Windows** — Window → Toggle System Console
- **macOS / Linux** — launch Blender from a terminal to see the output

## Dependency report

The preferences also list every component Kinema depends on and whether it loaded, with a
version number or the import error.

This is the detailed version of the [Solver panel](sidebar.md#solver) status, and it is the
right thing to screenshot when reporting a problem. Ten entries, all of which should read
`ok`:

`numpy`, `scipy`, `jax`, `jaxlib`, `jaxls`, `pyroki`, `yourdfpy`, `collada`,
`robot_descriptions`, `trimesh`.
