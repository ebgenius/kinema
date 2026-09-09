# Kinema

**Animation-ready robot rigs in Blender.** Import a robot description and get a single clean
armature you can actually animate — with IK that understands singularities, joint limits and
multi-turn joints. A built-in catalogue of 186 real robots says where to find one.

> Status: released as [v0.3.0]. Import (URDF, xacro or MJCF), rig, pose, solve and
> bake all function, and the built extension installs and runs from a clean Blender
> profile. Nothing is downloaded, no threads are started, and no environment
> variables are written — an import blocks Blender while it works.
>
> Docs: <https://ebgenius.github.io/kinema/>

[v0.3.0]: https://github.com/ebgenius/kinema/releases/tag/v0.3.0

## Why this exists

Every Blender robotics tool today — [Phobos], [LinkForge], [urdf_importer], [blender-urdf] —
treats Blender as a *design and export* station for URDF/SDF. They optimise for producing
simulation assets.

Kinema optimises for the opposite thing: **open a .blend, find a clean rig, animate it,
hit render.** Blender is the destination, not a stop along the way.

Concretely, that means:

- **One armature.** Not armatures nested inside armatures.
- **One 1-DoF bone per joint**, with the bone's local Y aligned to the URDF joint axis, so
  every joint is a true single-axis control with real limits.
- **Bone collections** (`Kinema/FK`, `Kinema/IK`, `Kinema/TCP`, `Kinema/Mechanism`) so an
  animator sees controls, not machinery.
- **IK that behaves like a normal Blender IK control** — keyframe the TCP and move on —
  but is solved by [PyRoki] rather than Blender's built-in solver.
- **A bake step**, so the finished .blend renders anywhere, with or without Kinema.

[Phobos]: https://github.com/dfki-ric/phobos
[LinkForge]: https://extensions.blender.org/add-ons/linkforge/
[urdf_importer]: https://github.com/HoangGiang93/urdf_importer
[blender-urdf]: https://github.com/kralf/blender-urdf
[PyRoki]: https://github.com/chungmin99/pyroki

## Dressing a rig, and where IK aims

The sidebar's **Bones** list is one row per bone, and it answers the two questions that come
up as soon as a robot is imported.

**Hang something off a bone.** Pick an object or a collection in a row and a linked copy
rides that bone — a gripper on the flange, a cable harness down the forearm. It shares its
mesh and materials with what you picked, so the same harness can dress six links and still
be edited in one place, and the source can live in any scene in the file. What arrives is a
placement and nothing more: the copy is stripped of the source's animation and constraints,
so nothing reaches in from outside to move it.

The offset is measured from the bone's **head**. Blender's own bone parenting measures from
the tail, which puts a fresh attachment at the far end of the bone and makes every offset
you then dial in relative to a frame that has nothing to do with the joint. Kinema cancels
that, so the attachment's plain location/rotation/scale *is* its offset from the joint —
edit it in the panel, or grab the object and use G/R/S, and keyframe it like anything else.

Clearing a row's picker removes the copy; the **✕** button instead unparents it and leaves
it in the scene exactly where it appears, for when you want to keep what you placed.

To resize an attachment, scale **the attachment** — its scale is in the *Offset from the
Bone* block with the rest of its transform. Scaling the source object does not carry over,
because the copy shares the source's *geometry* but keeps its own transform, and that
transform is what positions it on the bone. (Applying scale on the source will not work
either: Blender refuses `Ctrl+A` on a mesh with more than one user, which is exactly what a
linked copy makes it.)

**Choose what IK aims at.** The radio button in each row points the solver at that bone, the
tool centre point included — that one is the default. This matters on redundant robots: a
Panda imports with its tool frame on a fingertip, which leaves both gripper joints inside
the chain — 9 DoF against a 6-DoF task — and the solver then holds the fingertip still while
spinning the hand around it. Aim at the flange instead and the chain is the seven arm joints
it should be. Switching snaps the IK control onto the new target, so the arm does not jump.

The choice is keyframable, so a shot can hand the goal from the wrist to the elbow part-way
through. **Key Target Bone**, beside the field in the IK panel, is the way to key it: the
target is an index rather than a quantity, and those keys have to *step* between values.
Interpolated, a hand-off from the tool point to joint 3 would pass through joints 1 and 2 on
the frames in between and solve two chains nobody asked for. Baking follows a keyed target
across the range and keys every joint that is active anywhere in it.

Switching to a bone whose link the solver has not seen before pays a one-off compile, the
same wait as the first solve after adding an IK target. A handful of recently used ones are
kept compiled, so scrubbing back and forth over a hand-off is free after the first pass.

## Teaching a job

Rigging a robot answers *how do I get one into Blender*. The **Waypoints** panel answers
what it should do.

Pose the robot, press **Record**, and give the row a name — `home`, `approach`, `pick`,
`drop`. Each waypoint stores the tool pose *and* the joint vector it was taught in, because
a pose alone does not say which of a robot's up-to-eight solutions you meant. **Go To**
restores that exact configuration rather than re-solving and landing somewhere else.

A waypoint also carries a **frame**, and that is the only ordering there is — there is no
separate list order to keep in sync. Retime a move by editing the frame in its row and
regenerating; the list redraws in time order. Two waypoints may not share a frame, since
that would ask the robot to be in two places at once; generating says which pair collide.

(The keys **Generate Motion** writes are the *output*. Dragging those in the dope sheet
retimes the animation, as it would for anything else, but the next regeneration writes the
waypoint's own frame again — so the row is where a retime belongs if you want it to stick.
Waypoint frames are not yet draggable on the timeline itself.)

Each row says how the robot *arrives* there, which is how robot programs read
(`MoveL(pick)` describes getting *to* `pick`), and is why the first waypoint needs no move
type:

- **Joint** interpolates the joints. Fast, always reachable, and the tool takes whatever
  path falls out. What you want for crossing a workspace.
- **Linear** drives the tool along the straight line between the two poses. What a process
  move needs. On a KR120 the tool holds that line to 0.002 mm over a 300 mm plunge.

**Generate Motion** writes it out as ordinary keyframes: joint channels for joint moves, the
IK goal for linear ones, and the live-IK switch keyed so each span is solved the way it
should be. Nothing about the result is special — which is the point. Blender's own graph
editor shapes it afterwards, and easing a two-key linear span changes the *speed profile
along* the straight line without bending it, because X, Y and Z share the interpolation
shape and only the parameterisation changes. That is a real robot's acceleration ramp. Give
a corner three waypoints and Bezier handles bow the path through it, which is the blend
radius a real controller would apply.

Regenerating replaces the previous motion rather than layering on it, so moving a waypoint
from frame 40 to frame 30 does not leave the robot visiting frame 40 as well. When the shot
is right, **Bake IK to Keyframes** turns the whole thing into plain joint curves that render
with Kinema uninstalled.

## Choosing how the arm reaches

A six-axis arm can put its tool in one place up to eight different ways — elbow bent one way
or the other, wrist flipped, base swung round behind. Until now you got whichever one the
solver landed on from wherever the arm happened to be, which is how a shot ends up with the
elbow on the wrong side of the robot and no obvious way to fix it.

**Find Solutions**, in the IK panel, searches for the others. PyRoki is a nonlinear solver
rather than an analytic one, so it does not enumerate branches — it converges to whichever
is nearest its seed. So they are *found*: seed from configurations scattered across the joint
limits, solve each, keep the ones that reached the goal, fold away the duplicates. That makes
it a **lower bound** and the panel says so; more seeds may find more.

The ◀ ▶ arrows cycle through what turned up. Applying one only writes joint values, so
**Key All** keyframes it and it bakes like anything else — flipping the elbow between two
frames is two keyed poses, which is what a robot programmer would do anyway.

Solutions belong to one goal pose. Move the IK target and the panel says they are stale
rather than offering configurations for a pose the robot is no longer being asked to reach.

On a KR120 at a working pose, this turns up the arm you are in plus the wrist-flipped one
— joints 1 to 3 identical, 4 to 6 turned through 180° — with the tool holding to 0.008 mm
across the switch. The other families a KR120 nominally has are ruled out by its own joint
limits at that pose, which is the correct answer rather than a shortfall of the search.

### The elbow on a redundant arm

Seven axes reach a pose an *infinite* number of ways, the elbow sweeping through a
continuous family while the tool stands still. That is a control, not a curiosity, and
Blender animators already have one for it: the pole target on a character's arm.

**Add Elbow Target** creates a bone you drag. It is a soft position goal on the elbow's link,
weighted far below the tool's — so at the default strength the elbow moves through the null
space and the tool holds to well under a millimetre. Drag it and the arm reconfigures around
a tool that stays put. It lands on the elbow itself, so adding it changes nothing until you
move it; the rig draws its bones in front of the geometry, so it is still there to grab.

It is an attractor rather than Blender's angle-reference pole, so the elbow is pulled
*toward* it rather than aimed *through* it. Close enough that the muscle memory transfers —
with one consequence worth knowing: the bone does not stick to the elbow and is not supposed
to. It has no constraint on it. Put it where you want the elbow to go, and the elbow reaches
for it as far as the arm's leftover freedom allows, then stops. On a seven-axis arm that
freedom is a single circle, so a target off that circle is approached, never met.

**Strength** is a cost weight, not a null-space projection, so turning it up does start
moving the tool: on the `arm7` fixture with the target dragged well out of reach, the tool
sits about 0.5 mm off its goal at the default 2, 3 mm at 5, and 11 mm at 10. It buys little
extra elbow travel in exchange — the null space runs out first. The solve readout shows the
tool error, so the trade is visible while you make it.

Offered only when the chain has more joints than the task needs. On a six-axis arm there is
no freedom left after a full pose, so the control would have nothing to steer, and it is
absent rather than present and inert. It also needs the PyRoki solver; the NumPy fallback
has no notion of a second goal and the panel says so instead of ignoring it quietly.

## Where the tool frame sits

The **Tool Centre Point** panel places the TCP on a joint bone and offsets it from there.
Pick the parent bone, type the offset, press *Update TCP*.

The offset is measured in that joint's own **link frame** — the flange, with its Z out of the
face — and the angles are roll, pitch and yaw about fixed X, Y and Z, which is the convention
URDF itself uses in `<origin rpy="...">`. So a tool transform copied out of a description
goes in unchanged, and `Z = 0.15` means 150 mm out of the flange rather than 150 mm up.

It usually arrives non-zero. The tool frame is the description's deepest link, which normally
sits behind one or more *fixed* joints from the last actuated one — and fixed joints get no
bone, so nothing on the rig shows that distance. The offset field is where it becomes
visible.

Bone axes are not tool axes and cannot be: a bone's +Y always runs head to tail. Kinema rides
the tool frame on the marker permuted — tool Z on the bone's Y — and reports the tool frame,
not the bone's, in the panel. The marker draws the approach axis long and arrowed so the
direction is readable at a glance.

## Requirements

Blender **5.2 LTS or newer** (embeds CPython 3.13). Everything else ships with the add-on.

## Development

Kinema is a [uv] project. The dev virtualenv pins Python 3.13 to match Blender's embedded
interpreter, which is what makes IDE debugging work without a Blender-specific extension.

```bash
uv sync --group dev                        # dev venv (Python 3.13)
uv run python tools/vendor.py              # vendor pyroki + jaxls source
uv run python tools/fetch_wheels.py        # download the wheel payload (~118 MB/platform)
uv run python tools/dev.py link            # live-link the add-on into Blender
uv run python tools/dev.py run             # launch Blender with Kinema enabled
```

Set `KINEMA_BLENDER` if Blender is not on `PATH` or you have several builds installed:

```bash
export KINEMA_BLENDER="/path/to/blender"
```

### Debugging from any IDE

`tools/dev.py debug` starts Blender with a [debugpy] DAP server. Attach from anything that
speaks DAP — VS Code, PyCharm, neovim-dap, Helix. No Blender IDE plugin required.

```bash
uv run python tools/dev.py debug --port 5678
```

VS Code `.vscode/launch.json`:

```json
{ "type": "debugpy", "request": "attach",
  "connect": { "host": "127.0.0.1", "port": 5678 } }
```

PyCharm: *Run → Attach to Process*, or a Python Debug Server on port 5678.

### Tests

```bash
uv run pytest                              # parsers, no Blender needed
uv run python tools/dev.py test            # in-Blender suite (real bpy)
```

Run the non-Blender tests with UTF-8 mode enabled (`PYTHONUTF8=1`) on Windows. Blender runs
its interpreter in UTF-8 mode already, but the system default is cp1252, and some ROS xacro
files contain non-ASCII bytes that fail to decode under it.

### Building

```bash
uv run python tools/vendor.py              # vendored solver source, at the pinned commits
uv run python tools/fetch_wheels.py        # all three platforms (~341 MB)
uv run python tools/dev.py validate
uv run python tools/dev.py build           # per-platform zips into dist/
```

`vendor/` and `wheels/` are both gitignored, so a fresh clone has neither and the first
two commands are not optional. They are one-time setup rather than per-build steps —
re-run `vendor.py` only when its pins move, and `fetch_wheels.py` only when the payload
should change. `dev.py build` refuses to run if either directory is missing, stale
against its pin, or holds one package at two versions; `vendor.py --check` reports the
vendored state on its own, offline.

`--split-platforms` is the default and is not optional in practice: a combined zip would
carry three copies of `jaxlib` and exceed the extensions platform's size limit. Current
output, all comfortably under the ~200 MB ceiling:

| Platform | Zip |
|---|---|
| `linux-x64` | 138.7 MB |
| `windows-x64` | 118.8 MB |
| `macos-arm64` | 102.0 MB |

Verified by installing the built zip into a clean Blender profile
(`BLENDER_USER_RESOURCES` pointed at an empty directory): all ten dependencies resolve
from the bundled wheels, a UR5e imports, PyRoki solves at 0.0001 mm in ~5 ms, and the
baked .blend still animates after the extension is removed entirely.

[uv]: https://docs.astral.sh/uv/
[debugpy]: https://github.com/microsoft/debugpy

## How dependencies ship

Blender extensions may not install packages at runtime, so everything is bundled:

| Dependency | How | Why |
|---|---|---|
| jax, jaxlib, scipy, lxml, … | bundled wheels (cp313) | normal PyPI packages |
| **pyroki** | vendored source | not on PyPI; `version = "0.0.0"`, git-install only |
| **jaxls** | vendored source | not on PyPI — the PyPI name `jaxls` is an unrelated "JAX Language Server" package |
| numpy | *not bundled* | Blender ships NumPy 2.3.4; a second copy risks an ABI clash |

Both vendored projects are MIT licensed; see `LICENSES/`. `tools/vendor.py` keeps the
vendored trees byte-identical to upstream except for dropping PyRoki's `viewer/`
subpackage, which depends on `viser` (a web 3D visualiser — pointless inside Blender).

Parsing runs on a worker thread and rig building is spread across modal timer ticks, so an
import never blocks Blender's event loop and can be cancelled with Esc.

### The robot catalogue

Kinema downloads nothing. It ships a catalogue of the 186 robots in
[`robot_descriptions`](https://github.com/robot-descriptions/robot_descriptions.py) as
static JSON, and *Browse Robot Catalog* turns a search into a `git clone` command on your
clipboard — with the pinned revision and, crucially, the path of the description file
*inside* that repository. `mujoco_menagerie` alone holds 2466 files and backs 49 of the
robots, so "clone this and find it yourself" would not be much of an answer.

`tools/build_catalog.py` generates `src/kinema/catalog/robots.json`. Resolving those inner
paths is the interesting part: a description module builds them with `os.path.join` over
whatever its cloning function returned, so importing it with that function stubbed to a
sentinel string yields every path relative to the repository root, offline. One of the 186
(`eve_r3_description`) rewrites its URDF at import and cannot be probed this way.

`src/kinema/catalog/curation.json` sits alongside it, hand-maintained and *subtractive*: an
entry named there is marked a duplicate, broken or partial and hidden from the picker until
**Show all variants** is ticked. Everything not named there shows normally.

## Known limitations

- **Ball joints are rejected.** A 3-DoF spherical joint has no honest single-axis bone
  equivalent. One catalog robot (Cassie) is affected; everything else parses.
- **PyRoki reads URDF only.** MJCF rigs are bridged by rendering the parsed kinematic
  tree back out as a minimal URDF, which works but is one more moving part.
- **Windows long paths.** The bundled wheels unpack to deeply nested directories —
  `jax/_src/internal_test_util/export_back_compat_test_data/…`. On a Windows profile
  without long-path support the install fails partway with `WinError 206`. Enable
  `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled`, or keep
  Blender's config directory shallow.
- **The first IK solve compiles.** JAX JITs the solver on first use, roughly 14 s. It is
  paid up front when the IK target is created, behind a wait cursor, and never again for
  that rig; warm solves are ~5–20 ms.
- **MuJoCo's OBJ meshes print MTL errors** on import. MJCF carries its own colours, so
  the missing .mtl files are harmless console noise from Blender's OBJ importer.
- **One import at a time.** The fetch hooks are process-global, so a second import while
  one is running is refused rather than queued.
- **Meshes assemble then snap.** They are parented and placed in a single pass after the
  last one loads, because doing it per-chunk would cost a depsgraph evaluation each time.
  On a big robot the parts visibly pile at the origin for a second before landing.
- **Changing the cache location mid-session** only affects robots not yet imported: a
  description module resolves `REPOSITORY_PATH` once, at first import.

## License

GPL-3.0-or-later (the add-on links `bpy`). Vendored PyRoki and jaxls remain MIT.
