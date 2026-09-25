# Kinema

**Animation-ready robot rigs in Blender.** Import a robot description and get a single clean
armature you can actually animate — with IK that understands singularities, joint limits and
multi-turn joints. A built-in catalogue of 186 real robots says where to find one.

> Status: released as [v0.5.0]. Import (URDF, xacro or MJCF), rig, pose, solve and
> bake all function, and the built extension installs and runs from a clean Blender
> profile. Nothing is downloaded, no threads are started, and no environment
> variables are written — an import blocks Blender while it works.
>
> Docs: <https://ebgenius.github.io/kinema/>

[v0.5.0]: https://github.com/ebgenius/kinema/releases/tag/v0.5.0

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
Panda imports with its tool frame on a fingertip, which puts gripper finger joints inside the
chain. They are slides, so **Add IK Target** holds them the way it holds a rail (see *Spare
joints* below), but the arm is still solving to a finger rather than to its hand. Aim at the
flange instead and the chain is the seven arm joints it should be. Switching snaps the IK
control onto the new target, so the arm does not jump.

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

## Checking joint speeds

A robot can't follow motion faster than its motors, and Blender plays back a joint keyed
through half a turn in two frames as smoothly as one given two seconds. Most robot
descriptions give each joint a top speed in `<limit velocity>`. Kinema keeps it on the joint
and checks the frame on screen against it.

A joint over its limit is flagged in two places:

- **In the panel.** Its slider in **Joints (FK)** turns red, and **Velocity Limits** lists
  every joint's speed against its limit.
- **In the viewport.** Its dial or arrow is redrawn in red, with its name and its speed as a
  percentage of the limit.

**Ignore Velocity Limits**, in that section, turns both off for one rig: for a shot that will
never run on a robot, or a description whose limits are placeholders. The numbers stay
listed, greyed out.

| Joint | Measured against | After a jump or a scrub |
|---|---|---|
| Keyed, including joint moves from Generate Motion and baked IK | its own curve, one frame back | exact |
| Driven by live IK, including linear moves | the pose seen one frame earlier | `—` until you play or step a frame |
| Neither | the pose seen one frame earlier | still |

Speeds use the scene's frame rate. A gap of up to three frames, from playback dropping
frames, is averaged. An average can miss a spike inside the gap, but never reports one that
isn't there. A joint pressed against its range stop doesn't count as moving, whatever its
curve does.

**Load Joint Limits…** reads a MoveIt or ros2_control `joint_limits.yaml`. That is often
where a robot's real running speeds live. It works joint by joint:

- `has_velocity_limits: true` sets a limit;
- `false` turns it off;
- a joint the file doesn't mention keeps the description's value.

MJCF has no velocity limits, so an MJCF rig only gets them from such a file.

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
Held joints stay where they stand in every solution, so on a robot with its rail held, every
alternative is an arm reaching from where the carriage already is.

On a KR120 at a working pose, this turns up the arm you are in plus the wrist-flipped one
— joints 1 to 3 identical, 4 to 6 turned through 180° — with the tool holding to 0.008 mm
across the switch. The other families a KR120 nominally has are ruled out by its own joint
limits at that pose, which is the correct answer rather than a shortfall of the search.

### Spare joints: rails you move by hand, and the elbow swivel

A robot with more joints than a tool pose needs can reach that pose a continuous family of
ways. What the spare joint *is* decides which control makes sense, so there are two.

**Held joints.** Most redundant robots in a cell are a six-axis arm on a rail or a gantry.
The extra axes are not something to solve for: an operator positions them, and the arm
reaches from wherever they stand. So a joint can be **held by hand**. Click the pin beside
its slider in the **Joints (FK)** panel, then move it with that slider or by dragging its
bone along its axis, and IK solves the rest of the chain with the tool held where it is. The
pin is keyframable, so a rail can be handed back to the solver part-way through a shot.

A held joint is solved where the rig shows it. Drag a rail past its limit and the arm
reaches from where the limit stops the carriage, while the channel keeps the value you
dragged, rather than planning around a carriage that isn't there.

**Add IK Target** pins rails for you. While a chain still has more than six unheld joints,
its prismatic joints are held, base first. So a six-axis arm on a rail gets its rail held and
nothing else, and an arm with nothing to spare, or whose spare joint is rotary, holds
nothing. A turntable can't be told apart from a seventh arm joint, so pin that one yourself.

**The elbow swivel.** On a seven-axis arm the spare freedom *is* the elbow: with the tool
held still, it can swing round the line from shoulder to wrist. **Add Elbow Swivel** puts a
ring on that line, and turning the ring swings the elbow round it. It's the same idea as the
control that steers a character's elbow once IK has placed the hand.

- **The ring rides the arm.** A hidden bone follows the shoulder-to-wrist line through
  Blender's own constraints, so the ring stays with the arm with IK off, during playback and
  at render.
- **Its one rotation channel is the elbow's angle.** It keys as a single curve, and the IK
  panel shows it as a **Swivel** slider with its own keyframe dot.
- **It aims at a point the elbow can actually reach:** a point on the circle the elbow
  sweeps, not wherever a bone happened to be dragged. On the `arm7` fixture, swinging the
  elbow through ±1.5 rad keeps the tool within 0.0003 mm and the elbow within 0.02° of the
  angle asked for, at every strength from 0.5 to 20. The free-floating elbow target this
  replaced pulled toward points off that circle and gave up tool accuracy instead: 0.55 mm at
  its default strength, 30.7 mm at 20.
- **It keeps up with the target.** The circle is built about where the wrist is going, not
  where it is, so the single live update a drag produces already puts the tool on its goal.
  Dragged past the arm's reach, the elbow stays on the side the ring put it instead of
  flipping over.
- **Adding one changes nothing.** The ring is turned onto the elbow before anything solves.
- **It doesn't vanish when a hold takes its freedom.** Pin a joint so that six or fewer are
  left to IK and the section stays, with the slider greyed, a line saying why, and
  **Remove** still there. Release the joint and the swivel works again.

Shoulder, elbow and wrist default to the second, middle and second-to-last joints. That is
exact for an arm with a spherical shoulder and wrist, like the `arm7` fixture or an iiwa, and
only approximate, and not yet measured, on an arm whose links are offset, like the Panda.
**Add Elbow Swivel** is offered only when more than six joints are left to IK, so a rail robot
with its rail held has no elbow to spare. The swivel also needs the PyRoki solver: the NumPy
fallback has no notion of a second goal, and the panel says so.

## External axes the description doesn't have

A robot often arrives as the arm alone, while the cell it works in has more: a track it
rides, a turntable under it, a positioner holding the part, a spindle on its flange.
**External Axes → Add External Axis…** adds one to the rig. The dialog asks for:

- **Preset:** Linear Track, Rotary Base, Positioner, Tool Spindle or Tool Slide. Each one
  just fills in the fields below, and any of them can still be changed.
- **Mount:**
  - *Under the Robot*: the robot rides it.
  - *On the Tool*: it rides the flange and carries the TCP.
  - *Standalone*: it stands beside the robot and carries nothing until you attach a part to
    it in the **Bones** panel.
- **Motion and direction:** linear or rotary, along or about ±X, ±Y or ±Z of the axis's base
  placement. A rotary axis can be continuous.
- **Limits and top speed.** The speed feeds the **Velocity Limits** check like any other
  joint's.
- **External axis base:** where the axis sits, calculated from the current robot base, or
  from the current tool frame for an axis on the tool. Location, plus roll, pitch and yaw
  about fixed X, Y and Z, as in URDF.
- **External axis offset:** where what the axis carries sits on its moving part. That is the
  current robot base for an axis under the robot, and the TCP for one on the tool.

**The current robot base** includes any axes already under the robot. A new axis under the
robot goes in at the bottom of that stack, and its base is measured from just under the
lowest one's rail or turntable. Add an X track, then a Y track, and the Y track goes under
the X track's rail and carries it, as if the X track were already part of the robot.

The axis is an ordinary joint bone, built by the same code as an imported joint. It gets a
slider in **Joints (FK)**, keys, bake and a velocity check, and it starts **held**, since it
is positioned by hand and the arm reaches from wherever it stands. Release the pin to let IK
move it too.

**See it before you add it.** While the dialog is open, the viewport shows the result and
follows every field you change:

- the rail and carriage, or base and plate, when **Placeholder** is ticked;
- the travel: the carriage outlined at both end stops, or an arc from stop to stop, with the
  limits written beside them;
- for an axis under the robot that moves it, a ghost of the robot and the axes it already
  rides, where they will stand.

The ghost is a coarse copy of the meshes, moved as one block. Nothing is solved, and nothing
is added to the scene until you press OK. Cancel leaves no trace. Clicking in the viewport
to look closer closes the dialog, but opening it again brings back everything you typed.

**No compile while you adjust it.** After OK, the **Adjust Last Operation** panel re-runs
the add on every change. A PyRoki compile is tens of seconds, so none happens there: the rig
solves on NumPy, and the panel says so, until you move on to something else or leave the
axis alone for ten seconds. Then PyRoki is compiled once, behind a wait cursor.

**Moving the robot, and putting it back.** With base and offset at zero nothing moves: the
axis slides in under the robot and its axes, or behind the TCP, exactly where they are. Give
them values and the robot ends up at base-then-offset on the carriage. Everything riding the
robot moves with it by one rigid transform, so the pose you had is still the pose you have:

- the robot's bones and its TCP;
- the axes already under it, and their placeholders;
- the IK target and the swivel;
- the base link's meshes;
- the waypoints.

**Remove** undoes all of it, stacked axes in any order included. Something you parented to
the axis bone by hand stays where it stands. An axis on the tool moves the TCP to its offset,
and removing the axis puts the TCP back with the offset it had.

**Joint indices follow names.** Blender orders bones by hierarchy, so an axis under the robot
becomes joint 0. Waypoints' joint vectors and the IK tip, keyframes included, are remapped
by name when an axis comes or goes.

**Placeholders are generated, not shipped.** A rail with a carriage, or a base with a
turning plate and a pointer tab, built from a few dozen vertices at the size the dialog
suggests from the robot's reach. They ride their bones, use one material you can restyle,
and leave with their axis. Turn **Placeholder** off if you have a model of the real thing;
attach it to the axis bone in the **Bones** panel.

**The solver sees the axis.** The rig is no longer the robot in its description, so PyRoki
is handed a model written from the rig's own bones instead of the file. Each joint bone
already records its type, limits, link and link frame, so the model is exact. It matches
the description to 1e-5 on the fixtures, and arm6 with a track added is joint-for-joint the
hand-written `rail6.urdf`. Mimic joints get no bone, so they are left out. They never move
the tool.

## Where the tool frame sits

The **Tool Centre Point** panel places the TCP on a joint bone and offsets it from there.
**Edit TCP…** opens a dialog for it that previews before anything moves:

- **Offset:** type a location and rotation, as the panel's fields take them.
- **3D Cursor:** the TCP goes where the cursor is. Snap the cursor to a vertex of the tool
  mesh first (Shift+S), and the TCP lands on that vertex.
- **Object Origin:** the TCP goes on an empty, or on the origin of the tool mesh itself.

The cursor and the object can give the orientation too, or leave it at the typed one. Either
way the dialog shows the offset it will store, measured from the link where it stands now,
so a TCP can be picked off a tool on a posed robot. While it is open, the viewport draws the
new TCP next to the current one. Clicking away closes the dialog, and opening it again brings
back what you had. The panel's fields and *Update TCP* still work as before.

**Changing the TCP doesn't move the robot.** Redefining the tool, as on a real controller,
leaves the joints where they are, and the IK target comes to the new TCP. A keyed IK target
goes back to its keys on the next frame, because an animated path is the tool's path, and
the new tool follows it.

**Changing the TCP doesn't recompile the solver.** PyRoki's compiled solver depends on the
robot and on the joint the TCP rides, not on the tool offset. Re-offsetting the TCP, or
removing and re-adding the IK target, reuses it. A TCP moved onto another joint needs a new
one, and it waits until you're done: live IK solves on NumPy meanwhile, and the panel says
so. It then compiles once, when you move on to something else or leave it for ten seconds.
A bake, Find Solutions, playback and rendering never wait. They use PyRoki, compiling it if
they must.

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
uv run python tools/fetch_wheels.py        # download the wheel payload (~340 MB, 3 platforms)
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
uv run python tools/fetch_wheels.py        # all three platforms (~340 MB)
uv run python tools/dev.py validate
uv run python tools/dev.py build           # per-platform zips into dist/
```

`vendor/` and `wheels/` are both gitignored, so a fresh clone has neither and the first
two commands are not optional. They are one-time setup rather than per-build steps —
re-run `vendor.py` only when its pins move, and `fetch_wheels.py` only when `uv.lock` does.
`dev.py build` refuses to run if either directory is missing, stale against its pin, or
holds one package at two versions; `vendor.py --check` reports the vendored state on its
own, offline.

The wheels are downloaded at exactly the versions `uv.lock` pins, so the zip carries what
the dev venv and the tests ran against. To move a package, run
`uv lock --upgrade-package <name>` and fetch again. A unit test fails if the manifest and
the lock disagree.

`--split-platforms` is the default and is not optional in practice: a combined zip would
carry three copies of `jaxlib` and exceed the extensions platform's size limit. Current
output, all comfortably under the ~200 MB ceiling:

| Platform | Zip |
|---|---|
| `linux-x64` | 136.5 MB |
| `windows-x64` | 116.5 MB |
| `macos-arm64` | 99.8 MB |

For 0.5.0, the Windows zip was installed into a clean Blender 5.2.1 profile
(`BLENDER_USER_RESOURCES` pointed at an empty directory) and checked:

- every bundled dependency loads from the zip's own wheels;
- a UR5e imports, and PyRoki solves it from a working pose to 0.00 mm in ~2 ms warm;
- xacro, MJCF and COLLADA files import, and Load Joint Limits reads a yaml;
- Blender reports no extension policy warnings, and no thread is started;
- a baked .blend still animates in a Blender with no Kinema installed at all.

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

An import runs as one blocking call behind a wait cursor. Parsing used to run on a worker
thread, with the build spread across modal timer ticks so Esc could cancel it, but Blender's
extension guidelines do not allow an extension to start threads.

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
  paid up front when the IK target is created, behind a wait cursor. Adding an **Elbow
  Swivel** pays one more compile, because an elbow goal makes a different problem; pinning a
  joint pays none. Warm solves are ~5–20 ms.
- **Imports block.** A big robot freezes Blender until it has finished loading, with no
  progress and no cancel. Extensions may not start threads, which is what kept the viewport
  alive before.
- **MuJoCo's OBJ meshes print MTL errors** on import. MJCF carries its own colours, so
  the missing .mtl files are harmless console noise from Blender's OBJ importer.

## License

GPL-3.0-or-later (the add-on links `bpy`). Vendored PyRoki and jaxls remain MIT.
