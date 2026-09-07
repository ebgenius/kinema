# Changelog

All notable changes to Kinema are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Find Solutions**, in the IK panel: the other arm configurations that reach the same tool
  pose. A six-axis arm can reach one up to eight different ways — elbow bent one way or the
  other, wrist flipped, base swung round behind — and until now you got whichever one the
  solver landed on from wherever the arm happened to be.

  PyRoki is a nonlinear solver rather than an analytic one, so it does not enumerate
  branches; it converges to whichever is nearest its seed. They are therefore *found* rather
  than derived: seed from configurations scattered across the joint limits, solve each, keep
  the ones that reached the goal, fold away the duplicates. That makes the result a lower
  bound, and the panel says so rather than claiming to have enumerated anything.

  The ◀ ▶ arrows cycle through what turned up, and applying one only writes joint values —
  so **Key All** keyframes it and it bakes like anything else. Solutions belong to one goal
  pose; move the target and the panel reports them stale instead of offering configurations
  for a pose the robot is no longer being asked to reach.

  On a KR120 at a working pose this finds the arm you are in plus the wrist-flipped one,
  with the tool holding to 0.008 mm across the switch.

- **An elbow target for redundant arms**, the way a pole target steers a character's arm in
  Blender. Seven axes reach a pose an infinite number of ways, the elbow sweeping through a
  continuous family while the tool stands still — so **Add Elbow Target** gives you a bone
  to drag, and the arm reconfigures around a tool that does not move.

  It is a soft position goal on the elbow's link, weighted far below the tool's, so the
  elbow may only move where the arm has freedom left over. An attractor rather than an
  angle reference: the elbow is pulled *toward* it rather than aimed *through* it.

  Offered only when the chain has more joints than the task needs — on a six-axis arm there
  is nothing left to steer, so the control is absent rather than present and inert — and it
  needs the PyRoki solver, which the panel says rather than ignoring the control quietly.

## [0.4.0] - 2026-09-07

The KUKA descriptions that would not import correctly, and then the thing that became
worth building once they did.

A KR120 imported with its bones up to 5° away from the geometry they were supposed to
carry — 0.22 m of error at the flange — because its URDF forbids its own zero pose, which
is true of every KUKA quantec. Chasing that turned up two more: link meshes landing with
their origins metres underground, and collision geometry that was parsed and then thrown
away. With those fixed, the robots stood where their descriptions said, and the question
stopped being *how do I get this robot into Blender* and became *what do I want it to do*.

**Waypoints** are the answer to the second question, and the headline here. Everything
before them in this list is what had to be true first.

### Added

- **Collision geometry**, behind a new **Import Collision Meshes** option, off by default.
  A URDF describes a robot twice — what it looks like, and the coarse hulls a planner
  sweeps — and Kinema only ever loaded the first. On a KR120 that is seven STL hulls
  alongside the visual geometry.

  Both kinds now go into their own collection under the robot's, `<robot> Visual` and
  `<robot> Collision`, so either can be hidden as a group: see the robot without the hulls
  over it, or hide the robot and check the hulls alone. Collision starts hidden — with the
  collection's eye rather than its monitor toggle, which would drop the objects from the
  depsgraph and leave their placement stale — and draws as wire, so a hull switched on over
  the robot reads as the envelope it is.

  For **MJCF**, geoms in groups 3 to 5 now become collision geometry instead of being
  dropped. MuJoCo has no visual/collision distinction in the format, so the group is the
  whole signal; anything else stays visual, which is the only safe default for a model that
  groups nothing. Silently losing a model's hulls was the worse trade.

- **Meshes at File Origin**, a checkbox beside Reset Meshes. Improving a robot's geometry
  means editing a mesh and writing it back as a drop-in replacement for the file the URDF
  points at — and exporters write world space, so the mesh has to actually *be* at that
  file's coordinates. On the rig it never is: the link frame, the visual origin, the mesh
  scale and the file's own units and up-axis all stand between the two.

  Ticking the box sends every link mesh to the coordinates its own file uses; unticking puts
  them back exactly. On a KR120 that means `link_4` lands at its `.dae`'s own bounds, in the
  frame that file was authored in, ready to edit and export. The meshes stay parented, so
  posing the rig while it is on drags them — the panel says so, and unticking is the way
  back.

  Rigs imported before this release recorded nothing about what was baked into their
  vertices, so the file's coordinates are genuinely unrecoverable for them; the operator says
  so rather than moving them somewhere wrong.

- **A warning when a URDF's zero pose is outside a joint's limits.** The rig still rests at
  the nearest allowed value, which is the honest answer — the robot genuinely cannot stand
  where the description says — but arriving there silently is confusing, and Rest Pose
  cannot undo it. The message names the joint, its range, and where the rig will rest, and
  points at the "Enforce Joint Limits" option for seeing the URDF's own zero pose.

- **Waypoints**, and motion generated from them — the answer to what a rigged robot should
  actually *do*. Until now the only motion authoring was keyframing the IK target by hand,
  which is the same workflow as animating any Empty, so nothing about it was robot-shaped.

  Pose the robot and press Record. Each waypoint stores the tool pose *and* the joint vector
  it was taught in, because a pose alone does not say which of up to eight solutions you
  meant; Go To restores that exact configuration instead of re-solving. Every waypoint also
  gets an Empty in the viewport, so one can be snapped to a feature on the part being worked,
  and dragging it re-aims the move.

  **Time is the ordering.** A waypoint carries a frame, not a list position, so there is no
  second order to keep in sync — retime a move in its row and regenerate, and no two
  waypoints may share a frame. Each row says how the robot arrives there: **Joint** to
  interpolate the joints, **Linear** to drive the tool along the straight line between two
  poses. On a KR120 a linear move holds that line to 0.002 mm over a 300 mm plunge.

  Generate Motion writes ordinary keyframes: joint channels for joint moves, the IK goal for
  linear ones, and the live-IK switch keyed so each span is solved the way it should be.
  Nothing about the result is special, which is the point — Blender's own graph editor
  shapes it afterwards, and Bake IK to Keyframes still turns it into plain joint curves that
  render with the add-on gone. Regenerating clears only the frames the job owns, so
  animation outside it survives.

### Changed

- **Link mesh origins now sit on their link frames.** A URDF is free to author every mesh in
  one shared frame and correct with a large `<visual><origin>`, and the KUKA quantec
  descriptions do exactly that — which put `link_3` through `link_6` at object origins of
  (1.15, 0, −1.15), metres underground and nowhere near their geometry. Blender draws a
  parent relationship line from an object's origin, so the viewport filled with dashed lines
  converging on a point under the floor, and selecting a link put its origin gizmo somewhere
  unrelated to the part.

  The visual origin now goes into the vertices along with the mesh scale and the file's own
  unit and up-axis correction, leaving the link frame alone as the object's transform. For an
  actuated link that is the bone's own head, so a relationship line is one bone long. Nothing
  renders differently: it is the same product, split between the geometry and the transform
  in a different place.

- **`rospkg` moves to 1.6.2** in the bundled wheel payload.

### Fixed

- **Link meshes drifting away from their bones**, on any robot whose URDF zero pose its own
  joint limits forbid. Every KUKA quantec is one — `kr120_r2700_2`, `kr150_r3100_2`, the
  `kr210` family, `kr240`, `kr300`, `kr340`, `kr500` all limit `joint_2` to a range ending
  below zero, because the real A2 axis cannot reach 0°. The rig imported with its bones
  rotated up to 5° away from the geometry they were supposed to carry: 0.22 m of error at
  a KR120's flange, growing along the arm, with the solver targeting one pose and the
  screen showing another.

  Meshes were placed by assigning `matrix_world`, which Blender resolves against the parent
  bone's *evaluated* pose — and the limit constraints, added a step earlier, had already
  pulled the bones off the rest pose. The placement cancelled the clamp instead of following
  it, and the cancellation was baked in permanently. Placement is now computed from each
  bone's rest matrix, so where a mesh sits relative to its bone no longer depends on whether
  a constraint happened to be evaluated first.

## [0.3.2] - 2026-09-04

Real robot descriptions, from the repositories they actually ship in.

Two of the most widely used ROS description repositories would not import: a KUKA LBR iiwa
because its xacro reaches into a sibling package for its materials, and a Universal Robots
arm because its xacro requires an argument. Neither is unusual — that is simply how vendor
descriptions are laid out — and behind each was a small stack of further defects that only
became visible once the one in front was fixed.

Nothing here changes an existing rig. If your robots already import, this release is about
the ones that did not.

### Added

- **Xacro arguments.** A field in the import options — file picker and sidebar both — taking
  the syntax the `xacro` command line uses: `name:=ur5e ur_type:=ur5e`. Some descriptions
  cannot render without one. The Universal Robots `ur.urdf.xacro` opens by using `$(arg name)`
  on the line *above* the one that declares it, so its default never applies and the file
  simply will not load unless you supply a name.

  Pick a `.xacro` and the file browser now lists the arguments it declares, with their
  defaults, so you can see what it wants before importing rather than after failing. Get one
  wrong and the error names the argument xacro asked for and lists the rest, instead of
  `Undefined substitution argument name`.

  The arguments are stored on the rig, because the solver reloads the description later to
  build its model — a rig that imported cleanly and then quietly lost PyRoki would be a poor
  trade.
- **Package Search Paths** in the preferences: extra directories to find ROS packages in,
  searched in addition to the repository holding the file. For a cell whose macros live in a
  different repository from its robots.

### Fixed

- **Xacro robots that reach into a sibling package now import.** A ROS description
  repository puts packages side by side, and a robot in one routinely includes files from
  another — a KUKA LBR iiwa pulls its materials from `kuka_resources` next door. Kinema never
  told the xacro renderer where those packages were, so any cross-package `$(find …)` failed
  with `PackageNotFoundError` even though the package was sitting in the same clone. This
  blocked most vendor description repos, not one robot.
- **Mesh paths in rendered xacros resolved to the wrong place on Windows.** Rendering a xacro
  rewrites every `package://` reference into a `file://` URI, and on Windows the drive letter
  ends up in the URI's authority rather than its path — so Kinema, which read only the path,
  either got nothing at all or a path with the drive silently removed. It fell back to the
  description's own directory, which exists, so the robot imported without a single mesh and
  without an error.
- **Xacro rigs no longer lose the PyRoki solver.** The solver reloads the description to
  build its model, and handed the xacro to the URDF parser unrendered — so `$(arg …)` reached
  a float parser and the rig dropped to the NumPy fallback with
  `could not convert string to float: '$(arg'`. Silently, because falling back is what
  happens whenever the reload fails, and 38 of the catalogue's robots ship only a xacro.
- **A package is now identified by the name it declares**, not by the name of the folder it
  sits in. `Universal_Robots_ROS2_Description` declares itself `ur_description`, so every
  `$(find ur_description)` inside it failed to resolve. Both names work, since descriptions
  in the wild reference either.
- **A package whose `package.xml` is not plain ASCII no longer breaks the import.** The file
  is parsed as XML rather than read with the system codec, so a maintainer's name with an
  accent in it stops mattering — on Windows this crashed with
  `'charmap' codec can't decode byte 0x9d`.

### Changed

- **Package discovery is bounded to the checkout.** Kinema used to look for ROS packages by
  scanning several directories above the description, which for a robot saved anywhere near
  a drive root meant scanning the entire drive — no error, just an import that never
  finished. It now searches the containing package's parent, stops at a repository boundary,
  and never walks to a filesystem root.

## [0.3.1] - 2026-09-03

Clears the warning triangle Blender puts on Kinema in *Preferences → Add-ons*. Nothing about
using the add-on changes.

### Fixed

- **28 extension policy violations.** Blender flags any module loaded from inside an
  extension's folder that is not named under the extension's own package, and any
  `sys.path` entry pointing inside it. Kinema tripped both: the vendored PyRoki and jaxls
  source was put on `sys.path` and imported under its own top-level names, giving one
  warning per vendored module plus one for the path. Present since 0.2.0.

  The two packages now import as part of the add-on, so nothing goes on `sys.path` and every
  module is namespaced. This is what the extensions.blender.org review was asking for, and
  it confirms vendored source is allowed — it just has to live under the add-on's name.

### Removed

- **`jaxls/_py310`** from the vendored payload: 13 files selected by `sys.version_info` for
  Python 3.10 and 3.11, which Blender 5.2's Python 3.13 can never reach.

## [0.3.0] - 2026-09-03

Kinema no longer downloads anything, starts no threads, and writes nothing to your
environment. The robot catalogue stays, and gets better at the part it was always best at —
telling you which of 186 robots exists, where it lives and which file to open — but fetching
one is now your `git clone`, not a hidden network call from an add-on.

All of this is what the Blender extension guidelines ask for, and every piece of it makes
the add-on more honest about what it does when you press a button. The one visible cost is
that importing a robot now blocks Blender while it works.

### Changed

- **Importing a robot blocks.** It used to parse on a background thread and build the rig
  across timer ticks, so the viewport stayed live and Esc could cancel. Extensions may not
  start threads, and that machinery bought responsiveness and nothing else — so an import is
  now a wait cursor and a pause: a second or two for an arm, longer for a humanoid, where
  the time goes on one Blender mesh-importer call per visual.
- **The catalogue is offline.** *Import from Catalog* is now **Browse Robot Catalog**: search
  186 robots by name, maker or category, pick one, and Kinema puts a `git clone` command on
  your clipboard along with the pinned revision and the path of the description file inside
  that repository. Load the file with *Import URDF File…* as usual. The catalogue data ships
  in the extension as JSON, generated by `tools/build_catalog.py`.
- **Curated entries.** The catalogue can now mark a robot as a duplicate, broken or only
  partly supported, with a note and a pointer to the entry to use instead. Marked entries are
  hidden until you tick **Show all variants**, so searching "UR5e" stops returning three of
  them.

### Fixed

- **Importing a xacro failed on Windows**, always, with `WinError 32`. The renderer handed
  the parser a temporary file it was still holding open, which Windows will not allow a
  second handle on. 38 of the catalogue's robots ship only a xacro, so this closed off all of
  them. Now covered by tests.
- **Kinema silenced every other add-on's logging.** Muting the solver's per-solve output
  called loguru's `logger.remove()`, which drops every sink in the process — including those
  belonging to other add-ons, since loguru's logger is a process-wide singleton. It now
  disables just its own vendored packages.

### Removed

- The **network permission**. The extension no longer declares or uses one.
- **Every thread.** The solver's background preload and the import worker are both gone, and
  with them the **Preload Solver in Background** preference. The solver stack is imported on
  first use instead, which costs 2–5 s once per session at the first solve.
- **Every environment variable Kinema set**: `JAX_PLATFORMS`, `JAX_ENABLE_X64` and
  `ROBOT_DESCRIPTIONS_CACHE`. 64-bit mode moved to `jax.config.update` after the import; the
  CPU pin turned out to have nothing to pin, since only the CPU `jaxlib` wheels are bundled;
  and the cache variable went with the **Robot Cache** preference, which is also gone.
- Ten wheels from the payload. `robot-descriptions`, `GitPython`, `gitdb` and `smmap` went
  with the downloading catalogue. `packaging`, `setuptools`, `docutils` and `pyparsing`
  satisfied nothing at all — transitive requirements of a package that was never bundled,
  and three of them Blender already provides. `tqdm` and `typeguard` turned out to be the
  same story: nothing in Kinema or the vendored solver imports either, and nothing requires
  them. The payload is **38 wheels, down from 48** — 28 per platform, since five are
  compiled once each for Windows, Linux and macOS.

Rigs built by 0.2.0 or earlier from the downloading catalogue will report that their
description can no longer be reloaded; re-import it from disk to restore the PyRoki solver.

## [0.2.0] - 2026-09-01

Dressing and aiming. A robot can now carry the things that make it a working cell — a
gripper, a tool, a cable harness — and the solver can be pointed at any bone in the chain,
keyframe included. The tool frame became something you place and offset from the panel
rather than something you fight in Edit mode.

Requires Blender 5.2 LTS or newer; every dependency still ships inside the extension zip.

### Added

- A **Bones** list in the sidebar, one row per bone, that does the two things you reach for
  once a rig is imported: aim the solver at a bone, and hang something off it.
- **Attachments.** Pick an object or a collection in a bone's row and a linked copy rides
  that bone — a gripper on the flange, a cable harness down the forearm. The copy shares
  its mesh and materials with the source, so the same harness can dress six links and still
  be edited in one place, and the source may live in any scene in the file. It arrives as a
  placement and nothing more: the source's animation and constraints are stripped, so
  nothing reaches in from outside to move it.
- Offsets are measured from the bone's **head**, not the tail Blender's own bone parenting
  uses, so the attachment's plain location/rotation/scale *is* its offset from the joint —
  editable in the sidebar or with G/R/S, and keyframable. The panel edits whichever rotation
  channel the attachment's rotation mode actually uses.
- Clearing a row's picker removes the copy; the **✕** button unparents it and leaves it in
  the scene exactly where it appears.
- **A keyframable IK target bone.** The solver's tip is now a property on the rig rather
  than the position of the TCP marker, so it can be changed from the panel without entering
  Pose mode — and keyed, so a shot can hand the goal from the wrist to the elbow part-way
  through. Aiming at a bone further up the chain is how you cut a gripper's joints out of a
  redundant solve; that previously needed a script.
- **Key Target Bone**, which keys the target with stepped interpolation. The target is an
  index, not a quantity: interpolated, a hand-off from the tool point to joint 3 would pass
  through joints 1 and 2 on the frames between and solve two chains nobody asked for.
- Baking follows a keyed target across the range, keying every joint active anywhere in it.

- **A tool offset you can type.** The Tool Centre Point panel gains a parent-bone field and
  six fields — location and roll/pitch/yaw — that place the TCP relative to the flange's own
  link frame. The angles use URDF's convention (`Rz(y) @ Ry(p) @ Rx(r)`), so an offset copied
  out of a description can be typed in unchanged. Mounting a tool 150 mm off the flange no
  longer means adding a bone by hand and positioning it in Edit mode.
- The panel reports the tool's orientation beside its position, and the offset the importer
  recorded — which is rarely zero, because the tool frame usually sits behind one or more
  fixed joints from the last actuated one, and fixed joints get no bone to show it.

- **Reset Meshes**, beside *Rest Pose*, puts every link mesh back where the importer placed
  it. Link meshes are also locked now, so a stray grab is harder to have in the first place.

### Fixed

- **A mesh file's own units and up-axis were discarded** ([#12]). The COLLADA reader works
  them out — that is most of why it exists — and the rig builder then overwrote the result,
  so a millimetre `.dae` imported a thousand times too large and a Y-up one lay on its side.
  A test proved the correction was *computed*; none proved it survived the rig build.
- **A URDF `<mesh scale>` was parked in the object's scale instead of applied** ([#12]). The
  link looked right, but modifiers, physics and exporters that do not bake transforms all
  quietly used the unscaled mesh. Both the scale and the file's own correction now go into
  the geometry, so link meshes arrive at a scale of 1.
- **`<mesh scale="0.001"/>` aborted the whole import** ([#12]). The single-value shorthand
  comes back from the URDF parser as a bare number, which failed deep enough in the builder
  to take the entire robot with it rather than the one visual.
- **A link mesh moved by accident could not be put back** ([#14]). *Rest Pose* only returns
  the joints, and a mesh keeps its whole placement in its own transform channels, so a stray
  G/R/S destroyed the only copy of it. The placement is now recorded at import.
- A mirrored `<mesh scale>` — used by symmetric robots to reuse one file for a left and a
  right part — left the normals inside out.
- *Move TCP to Active Bone* could refuse **after** making the rig active and dropping it into
  Object mode, leaving you somewhere other than where you started ([#19]).
- A rig whose TCP bone had been deleted still offered *Update TCP* while reporting that it
  had no TCP ([#19]).
- **Clicking *Move TCP to Active Bone* raised `AttributeError: 'Bone' object has no
  attribute 'select'`** ([#16]). `bpy.types.Bone` carries no selection flag in Blender 5.x,
  and the operator's fallback read one whenever there was no active pose bone — which is
  always, in Object mode, so a freshly imported rig failed every time. It now takes the bone
  from the panel, the active pose bone, or the active edit bone, and says so plainly when it
  has none.
- **Re-placing the TCP flipped its orientation.** The importer built it from the URDF link
  frame, but *Move TCP to Active Bone* built it from the bone's own — and Kinema aligns each
  joint bone's Y to the joint axis, so the two disagree. Putting the TCP back on the last
  link left its Z pointing the wrong way, fixable only by hand in Edit mode. Both routes now
  build from the link frame.
- The TCP marker's two transverse axes were the same length and its approach axis had no
  arrowhead, so the widget showed where the tool was and roughly which line it lay on, but
  not which way it pointed or which way up it was. The approach axis is now arrowed, and the
  other two differ in length from each other.
- *Move TCP to Active Bone* wrote the **bone** name into `kinema_tcp_link` while the importer
  wrote the **URDF link**, so the panel's `Link:` label showed a bone as soon as the button
  had been used once.
- Placing the TCP on the IK control — the likeliest thing selected in Pose mode once IK
  exists — left the marker riding the goal it defines, which silently dropped the rig to the
  NumPy solver. Only joint bones are accepted now.
- *Move TCP to Active Bone* re-roots the IK chain, but the bone keeps its name, so the
  cached solver was not rebuilt and went on driving the old chain. It now invalidates.

[#12]: https://github.com/ebgenius/kinema/issues/12
[#14]: https://github.com/ebgenius/kinema/issues/14
[#16]: https://github.com/ebgenius/kinema/issues/16
[#19]: https://github.com/ebgenius/kinema/issues/19
- Baking solved every frame twice — once through the frame-change handler that `frame_set`
  fires, once in the bake loop — and on a rig whose target changed mid-range the two could
  disagree.
- F-curve lookups walked every slot of an Action. Two rigs sharing one could have the wrong
  rig's channels read, re-keyed, or cleared.

## [0.1.0] - 2026-08-29

First release. Kinema turns a robot description into a single animation-ready Blender
armature: one bone per joint, real limits, IK solved by [PyRoki], and a bake step so the
finished .blend renders without the add-on installed. Requires Blender 5.2 LTS or newer;
every dependency ships inside the extension zip.

Known limitations are listed in the [README](README.md#known-limitations) — ball joints
are rejected, the first IK solve compiles for ~14 s, and Windows needs long paths enabled.

### Added

#### Import

- Import from a catalog of 186 robots, filterable by category — arms, dual arms, end
  effectors, humanoids, quadrupeds, bipeds, mobile manipulators, wheeled robots, drones.
  Descriptions download on first use over HTTPS, with no `git` binary required.
- Import a local URDF, xacro or MJCF file, with `package://` references resolved by
  searching the file's own directory tree.
- MJCF descriptions read into the same internal model as URDF, accounting for MJCF's
  degree-valued angles, multiple joints per body, and per-joint pivots.
- A COLLADA (`.dae`) reader, also exposed as *File > Import > COLLADA*, restoring the
  format Blender 5.0 removed. It honours `<unit meter>` and `<up_axis>`, which is what
  most ROS and Gazebo descriptions need to arrive at the right scale and the right way up.
- Imports do not block Blender. Downloading and parsing run on a worker thread, rig
  building is spread across modal timer ticks, progress is reported, and Esc cancels.
  Blender's offline mode is honoured.

#### The rig

- One armature per robot, with one 1-DoF bone per actuated joint and the bone's local Y
  aligned to the joint axis, so a single channel *is* the joint value and every other
  channel is locked.
- Joint limits from the description, applied as constraints and optional, so a shot can be
  posed past a robot's real travel. Continuous joints stay unbounded and keep their
  multi-turn spins.
- Fixed joints get no bone; their transforms fold into the chain, so meshes and tool frames
  still land exactly where the description puts them.
- Bones sorted into `Kinema/FK`, `Kinema/IK`, `Kinema/TCP` and `Kinema/Mechanism`
  collections, with custom shapes, so an animator sees controls and not machinery.
- Link meshes parented rigidly to bones rather than skinned, because robot links are rigid
  bodies.
- A tool centre point bone, created at import or moved onto any bone afterwards.
- Joint names, types, limits, axes and link corrections stored on the bones, so a saved
  .blend keeps working on a machine that has never seen the original description.

#### Inverse kinematics

- A keyframable IK target bone that behaves like an ordinary Blender IK control but is
  solved by [PyRoki], a nonlinear least-squares solver that respects joint limits and
  steers away from singularities.
- A pure-NumPy damped-least-squares fallback that needs nothing but the rig itself, used
  when JAX is unavailable or still loading. The backend is switchable per rig, including
  off.
- Live solving as the target moves, with a solve budget that skips updates rather than
  dragging the viewport down on very high-DoF rigs.
- *Bake to Keyframes*, which solves a frame range and writes plain FK keyframes. After
  baking, the .blend animates and renders with Kinema uninstalled.

#### Downloads and cache

- Per-robot fetching for large monorepos. `mujoco_menagerie` is 1.64 GB and backs 49 of
  the 186 catalog robots, so Kinema downloads only the subdirectory a robot needs: a
  Unitree Go2 costs 31 MB instead of 1.7 GB. An existing full checkout is reused, and a
  second robot from the same repository is added incrementally.
- The cache lives at `~/.cache/robot_descriptions`, shared with `robot_descriptions.py`,
  and honours `ROBOT_DESCRIPTIONS_CACHE` and the *Robot Cache* preference.

#### Packaging

- Three per-platform extension zips: `linux-x64` (138.7 MB), `windows-x64` (118.8 MB),
  `macos-arm64` (102.0 MB). Every dependency is bundled and nothing is installed at
  runtime.
- Preferences for the default solver, background preloading of the solver stack, the live
  solve budget, the cache location, debug logging, and a dependency status readout.

[PyRoki]: https://github.com/chungmin99/pyroki

[Unreleased]: https://github.com/ebgenius/kinema/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/ebgenius/kinema/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/ebgenius/kinema/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/ebgenius/kinema/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/ebgenius/kinema/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ebgenius/kinema/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ebgenius/kinema/releases/tag/v0.1.0


