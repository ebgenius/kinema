# Changelog

All notable changes to Kinema are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/ebgenius/kinema/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/ebgenius/kinema/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ebgenius/kinema/releases/tag/v0.1.0
