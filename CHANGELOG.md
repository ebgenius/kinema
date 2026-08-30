# Changelog

All notable changes to Kinema are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- A **Bones** list in the sidebar, one row per bone, that does the two things you reach for
  once a rig is imported: aim the solver at a bone, and hang something off it.
- **Attachments.** Pick an object or a collection in a bone's row and a linked copy rides
  that bone — a gripper on the flange, a cable harness down the forearm. The copy shares
  its mesh and materials with the source, so the same harness can dress six links and still
  be edited in one place, and the source may live in any scene in the file. Offsets are
  measured from the bone's **head**, not the tail Blender's own bone parenting uses, so the
  attachment's plain location/rotation/scale *is* its offset from the joint — editable in
  the sidebar or with G/R/S, and keyframable. *Detach* unparents and leaves the object
  exactly where it appears on screen.
- **A keyframable IK target bone.** The solver's tip is now a property on the rig rather
  than the position of the TCP marker, so it can be changed from the panel without entering
  Pose mode — and keyed, so a shot can hand the goal from the wrist to the elbow part-way
  through. Aiming at a bone further up the chain is how you cut a gripper's joints out of a
  redundant solve; that previously needed a script.

### Fixed

- *Move TCP to Active Bone* re-roots the IK chain, but the bone keeps its name, so the
  cached solver was not rebuilt and went on driving the old chain. It now invalidates.

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

[Unreleased]: https://github.com/ebgenius/kinema/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ebgenius/kinema/releases/tag/v0.1.0
