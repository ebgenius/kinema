# URDF, xacro and MJCF

Robots do not ship as `.fbx`. They ship as a description file plus a folder of meshes, in
one of a handful of formats. Here is what each one is and what it carries.

## URDF

**Unified Robot Description Format.** An XML file describing a robot as a chain of
[links and joints](links-and-joints.md). It is the closest thing the field has to a
universal format, and most robots you meet will be URDF or convertible to it.

A URDF carries:

- The link structure — what connects to what
- Each joint's type, axis and limits
- Where each mesh sits relative to its link
- Mass and inertia (Kinema ignores these; they matter for simulation, not animation)
- Collision geometry, usually a crude stand-in for the visual mesh

It does **not** carry materials in any useful sense, animation, or a scene. It is a
description of a machine, not a shot.

### The `package://` problem

URDF files refer to meshes with paths like:

```xml
<mesh filename="package://ur_description/meshes/ur5e/forearm.dae"/>
```

That `package://` prefix assumes a ROS workspace on the machine that reads it. You do not
have one, and the path is meaningless outside it.

Kinema resolves these by searching the description file's own directory tree for the named
file. This handles the overwhelming majority of real robot packages with no configuration
— but it does mean you need the **whole package**, not just the `.urdf`. A lone
description file with no `meshes/` folder gives you a correct, invisible robot.

## xacro

**XML macros.** A URDF with programming in it — variables, arithmetic, conditionals, and
macros that expand into repeated structure.

Manufacturers use xacro because one file can describe a whole family: the same arm with
three different grippers, or a robot whose link lengths are parameters. A `.xacro` is not
a robot description until it has been expanded into one.

Kinema expands xacro on import, so you can point straight at the `.xacro` file. When one
fails, the usual cause is a required argument the file expects to be given — the error
message names it.

## MJCF

**MuJoCo XML.** The format used by the MuJoCo physics engine, and increasingly the
distribution format for robot-learning research — a lot of recent robots appear as MJCF
first and URDF never.

MJCF is richer than URDF about physics: contacts, actuators, tendons, sensors. Kinema
reads only the kinematic subset it needs — bodies, joints, limits, and visual geometry —
and ignores the rest.

Two practical notes:

- MJCF robots frequently ship OBJ meshes without their `.mtl` companion files, and
  Blender's OBJ importer prints errors about the missing files. These are harmless: MJCF
  specifies its own colours.
- The IK solver reads URDF internally, so an MJCF rig is bridged by writing the parsed
  structure back out as a minimal URDF. It works, and it is one more moving part.

## `.xml` is ambiguous

Both URDF and MJCF commonly use the `.xml` extension, so the extension tells you nothing.
Kinema reads the opening bytes of the file and dispatches to the right reader. You never
have to declare which you have.

## What Kinema does not read

- **SDF** (Gazebo's format) — not supported
- **USD** — Blender reads USD natively, but a USD robot has no joint limits or joint axes
  to build a rig from
- **Vendor formats** — KUKA, ABB, FANUC and the rest each have their own; you will need a
  URDF export

## Which should you want?

If you have a choice, take the **URDF**. It is the best-supported path, it is what the
solver speaks natively, and it avoids the MJCF bridge. Take MJCF when it is what exists —
which, for recent research robots, is increasingly the case.
