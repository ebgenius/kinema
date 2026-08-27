# Robot catalog

**Kinema panel → Import from Catalog…** opens a picker listing 186 real robots — arms,
humanoids, quadrupeds, drones, grippers and mobile manipulators — ready to import and rig.

> 📷 *Screenshot: the catalog picker, filtered.*

## What each entry tells you

Robots are listed as **maker + model** (`Universal Robots UR5e`, `Boston Dynamics Spot`),
with a line underneath carrying:

- **Degrees of freedom** — how many joints move
- **Tags** — what kind of machine it is
- **Licence** — the SPDX identifier of the robot description's licence

Type to filter. The list is long; searching by maker or model is faster than scrolling.

## Tags

| Tag | What it is |
|---|---|
| `arm` | A single robot arm — the classic industrial manipulator |
| `dual_arm` | Two arms on one torso |
| `end_effector` | A gripper or hand on its own, no arm |
| `humanoid` | Two arms, two legs, a torso |
| `quadruped` | Four legs |
| `biped` | Two legs, no arms |
| `mobile_manipulator` | An arm on a moving base |
| `wheeled` | A wheeled base |
| `drone` | A flying platform |

## Licences

Robot descriptions carry their own licences, independent of Kinema's. The catalog shows
each one because it matters: some are permissive, some restrict commercial use, some
require attribution.

**If the robot appears in work you ship, check its licence.** Kinema being GPL says
nothing about the terms attached to a given manufacturer's CAD.

## Downloads and the cache

Kinema ships the *catalog* — names, makers, joint counts, tags — but not the robots. The
meshes alone would be gigabytes.

So:

- **Browsing the catalog is offline.** The metadata is bundled.
- **Importing a robot downloads it**, the first time only.
- **After that it loads from cache**, offline.

Downloads go to `~/.cache/robot_descriptions` by default. You can point somewhere else
with the **Robot Cache** [preference](preferences.md#robot-cache), or with the
`ROBOT_DESCRIPTIONS_CACHE` environment variable — Kinema honours the same variable as the
underlying library, so it shares a cache with any other tooling you already use.

!!! info "Cached by repository, not by robot"
    Several robots often live in one upstream repository, so the cache is organised by
    repository. Importing one arm from a family may leave its siblings already downloaded.

If a download fails, the usual causes are no network, a firewall blocking the host, or the
upstream repository having moved. The cache is safe to delete at any time — the next
import re-fetches.

## URDF and MJCF entries

Catalog robots come as [URDF or MJCF](../concepts/formats.md), and a few offer both.
Kinema reads either, so the format is not something you need to choose. Where both exist,
URDF is used — it is what the solver speaks natively, avoiding the MJCF bridge.

## When your robot is not listed

Use [Import URDF File…](../tutorials/import-your-own.md) with a local file. The catalog is
a convenience for well-known machines, not a limit on what Kinema can rig.

## Known exception

**Cassie** contains a ball joint and will not import. A 3-DoF spherical joint has no
honest single-axis bone equivalent. It is the only catalog robot affected.
