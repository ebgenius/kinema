# Kinema

**Animation-ready robot rigs in Blender.** Import any of 186 real robots and get a single
clean armature you can actually animate — with IK that understands singularities, joint
limits and multi-turn joints.

> Status: released as [v0.1.0]. Import (URDF, xacro, MJCF, or the 186-robot
> catalog), rig, pose, solve and bake all function, imports run without blocking
> Blender, and the built extension installs and runs from a clean Blender profile.
>
> Docs: <https://ebgenius.github.io/kinema/>

[v0.1.0]: https://github.com/ebgenius/kinema/releases/tag/v0.1.0

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
be edited in one place, and the source can live in any scene in the file.

The offset is measured from the bone's **head**. Blender's own bone parenting measures from
the tail, which puts a fresh attachment at the far end of the bone and makes every offset
you then dial in relative to a frame that has nothing to do with the joint. Kinema cancels
that, so the attachment's plain location/rotation/scale *is* its offset from the joint —
edit it in the panel, or grab the object and use G/R/S, and keyframe it like anything else.
*Detach* unparents and leaves the object exactly where it appears.

**Choose what IK aims at.** The radio button in each row points the solver at that bone.
This matters on redundant robots: a Panda imports with its tool frame on a fingertip, which
leaves both gripper joints inside the chain — 9 DoF against a 6-DoF task — and the solver
then holds the fingertip still while spinning the hand around it. Aim at the flange instead
and the chain is the seven arm joints it should be.

The choice is keyframable, so a shot can hand the goal from the wrist to the elbow part-way
through; the *Target Bone* field in the IK panel is the channel to key. Switching to a bone
whose link the solver has not seen before pays a one-off compile, the same wait as the first
solve after adding an IK target — after that, switching back and forth is free.

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

Robot descriptions are downloaded on first use. `robot_descriptions` normally shells out to
a `git` binary, which most Blender artists do not have; Kinema replaces that one function
with an HTTPS fetch (`src/kinema/catalog/fetch.py`).

Downloads run on a worker thread and rig building is spread across modal timer ticks, so
the import never blocks Blender's event loop and can be cancelled with Esc.

### Per-robot downloads

`mujoco_menagerie` is 1.64 GB and backs 49 of the 186 catalog robots, so a naive fetch pulls
the whole Menagerie to get one quadruped. Description modules only call `os.path.join` at
import time, so the directory a robot needs is derivable offline; GitHub's tree API plus the
raw CDN then fetch just that. A Unitree Go2 costs **31 MB instead of 1.7 GB**.

The cache layout is unchanged — the repository-named directory stays and only one
subdirectory inside it is populated — so paths still resolve, an existing full checkout is
reused rather than re-fetched, and a second robot from the same repository is added
incrementally. Sparse fetching is opt-in per repository (`SPARSE_REPOSITORIES`), because a
subtree referencing shared assets outside itself would fetch cleanly and then fail at
mesh-load time with only a warning.

Kinema shares `~/.cache/robot_descriptions` with robot_descriptions itself and honours
`ROBOT_DESCRIPTIONS_CACHE`; the *Robot Cache* preference sets the same variable. Sharing is
one-way: Kinema reuses an existing git clone, but plain robot_descriptions sees no `.git` in
Kinema's cache and will re-clone.

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
