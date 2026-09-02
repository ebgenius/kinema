"""Resolve demo robot keys to local description files.

The demo scripts run inside Blender, and Kinema no longer downloads anything:
its catalogue is an offline index that hands the user a ``git clone`` command.
That is the right behaviour for the add-on and the wrong one for a demo, which
must not need a human to fetch a UR5e by hand before it can render a GIF.

So the download moves to the developer's venv, where ``robot_descriptions``
still lives as a dev dependency, and the two halves meet through a small JSON
map on disk:

    uv run python tools/demo/robots.py          # once: fetch and record paths
    blender --background --python tools/dev_bootstrap.py \
            --python tools/demo/sweep.py -- measurements.csv

Inside Blender, :func:`resolve` reads that map. Set ``KINEMA_DEMO_ROBOTS`` to
put it somewhere other than beside this file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

#: Every robot the demo scripts ask for. Keys are ``robot_descriptions`` names.
DEMO_ROBOTS = ("ur5e_description", "panda_mj_description")


def map_path() -> Path:
    override = os.environ.get("KINEMA_DEMO_ROBOTS")
    return Path(override) if override else Path(__file__).resolve().parent / ".robots.json"


def resolve(key: str) -> str:
    """The local description file for one demo robot.

    Blender-side. Raises with the command to run rather than a KeyError, so a
    stale or missing map is self-explanatory in a headless log.
    """
    path = map_path()
    try:
        mapping = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"demo robot map missing or unreadable ({path}): {exc}. "
            "Run: uv run python tools/demo/robots.py"
        ) from exc

    resolved = mapping.get(key)
    if not resolved or not Path(resolved).is_file():
        raise RuntimeError(
            f"'{key}' is not in the demo robot map ({path}). "
            "Run: uv run python tools/demo/robots.py"
        )
    return resolved


def fetch() -> dict[str, str]:
    """Dev-venv side: download each demo robot and record where it landed."""
    os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
    import importlib

    mapping: dict[str, str] = {}
    for key in DEMO_ROBOTS:
        module = importlib.import_module(f"robot_descriptions.{key}")
        # URDF first: the demos rig from a URDF wherever one exists, and fall
        # back to MJCF for the menagerie-only robots.
        for attr in ("URDF_PATH", "MJCF_PATH"):
            found = getattr(module, attr, None)
            if found:
                mapping[key] = str(Path(found).resolve())
                break
        else:
            raise RuntimeError(f"'{key}' provides neither URDF_PATH nor MJCF_PATH")
        print(f"  {key} -> {mapping[key]}")
    return mapping


def main() -> int:
    print(f"fetching {len(DEMO_ROBOTS)} demo robots (this downloads on first run)")
    mapping = fetch()
    target = map_path()
    target.write_text(json.dumps(mapping, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
