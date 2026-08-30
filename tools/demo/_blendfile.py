"""Save the scene each demo script builds, for opening by hand afterwards.

These are inspection aids, not build outputs: a .blend here lets you open the
exact scene a headless run produced, scrub the bake and check the claim in the
viewport rather than trusting a number in a log.

They are deliberately **not committed** -- each is tens of megabytes of packed
robot meshes, and every one is regenerable by re-running its script. The default
location is gitignored; set ``KINEMA_DEMO_BLEND_DIR`` to put them elsewhere.
"""
from __future__ import annotations

import os
from pathlib import Path

#: Gitignored; see .gitignore.
DEFAULT_DIR = Path(__file__).resolve().parent / "blend"


def blend_dir() -> Path:
    return Path(os.environ.get("KINEMA_DEMO_BLEND_DIR", str(DEFAULT_DIR)))


def save_blend(name: str) -> Path | None:
    """Write the current scene to ``<blend_dir>/<name>.blend``.

    Never fatal: a demo that rendered correctly should not fail at the last step
    because a scratch file could not be written.
    """
    import bpy

    target = blend_dir() / f"{name}.blend"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # compress: these are mostly mesh data and shrink a long way.
        bpy.ops.wm.save_as_mainfile(filepath=str(target), compress=True,
                                    copy=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[blend] could not save {target}: {exc}", flush=True)
        return None
    size = target.stat().st_size / 1e6 if target.exists() else 0.0
    print(f"[blend] saved {target}  ({size:.1f} MB, not committed)", flush=True)
    return target
