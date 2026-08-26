"""Startup script run inside Blender by ``tools/dev.py``.

Passed to Blender as ``--python``. It does three things:

1. Puts the dev virtualenv's ``site-packages`` on Blender's ``sys.path``. This
   is only safe because ``pyproject.toml`` pins ``requires-python == 3.13.*``
   to match Blender 5.2's embedded CPython 3.13.13 -- a mismatched venv would
   put wheels with the wrong ABI in front of Blender's own.
2. Optionally starts a ``debugpy`` DAP server. Any DAP-capable editor attaches
   to ``localhost:<port>``: VS Code, PyCharm, neovim-dap, Helix. No
   Blender-specific IDE extension is involved, which is the point.
3. Enables the Kinema extension so a launch lands in a ready state.

Configuration comes from the environment so no arguments have to survive
Blender's own argv parsing:

``KINEMA_VENV_SITE``   path to the dev venv site-packages
``KINEMA_DEBUG_PORT``  port to listen on (unset disables debugging)
``KINEMA_DEBUG_WAIT``  "1" to block until a client attaches
``KINEMA_EXT_ID``      extension module name to enable
"""

from __future__ import annotations

import os
import sys


def _add_venv_site_packages() -> None:
    site = os.environ.get("KINEMA_VENV_SITE")
    if not site or not os.path.isdir(site):
        return
    # Appended, not prepended: Blender's own NumPy must keep priority over any
    # copy in the dev venv, or compiled modules can bind to the wrong ABI.
    if site not in sys.path:
        sys.path.append(site)
    print(f"[kinema-dev] dev site-packages: {site}")


def _start_debugpy() -> None:
    port_text = os.environ.get("KINEMA_DEBUG_PORT")
    if not port_text:
        return
    try:
        import debugpy
    except ImportError:
        print("[kinema-dev] debugpy not installed; run: uv sync --group dev")
        return

    port = int(port_text)
    try:
        # Blender's own interpreter is the one to debug, so point debugpy at it
        # explicitly -- otherwise it may try to launch a separate python.exe.
        debugpy.configure(python=sys.executable)
        debugpy.listen(("127.0.0.1", port))
    except Exception as exc:  # noqa: BLE001 - never block a dev launch
        print(f"[kinema-dev] could not start debugpy: {exc}")
        return

    print(f"[kinema-dev] debugpy listening on 127.0.0.1:{port}")
    if os.environ.get("KINEMA_DEBUG_WAIT") == "1":
        print("[kinema-dev] waiting for debugger to attach…")
        debugpy.wait_for_client()
        print("[kinema-dev] debugger attached")


def _enable_extension() -> None:
    ext_id = os.environ.get("KINEMA_EXT_ID")
    if not ext_id:
        return
    import addon_utils
    import bpy

    try:
        bpy.ops.preferences.addon_refresh()
    except Exception:  # noqa: BLE001
        pass

    default, enabled = addon_utils.check(ext_id)
    if not enabled:
        addon_utils.enable(ext_id, default_set=True, persistent=True)
    state = addon_utils.check(ext_id)[1]
    print(f"[kinema-dev] extension {ext_id}: {'enabled' if state else 'FAILED TO ENABLE'}")


def main() -> None:
    _add_venv_site_packages()
    _start_debugpy()
    _enable_extension()


main()
