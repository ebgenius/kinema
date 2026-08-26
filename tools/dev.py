"""Developer entry point for Kinema.

    uv run python tools/dev.py link       # symlink the add-on into Blender
    uv run python tools/dev.py run        # launch Blender with Kinema enabled
    uv run python tools/dev.py debug      # ... and wait for a debugger to attach
    uv run python tools/dev.py test       # run the in-Blender test suite
    uv run python tools/dev.py validate   # blender --command extension validate
    uv run python tools/dev.py build      # build per-platform extension zips

``link`` installs the add-on as a live link rather than a copy, so editing a
file under ``src/kinema/`` and toggling the extension off/on in Blender picks
the change up immediately -- no rebuild, no reinstall.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ADDON_SRC = REPO_ROOT / "src" / "kinema"
BOOTSTRAP = REPO_ROOT / "tools" / "dev_bootstrap.py"
DIST_DIR = REPO_ROOT / "dist"

EXTENSION_REPO = "user_default"
EXTENSION_ID = f"bl_ext.{EXTENSION_REPO}.kinema"
DEFAULT_DEBUG_PORT = 5678


# --------------------------------------------------------------------------
# Locating Blender
# --------------------------------------------------------------------------
def find_blender() -> Path:
    """Locate a Blender 5.2+ executable.

    ``KINEMA_BLENDER`` wins; otherwise search the usual install roots plus
    Blender Launcher's build directory, which keeps versioned builds side by
    side and is not on PATH.
    """
    override = os.environ.get("KINEMA_BLENDER")
    if override:
        path = Path(override)
        if path.is_file():
            return path
        raise SystemExit(f"KINEMA_BLENDER does not point at a file: {override}")

    on_path = shutil.which("blender")
    if on_path:
        return Path(on_path)

    system = platform.system()
    if system == "Windows":
        exe, roots = "blender.exe", [
            Path.home() / "blender",                          # Blender Launcher
            Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Blender Foundation",
            Path.home() / "AppData/Local/Programs/Blender Foundation",
        ]
    elif system == "Darwin":
        exe, roots = "Blender", [
            Path("/Applications/Blender.app/Contents/MacOS"),
            Path.home() / "Applications/Blender.app/Contents/MacOS",
        ]
    else:
        exe, roots = "blender", [
            Path("/usr/bin"), Path("/usr/local/bin"),
            Path("/opt/blender"), Path.home() / ".local/share/blender",
        ]

    candidates: list[Path] = []
    for root in roots:
        if root.is_dir():
            candidates += [p for p in root.rglob(exe) if p.is_file()]
    if not candidates:
        raise SystemExit(
            "Could not find Blender. Set KINEMA_BLENDER to the executable path."
        )
    # Newest path wins -- version strings sort usefully here (5.2 > 5.1 > 4.5).
    return sorted(candidates)[-1]


def blender_version(blender: Path) -> str:
    out = subprocess.run([str(blender), "--version"], capture_output=True, text=True).stdout
    first = out.strip().splitlines()[0] if out.strip() else ""
    return first.replace("Blender", "").strip()


def blender_config_dir(blender: Path) -> Path:
    """User config root for the given Blender build (e.g. .../Blender/5.2)."""
    version = ".".join(blender_version(blender).split(".")[:2]) or "5.2"
    system = platform.system()
    if system == "Windows":
        base = Path(os.environ["APPDATA"]) / "Blender Foundation" / "Blender"
    elif system == "Darwin":
        base = Path.home() / "Library/Application Support/Blender"
    else:
        base = Path(
            os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")
        ) / "blender"
    return base / version


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
def cmd_link(args: argparse.Namespace) -> int:
    blender = find_blender()
    target = blender_config_dir(blender) / "extensions" / EXTENSION_REPO / "kinema"
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.is_symlink() or target.exists():
        if target.is_symlink():
            target.unlink()
        elif target.is_dir():
            # A junction reports as a dir; rmtree handles both.
            shutil.rmtree(target, ignore_errors=True)
        print(f"removed existing {target}")

    if platform.system() == "Windows":
        # A directory junction needs no admin rights and no Developer Mode,
        # unlike os.symlink on Windows.
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(target), str(ADDON_SRC)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise SystemExit(f"mklink failed: {result.stderr.strip()}")
    else:
        target.symlink_to(ADDON_SRC, target_is_directory=True)

    print(f"linked {target}\n    -> {ADDON_SRC}")
    print(f"Blender {blender_version(blender)} at {blender}")
    return 0


def cmd_unlink(args: argparse.Namespace) -> int:
    target = blender_config_dir(find_blender()) / "extensions" / EXTENSION_REPO / "kinema"
    if target.is_symlink():
        target.unlink()
    elif target.exists():
        shutil.rmtree(target, ignore_errors=True)
    else:
        print("nothing linked")
        return 0
    print(f"unlinked {target}")
    return 0


def _launch_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    env["KINEMA_EXT_ID"] = EXTENSION_ID
    # Point the bootstrap at this interpreter's site-packages (uv run gives us
    # the project venv), so debugpy and friends are importable inside Blender.
    env["KINEMA_VENV_SITE"] = sysconfig.get_paths()["purelib"]
    if getattr(args, "port", None):
        env["KINEMA_DEBUG_PORT"] = str(args.port)
        env["KINEMA_DEBUG_WAIT"] = "1" if getattr(args, "wait", False) else "0"
    return env


def cmd_run(args: argparse.Namespace) -> int:
    blender = find_blender()
    cmd = [str(blender)]
    if args.factory_startup:
        cmd.append("--factory-startup")
    cmd += ["--python", str(BOOTSTRAP)]
    if args.blend:
        cmd.insert(1, args.blend)
    print(f"launching {blender} ({blender_version(blender)})")
    return subprocess.call(cmd, env=_launch_env(args))


def cmd_debug(args: argparse.Namespace) -> int:
    args.port = args.port or DEFAULT_DEBUG_PORT
    print(
        f"Blender will listen for a debugger on 127.0.0.1:{args.port}.\n"
        f"Attach with any DAP client, e.g. VS Code launch.json:\n"
        f'  {{"type":"debugpy","request":"attach",'
        f'"connect":{{"host":"127.0.0.1","port":{args.port}}}}}'
    )
    return cmd_run(args)


def cmd_test(args: argparse.Namespace) -> int:
    blender = find_blender()
    runner = REPO_ROOT / "tests" / "run_in_blender.py"
    if not runner.is_file():
        raise SystemExit(f"missing test runner: {runner}")
    cmd = [str(blender), "--background", "--factory-startup",
           "--python", str(BOOTSTRAP), "--python", str(runner)]
    if args.pytest_args:
        cmd += ["--", *args.pytest_args]
    return subprocess.call(cmd, env=_launch_env(args))


def cmd_validate(args: argparse.Namespace) -> int:
    blender = find_blender()
    return subprocess.call(
        [str(blender), "--command", "extension", "validate", str(ADDON_SRC)]
    )


def cmd_build(args: argparse.Namespace) -> int:
    blender = find_blender()
    DIST_DIR.mkdir(exist_ok=True)
    cmd = [str(blender), "--command", "extension", "build",
           "--source-dir", str(ADDON_SRC), "--output-dir", str(DIST_DIR)]
    if not args.single:
        # Each platform zip carries only its own wheels; a combined zip would
        # hold three copies of jaxlib and blow the platform's size limit.
        cmd.append("--split-platforms")
    code = subprocess.call(cmd)
    if code == 0:
        for zip_path in sorted(DIST_DIR.glob("*.zip")):
            print(f"  {zip_path.name}  {zip_path.stat().st_size / 1e6:.1f} MB")
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("link", help="link the add-on into Blender's extensions dir")
    sub.add_parser("unlink", help="remove the link")

    run = sub.add_parser("run", help="launch Blender with Kinema enabled")
    run.add_argument("--blend", help="optional .blend file to open")
    run.add_argument("--factory-startup", action="store_true",
                     help="ignore user preferences and startup file")
    run.add_argument("--port", type=int, help="also start debugpy on this port")

    debug = sub.add_parser("debug", help="launch Blender waiting for a debugger")
    debug.add_argument("--blend")
    debug.add_argument("--factory-startup", action="store_true")
    debug.add_argument("--port", type=int, default=DEFAULT_DEBUG_PORT)
    debug.add_argument("--no-wait", dest="wait", action="store_false", default=True,
                       help="start listening but do not block on attach")

    test = sub.add_parser("test", help="run tests inside Blender")
    test.add_argument("pytest_args", nargs="*", help="extra args passed to pytest")

    sub.add_parser("validate", help="validate blender_manifest.toml")

    build = sub.add_parser("build", help="build extension zips")
    build.add_argument("--single", action="store_true",
                       help="one combined zip instead of per-platform zips")

    args = parser.parse_args()
    handlers = {
        "link": cmd_link, "unlink": cmd_unlink, "run": cmd_run, "debug": cmd_debug,
        "test": cmd_test, "validate": cmd_validate, "build": cmd_build,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
