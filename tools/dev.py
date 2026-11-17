"""Developer entry point for Kinema.

    uv run python tools/dev.py link       # symlink the add-on into Blender
    uv run python tools/dev.py run        # launch Blender with Kinema enabled
    uv run python tools/dev.py debug      # ... and wait for a debugger to attach
    uv run python tools/dev.py test       # run the in-Blender test suite
    uv run python tools/dev.py validate   # blender --command extension validate
    uv run python tools/dev.py build      # build per-platform extension zips

``build`` refuses to run unless ``tools/vendor.py`` and ``tools/fetch_wheels.py``
have populated ``vendor/`` and ``wheels/`` -- both are gitignored, so a fresh
clone has neither and the zip would otherwise be built without a solver in it.

``link`` installs the add-on as a live link rather than a copy, so editing a
file under ``src/kinema/`` and toggling the extension off/on in Blender picks
the change up immediately -- no rebuild, no reinstall.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ADDON_SRC = REPO_ROOT / "src" / "kinema"
VENDOR_DIR = ADDON_SRC / "vendor"
WHEEL_DIR = ADDON_SRC / "wheels"
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


#: Each test's last measured time, which a split suite is balanced by: tests/sharding.py.
DURATIONS = REPO_ROOT / "tests" / "durations.json"
#: Memory one Blender test process can take at its peak, solving and compiling.
GB_PER_JOB = 6


def _memory_gb() -> float | None:
    """The memory free for the tests now, in GB, or None where it can't be read.

    What is free, not the total: a Blender, a browser or another test run
    already open has its share. On Windows, the lower of the physical memory
    free and the commit charge left, since memory is refused past the commit
    limit even with RAM to spare. On Linux, ``MemAvailable``, and in a container
    no more than its limit less what it uses. Elsewhere, the total.
    """
    if sys.platform == "win32":
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        ("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                        ("page_total", ctypes.c_ulonglong), ("page_free", ctypes.c_ulonglong),
                        ("virtual_total", ctypes.c_ulonglong), ("virtual_free", ctypes.c_ulonglong),
                        ("extended", ctypes.c_ulonglong)]

        status = Status(length=ctypes.sizeof(Status))
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            # page_free is the commit charge left, page file and RAM together.
            return min(status.free, status.page_free) / 2**30
        return None
    free = [gb for gb in (_available_gb(), _container_free_gb()) if gb is not None]
    if free:
        return min(free)
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        return None


def _available_gb(meminfo: Path = Path("/proc/meminfo")) -> float | None:
    """Linux's estimate of the memory a new process can have, or None."""
    try:
        for line in meminfo.read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20  # in kB
    except (OSError, ValueError, IndexError):
        return None
    return None


def _container_free_gb(root: Path = Path("/sys/fs/cgroup")) -> float | None:
    """What this process's control group may still use on Linux, its limit less
    what it uses, or None where it has no limit."""
    for limit_name, used_name in (
        ("memory.max", "memory.current"),  # cgroup v2
        ("memory/memory.limit_in_bytes", "memory/memory.usage_in_bytes"),  # v1
    ):
        try:
            text = (root / limit_name).read_text(encoding="ascii").strip()
        except OSError:
            continue
        if not (text.isdigit() and int(text) < 2**60):  # "max", or v1's huge number
            return None
        try:
            used = int((root / used_name).read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            used = 0
        return max(0, int(text) - used) / 2**30
    return None


def _jobs(asked: str) -> int:
    """How many Blender processes to run the suite in: ``auto``, or a number.

    ``auto`` is half the logical cores, and no more than the memory free holds
    at :data:`GB_PER_JOB` each, at most 8. Measured on 20 cores and 64 GB: four
    processes solved as fast each as one did, and eight 25% slower each; the
    whole suite took 281 s in five and 225 s in eight, where one took 21 minutes.
    """
    if asked != "auto":
        return max(1, int(asked))
    cores = os.cpu_count() or 1
    memory = _memory_gb()
    by_memory = int(memory // GB_PER_JOB) if memory else 1
    return max(1, min(cores // 2, by_memory, 8))


def _summary(log: str) -> str:
    """The last line pytest printed with its counts, or how the process ended."""
    for line in reversed(log.splitlines()):
        if any(word in line for word in (" passed", " failed", " error", "no tests ran")):
            return line.strip().strip("=").strip()
    return "no summary: see its log"


def _read_json(path: Path):
    """A JSON file's value, or None where it is missing, unreadable or cut short."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _times(value) -> dict[str, float]:
    """``value``'s times in seconds by test, or none where it isn't a mapping of them."""
    if not isinstance(value, dict):
        return {}
    return {
        str(k): float(v) for k, v in value.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }


def _merged_durations(recorded, parts: list) -> dict[str, float]:
    """The recorded times, with each process's measured ones in their place.

    ``recorded`` and each part are as read, so maybe None or of the wrong shape;
    a bad file counts as no times, and is written over whole. A test no process
    collected -- renamed or removed since -- is dropped, so its time isn't kept
    forever, but only where every process says it collected the whole suite
    without an error. After part of it, a failed import, or a process whose
    report can't be read, the others are kept as they were.
    """
    merged = _times(recorded)
    collected: set[str] = set()
    whole = bool(parts)
    for part in parts:
        if not isinstance(part, dict):
            whole = False
            continue
        merged.update(_times(part.get("times")))
        collected.update(str(n) for n in part.get("collected") or ())
        whole = whole and part.get("whole") is True
    if whole and collected:
        merged = {k: v for k, v in merged.items() if k in collected}
    return dict(sorted(merged.items()))


def _reports(logs: Path, jobs: int) -> list:
    """Each process's report of the times it measured, by its number: one that
    wrote none is None, as one that can't be read, and keeps every time."""
    return [_read_json(logs / f"durations-{index}.json") for index in range(jobs)]


def _write_durations(path: Path, durations: dict[str, float]) -> None:
    """Write ``durations`` over ``path`` whole: to a file of this run's own beside
    it, moved over it, so it is never left cut short, and two runs recording at
    once never write or move each other's."""
    handle, written = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            out.write(json.dumps(durations, indent=0) + "\n")
        os.replace(written, path)
    except BaseException:
        Path(written).unlink(missing_ok=True)
        raise


def _run_shares(cmd: list[str], jobs: int, logs: Path, env_for) -> list[int]:
    """Start one Blender per share, logging to ``logs``, and wait for them all.

    Returns their exit codes. Should one fail to start, or the wait be
    interrupted, those already started are stopped: none is left running.
    """
    running = []
    try:
        for index in range(jobs):
            out = (logs / f"share-{index}.log").open("w", encoding="utf-8", errors="replace")
            try:
                process = subprocess.Popen(
                    cmd, env=env_for(index), stdout=out, stderr=subprocess.STDOUT
                )
            except BaseException:
                out.close()
                raise
            running.append((process, out))
        return [process.wait() for process, _ in running]
    finally:
        for process, out in running:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            out.close()


def cmd_test(args: argparse.Namespace) -> int:
    blender = find_blender()
    runner = REPO_ROOT / "tests" / "run_in_blender.py"
    if not runner.is_file():
        raise SystemExit(f"missing test runner: {runner}")
    cmd = [str(blender), "--background", "--factory-startup",
           "--python", str(BOOTSTRAP), "--python", str(runner)]
    if args.pytest_args:
        cmd += ["--", *args.pytest_args]
    jobs = _jobs(args.jobs)
    if jobs == 1 and not args.record_durations:
        return subprocess.call(cmd, env=_launch_env(args))

    # Each process works out its own share: tests/sharding.py.
    logs = Path(tempfile.mkdtemp(prefix="kinema-test-"))
    print(f"running the tests in {jobs} Blender processes at once; logs in {logs}")
    started = time.perf_counter()

    def env_for(index: int) -> dict[str, str]:
        env = _launch_env(args)
        env["KINEMA_TEST_SHARD"] = f"{index}/{jobs}"
        if args.record_durations:
            env["KINEMA_RECORD_DURATIONS"] = str(logs / f"durations-{index}.json")
        return env

    codes = _run_shares(cmd, jobs, logs, env_for)
    worst = 0
    for index, code in enumerate(codes):
        log = (logs / f"share-{index}.log").read_text(encoding="utf-8", errors="replace")
        for line in log.splitlines():
            if line.startswith(("FAILED", "ERROR")):
                print(f"  [{index + 1}/{jobs}] {line}")
        print(f"share {index + 1}/{jobs}: {_summary(log)}")
        # 5 is pytest's "no tests collected": a share a narrow selection left empty.
        if code not in (0, 5):
            worst = worst or code
    print(f"{jobs} processes, {time.perf_counter() - started:.0f} s in all")

    if args.record_durations:
        recorded = _merged_durations(_read_json(DURATIONS), _reports(logs, jobs))
        _write_durations(DURATIONS, recorded)
        print(f"times of {len(recorded)} tests written to {DURATIONS.relative_to(REPO_ROOT)}")
    if all(code == 5 for code in codes):
        return 5
    return worst


def cmd_validate(args: argparse.Namespace) -> int:
    blender = find_blender()
    return subprocess.call(
        [str(blender), "--command", "extension", "validate", str(ADDON_SRC)]
    )


def preflight() -> None:
    """Refuse to build a payload that is missing pieces.

    ``blender --command extension build`` zips whatever is on disk. A missing
    ``vendor/`` or ``wheels/`` is a smaller zip, not an error, and ``validate``
    only parses the manifest -- so both pass. Both directories are gitignored,
    so a fresh clone has neither, and an extension built without them installs
    cleanly and fails at the first solve, in front of a user.

    Being gitignored also means they *persist* across branches and pulls, so a
    tree can be present, at the right commit, and still wrong: ``vendor.py``
    changes what it does to a tree after cloning -- dropped directories,
    rewritten imports -- independently of the pins. The recipe fingerprint is
    checked alongside the commit for exactly that case.
    """
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    import fetch_wheels
    import vendor

    problems: list[str] = []

    for pkg in vendor.PACKAGES:
        state = vendor.recorded_state(VENDOR_DIR / pkg.name)
        if state is None:
            problems.append(f"vendored {pkg.name} is missing from {VENDOR_DIR}")
            continue
        recorded, recipe = state
        if recorded != pkg.commit:
            problems.append(
                f"vendored {pkg.name} is at {recorded[:10]}, "
                f"pinned at {pkg.commit[:10]}"
            )
        expected = vendor.recipe_fingerprint(pkg)
        if recipe != expected:
            problems.append(
                f"vendored {pkg.name} was built with recipe "
                f"{recipe or '(none)'}, tools/vendor.py now specifies {expected}"
            )

    wheels = sorted(w.name for w in WHEEL_DIR.glob("*.whl"))
    if not wheels:
        problems.append(f"no bundled wheels in {WHEEL_DIR}")
    else:
        # The payload can also be present but incoherent -- one package at two
        # versions because a platform tag went unmatched and pip fell back.
        try:
            fetch_wheels.check_version_skew(wheels)
        except SystemExit as exc:
            problems.append(str(exc))

    if problems:
        raise SystemExit(
            "dev.py build: the add-on tree is incomplete\n\n"
            + "\n".join(f"  - {p}" for p in problems)
            + "\n\nRun, from the repo root:\n"
            "  uv run python tools/vendor.py\n"
            "  uv run python tools/fetch_wheels.py"
        )


def cmd_build(args: argparse.Namespace) -> int:
    if args.skip_preflight:
        print("skipping preflight: the zips may be incomplete")
    else:
        preflight()
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
    test.add_argument(
        "-j", "--jobs", default="auto",
        help="Blender processes to split the tests across: a number, or auto "
             "(half the cores, within the memory, at most 8). 1 runs them in one, "
             "as before",
    )
    test.add_argument(
        "--record-durations", action="store_true",
        help="time every test and write the times to tests/durations.json, "
             "which the split is balanced by",
    )
    test.add_argument("pytest_args", nargs="*", help="extra args passed to pytest")

    sub.add_parser("validate", help="validate blender_manifest.toml")

    build = sub.add_parser("build", help="build extension zips")
    build.add_argument("--single", action="store_true",
                       help="one combined zip instead of per-platform zips")
    build.add_argument("--skip-preflight", action="store_true",
                       help="build even if vendor/ or wheels/ are incomplete")

    args = parser.parse_args()
    handlers = {
        "link": cmd_link, "unlink": cmd_unlink, "run": cmd_run, "debug": cmd_debug,
        "test": cmd_test, "validate": cmd_validate, "build": cmd_build,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
