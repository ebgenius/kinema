"""Operators that turn a robot description into a Kinema rig.

One import path -- **Import URDF File**: a local URDF, xacro or MJCF, with
``package://`` references resolved by searching the file's own tree.

Alongside it sits **Browse Robot Catalog**, which imports nothing. It searches
the offline catalogue (``catalog.index``) and hands back a ``git clone``
command plus the path of the description file inside that repository, for the
user to fetch by hand. Kinema does not download robots.

The import hands off to a modal worker, :class:`KINEMA_OT_build_robot`, because
it is slow enough to freeze Blender if run straight through: a humanoid costs
one Blender mesh-importer call per visual -- a couple of hundred of them. Run
synchronously that is long enough for Windows to decide the process has hung
and offer to kill it.

So the work is split by what may touch ``bpy``:

* **Parsing** goes to a worker thread (``io.loader``, which imports no ``bpy``
  at all).
* **Armature and mesh building** stay on the main thread, because Blender's API
  is not thread-safe, but run in short slices across modal timer ticks
  (``rig.builder.build_rig_iter``).

The modal timer is the load-bearing part. ``wm.progress_update`` draws a cursor
but does not pump events; only returning to Blender's event loop keeps the
window alive.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper

from ..catalog import index as catalog
from ..io import loader
from ..rig import builder

#: How often the modal operator wakes up, and how long it works before handing
#: control back. The gap between them is the responsiveness dial: 40 ms of work
#: per 50 ms of wall clock keeps ~80% throughput with a 20 Hz UI floor. Setting
#: them equal would halve throughput for nothing.
_TIMER_INTERVAL = 0.01
_TICK_BUDGET = 0.04

#: One import at a time: two concurrent builds would interleave objects in the
#: scene.
_JOB_LOCK = threading.Lock()


def import_settings() -> dict:
    """Property annotations shared by the import operators and the sidebar.

    Blender reads properties from a class's *own* ``__annotations__``; it does
    not walk base classes. So a mixin can share methods but not properties, and
    each class merges this dict into its own annotations instead.
    """
    return {
        "bone_length": FloatProperty(
            name="Bone Size",
            description=(
                "Display size of the joint controls. 0 picks one from the robot's scale"
            ),
            default=0.0, min=0.0, max=1.0, subtype="DISTANCE",
        ),
        "enforce_limits": BoolProperty(
            name="Enforce Joint Limits",
            description=(
                "Add limit constraints from the URDF. Turn off to pose past a robot's "
                "real limits for a shot"
            ),
            default=True,
        ),
        "import_visuals": BoolProperty(
            name="Import Meshes",
            description="Load the robot's visual geometry and parent it to the rig",
            default=True,
        ),
        "create_tcp": BoolProperty(
            name="Create TCP",
            description="Add a tool-centre-point marker at the end of the chain",
            default=True,
        ),
    }


#: The four names in :func:`import_settings`, for copying between an operator,
#: the scene property group, and a RigBuildOptions.
SETTING_NAMES = ("bone_length", "enforce_limits", "import_visuals", "create_tcp")


def setting_values(source) -> dict:
    """Read the import settings off an operator or the scene property group."""
    return {name: getattr(source, name) for name in SETTING_NAMES}


def _options_from(source) -> builder.RigBuildOptions:
    values = setting_values(source)
    return builder.RigBuildOptions(
        bone_length=values["bone_length"] or None,
        enforce_limits=values["enforce_limits"],
        import_visuals=values["import_visuals"],
        create_tcp=values["create_tcp"],
    )


def _human_bytes(count: int) -> str:
    if count >= 1 << 20:
        return f"{count / (1 << 20):.0f} MB"
    return f"{count / 1024:.0f} KB"


# --------------------------------------------------------------------------
# the modal worker
# --------------------------------------------------------------------------
class _Shared:
    """State handed from the worker thread to ``modal()``.

    Guarded by a lock because the thread writes progress while the main thread
    reads it. Nothing here is a ``bpy`` object -- the thread must never touch
    one, and must never call ``Operator.report`` either; all reporting happens
    in ``modal()``.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.fraction = 0.0
        self.done = 0
        self.total = 0
        self.finished = False
        self.result: loader.LoadResult | None = None

    def progress(self, fraction: float, done: int, total: int) -> None:
        with self.lock:
            self.fraction, self.done, self.total = fraction, done, total

    def snapshot(self) -> tuple[float, int, int, bool, loader.LoadResult | None]:
        with self.lock:
            return self.fraction, self.done, self.total, self.finished, self.result


class KINEMA_OT_build_robot(Operator):
    """Parse and build a robot without blocking Blender.

    Not exposed in any menu: it is the engine behind the URDF file browser,
    which invokes it with the file already chosen.
    """

    bl_idname = "kinema.build_robot"
    bl_label = "Build Robot Rig"
    bl_description = "Parse and rig a robot description"
    # UNDO so Ctrl+Z removes the rig; deliberately no REGISTER, because the redo
    # panel would re-run the whole import on every parameter tweak.
    bl_options = {"UNDO"}

    # SKIP_SAVE: passed in by the caller, and without it Blender restores the
    # previous invocation's value over it.
    filepath: StringProperty(subtype="FILE_PATH", options={"SKIP_SAVE"})
    __annotations__.update(import_settings())

    # Modal state, defaulted at class level so the teardown guards hold even on
    # an instance that never went modal -- execute() returns before invoke()
    # initialises any of this, and Blender may still call cancel().
    _torn_down = False
    _timer = None
    _steps = None
    _result = None

    # -------------------------------------------------------- entry points
    def execute(self, context: bpy.types.Context) -> set[str]:
        """The whole import in one call, blocking.

        This is what a script or a background Blender gets. Interactive use goes
        through ``invoke`` and the modal path instead.
        """
        return self._run_synchronously(context)

    def invoke(self, context: bpy.types.Context, event) -> set[str]:
        if bpy.app.background or context.window is None:
            # No window means no modal handler and no TIMER events -- which is
            # exactly how `dev.py test` runs Blender.
            return self.execute(context)

        if not _JOB_LOCK.acquire(blocking=False):
            self.report({"WARNING"}, "Kinema: an import is already running")
            return {"CANCELLED"}

        self._stage = "FETCH"
        self._torn_down = False  # explicit, in case Blender reuses the instance
        self._shared = _Shared()
        self._cancel = threading.Event()
        self._steps = None
        self._result = None
        self._load: loader.LoadResult | None = None
        self._label = Path(self.filepath).name

        self._thread = threading.Thread(
            target=self._work, name="kinema-import", daemon=True
        )
        self._thread.start()

        window_manager = context.window_manager
        window_manager.progress_begin(0, 1000)
        self._set_status(context, f"Kinema: reading {self._label}… (Esc to cancel)")
        self._timer = window_manager.event_timer_add(
            _TIMER_INTERVAL, window=context.window
        )
        window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    # ---------------------------------------------------------------- worker
    def _work(self) -> None:
        """Runs on the worker thread. Must not touch bpy in any way."""
        shared = self._shared
        try:
            result = loader.load_file(self.filepath, should_cancel=self._cancel.is_set)
        except Exception as exc:  # noqa: BLE001 - a thread must not raise into nothing
            result = loader.LoadResult(error=f"{type(exc).__name__}: {exc}")
        with shared.lock:
            shared.result = result
            shared.finished = True

    # ----------------------------------------------------------------- modal
    def modal(self, context: bpy.types.Context, event) -> set[str]:
        if event.type == "ESC" and event.value == "PRESS":
            self.report({"INFO"}, "Kinema: import cancelled")
            return self._abort(context)
        if event.type != "TIMER":
            # Let the viewport keep working. A window that repaints but ignores
            # the mouse reads as barely better than a frozen one.
            return {"PASS_THROUGH"}

        if self._stage == "FETCH":
            return self._tick_fetch(context)
        return self._tick_build(context)

    def _tick_fetch(self, context: bpy.types.Context) -> set[str]:
        fraction, done, total, finished, result = self._shared.snapshot()

        if not finished:
            context.window_manager.progress_update(int(max(fraction, 0.0) * 1000))
            if total:
                self._set_status(
                    context,
                    f"Kinema: downloading {self._label} — "
                    f"{_human_bytes(done)} / {_human_bytes(total)} (Esc to cancel)",
                )
            elif done:
                self._set_status(
                    context,
                    f"Kinema: downloading {self._label} — "
                    f"{_human_bytes(done)} (Esc to cancel)",
                )
            return {"RUNNING_MODAL"}

        self._thread.join(timeout=1.0)
        if result is None or result.cancelled:
            return self._abort(context)
        if result.error:
            self.report({"ERROR"}, result.error)
            return self._abort(context)
        if not result.model.actuated_joints:
            self.report({"WARNING"}, f"'{result.model.name}' has no movable joints")

        self._load = result
        self._result = builder.RigBuildResult()
        self._steps = builder.build_rig_iter(
            result.model, _options_from(self), result=self._result
        )
        self._stage = "BUILD"
        return {"RUNNING_MODAL"}

    def _tick_build(self, context: bpy.types.Context) -> set[str]:
        deadline = time.perf_counter() + _TICK_BUDGET
        done = total = 0
        while time.perf_counter() < deadline:
            try:
                done, total = next(self._steps)
            except StopIteration as stop:
                self._result = stop.value
                return self._finish(context)
            except Exception as exc:  # noqa: BLE001
                self.report({"ERROR"}, f"Could not build rig: {exc}")
                return self._abort(context)

        if total:
            context.window_manager.progress_update(int(done / total * 1000))
            self._set_status(
                context, f"Kinema: building {self._label} — mesh {done}/{total}"
            )
        return {"RUNNING_MODAL"}

    # -------------------------------------------------------------- teardown
    def _finish(self, context: bpy.types.Context) -> set[str]:
        result, load = self._result, self._load

        if load.source is not None and result.armature_object is not None:
            # Record where this came from: the solver reloads the description
            # later to build PyRoki's robot model, and a saved .blend must still
            # know.
            kind, value = load.source
            result.armature_object[builder.PROP_SOURCE_KIND] = kind
            result.armature_object[builder.PROP_SOURCE] = value

        for warning in result.warnings[:3]:
            self.report({"WARNING"}, warning)
        if len(result.warnings) > 3:
            self.report(
                {"WARNING"}, f"...and {len(result.warnings) - 3} more mesh warnings"
            )
        self.report(
            {"INFO"},
            f"{load.model.name}: {len(result.joint_bones)} joints, "
            f"{len(result.mesh_objects)} meshes, TCP at '{result.tcp_link}'",
        )
        context.scene.kinema.last_import = load.model.name

        self._teardown(context)
        return {"FINISHED"}

    def cancel(self, context: bpy.types.Context) -> None:
        """Blender cancelled us from outside -- a file load, or quitting.

        Without this the timer and the job lock would leak, and the next import
        would be refused for the rest of the session.
        """
        self._abort(context)

    def _abort(self, context: bpy.types.Context) -> set[str]:
        # Once torn down, nothing here may touch the scene again. A finished
        # import calls _teardown from _finish and keeps its rig; a cancel()
        # arriving afterwards -- Blender closing the window, or loading a file
        # -- would otherwise delete that finished rig and free datablocks that
        # are already gone. Guarded here rather than in cancel() so the rule
        # holds for every caller, including the destructive step below.
        if self._torn_down:
            return {"CANCELLED"}

        self._cancel.set()
        if self._steps is not None:
            self._steps.close()
        if self._result is not None:
            builder.discard_rig(self._result)
        self._teardown(context)
        return {"CANCELLED"}

    def _teardown(self, context: bpy.types.Context) -> None:
        # Releasing the job lock twice would let two imports run at once.
        if self._torn_down:
            return
        self._torn_down = True

        window_manager = context.window_manager
        if self._timer is not None:
            window_manager.event_timer_remove(self._timer)
            self._timer = None
        window_manager.progress_end()
        self._set_status(context, None)
        if _JOB_LOCK.locked():
            _JOB_LOCK.release()

    @staticmethod
    def _set_status(context: bpy.types.Context, text: str | None) -> None:
        workspace = getattr(context, "workspace", None)
        if workspace is not None:
            workspace.status_text_set(text)

    # ----------------------------------------------------------- background
    def _run_synchronously(self, context: bpy.types.Context) -> set[str]:
        """Whole import in one call, for background Blender and headless tests."""
        result = loader.load_file(self.filepath)
        if result.error:
            self.report({"ERROR"}, result.error)
            return {"CANCELLED"}

        rig = builder.build_rig(result.model, _options_from(self))
        if result.source is not None and rig.armature_object is not None:
            kind, value = result.source
            rig.armature_object[builder.PROP_SOURCE_KIND] = kind
            rig.armature_object[builder.PROP_SOURCE] = value
        for warning in rig.warnings[:3]:
            self.report({"WARNING"}, warning)
        self.report(
            {"INFO"},
            f"{result.model.name}: {len(rig.joint_bones)} joints, "
            f"{len(rig.mesh_objects)} meshes, TCP at '{rig.tcp_link}'",
        )
        context.scene.kinema.last_import = result.model.name
        return {"FINISHED"}


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------
def _catalog_items(self, context):
    """Enum items for the catalog search popup.

    Built fresh on each invoke: the list is small, and caching Blender enum
    item strings is a well-known way to get garbage-collected labels.
    """
    props = getattr(context.scene, "kinema", None)
    entries = catalog.search(
        supported_only=True,
        include_curated_out=bool(getattr(props, "catalog_show_all", False)),
    )
    if not entries:
        return [("", "Catalog unavailable", "robots.json could not be read")]
    return [
        (
            entry.key,
            f"{entry.label} — {entry.status}" if entry.status else entry.label,
            f"{entry.key} · {entry.format_label} — {entry.note or entry.description}",
        )
        for entry in entries
    ]


def _hand_off(context: bpy.types.Context, **properties) -> None:
    """Invoke the modal worker from a zero-delay timer.

    ``invoke_search_popup`` and ``ImportHelper`` both route through
    ``execute()``, which cannot legally return ``{'RUNNING_MODAL'}``. Running
    the worker from a one-shot timer instead puts it on the main loop after the
    popup or file browser has gone, which is also the only way its modal handler
    attaches to a region that will still exist.

    The window has to be captured here and overridden there: a timer callback
    has no reliable window in ``bpy.context``, and ``modal_handler_add`` needs
    one.
    """
    if bpy.app.background or context.window is None:
        # No event loop to run the timer, so there is nothing to defer to.
        # Calling without INVOKE_DEFAULT runs the worker's execute(), which
        # does the whole import synchronously.
        bpy.ops.kinema.build_robot(**properties)
        return

    window = context.window

    def launch():
        try:
            with bpy.context.temp_override(window=window):
                bpy.ops.kinema.build_robot("INVOKE_DEFAULT", **properties)
        except RuntimeError as exc:
            print(f"Kinema: could not start import: {exc}")
        return None  # one-shot

    bpy.app.timers.register(launch, first_interval=0.0)


class KINEMA_OT_browse_catalog(Operator):
    """Look a robot up and hand back the command to fetch it.

    Kinema does not download robot descriptions. What this offers instead is
    the part that is actually hard to find by hand: which repository holds a
    given robot, at which commit, and which of its files -- out of up to 2466 --
    is the one to open.
    """

    bl_idname = "kinema.browse_catalog"
    bl_label = "Browse Robot Catalog"
    bl_description = (
        "Search 186 robot descriptions and copy the git command to download one"
    )
    # No UNDO or REGISTER: nothing in the scene changes.
    bl_options = set()
    # Makes invoke_search_popup show a fuzzy-searchable list of every robot.
    bl_property = "robot_key"

    robot_key: EnumProperty(name="Robot", items=_catalog_items)

    def invoke(self, context: bpy.types.Context, event) -> set[str]:
        context.window_manager.invoke_search_popup(self)
        return {"RUNNING_MODAL"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        key = self.robot_key
        entry = catalog.get(key)
        if entry is None:
            self.report({"ERROR"}, f"Unknown robot '{key}'")
            return {"CANCELLED"}

        context.window_manager.clipboard = entry.clone_command
        context.scene.kinema.catalog_pick = key
        self.report(
            {"INFO"},
            f"{entry.label}: clone command copied — then open "
            f"{entry.clone_dir}/{entry.file_path}",
        )
        return {"FINISHED"}


class KINEMA_OT_open_catalog_repo(Operator):
    """Open the picked robot's repository in a browser."""

    bl_idname = "kinema.open_catalog_repo"
    bl_label = "Open Repository"
    bl_description = "Open this robot's source repository in your web browser"
    bl_options = set()

    def execute(self, context: bpy.types.Context) -> set[str]:
        entry = catalog.get(context.scene.kinema.catalog_pick)
        if entry is None:
            self.report({"ERROR"}, "No robot picked")
            return {"CANCELLED"}
        # Strip the .git suffix: that URL is for git, not for a browser.
        url = entry.repo_url.removesuffix(".git")
        bpy.ops.wm.url_open(url=url)
        return {"FINISHED"}


class KINEMA_OT_import_urdf(Operator, ImportHelper):
    bl_idname = "kinema.import_urdf"
    bl_label = "Import URDF"
    bl_description = "Build a Kinema rig from a local URDF or xacro file"
    # See KINEMA_OT_build_robot: the worker owns the undo step, and REGISTER
    # here would give the file browser a redo panel that re-imports everything.
    bl_options = set()

    filename_ext = ".urdf"
    filter_glob: StringProperty(
        default="*.urdf;*.xacro;*.xml", options={"HIDDEN"}
    )
    __annotations__.update(import_settings())

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        layout.use_property_split = True
        column = layout.column()
        for name in SETTING_NAMES:
            column.prop(self, name)

    def execute(self, context: bpy.types.Context) -> set[str]:
        path = Path(self.filepath)
        if not path.is_file():
            self.report({"ERROR"}, f"No such file: {path}")
            return {"CANCELLED"}

        _hand_off(context, filepath=str(path), **setting_values(self))
        return {"FINISHED"}


def menu_draw(self, context: bpy.types.Context) -> None:
    self.layout.operator(KINEMA_OT_import_urdf.bl_idname, text="Robot URDF (.urdf/.xacro)")


classes = (
    KINEMA_OT_build_robot,
    KINEMA_OT_browse_catalog,
    KINEMA_OT_open_catalog_repo,
    KINEMA_OT_import_urdf,
)


def register_props() -> None:
    bpy.types.TOPBAR_MT_file_import.append(menu_draw)


def unregister_props() -> None:
    bpy.types.TOPBAR_MT_file_import.remove(menu_draw)
