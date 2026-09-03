"""Operators that turn a robot description into a Kinema rig.

One import path -- **Import URDF File**: a local URDF, xacro or MJCF, with
``package://`` references resolved by searching the file's own tree.

Alongside it sits **Browse Robot Catalog**, which imports nothing. It searches
the offline catalogue (``catalog.index``) and hands back a ``git clone``
command plus the path of the description file inside that repository, for the
user to fetch by hand. Kinema does not download robots.

Both hand off to :class:`KINEMA_OT_build_robot`, which does the parse and the
rig build in one blocking call behind a wait cursor.

It did not always. The parse ran on a worker thread and the build was chunked
across modal timer ticks, so the viewport stayed alive and Esc could cancel a
long import. Blender's extension guidelines do not allow an extension to start
threads, and that machinery bought responsiveness and nothing else -- so it is
gone, and an import now costs a freeze of a second or two for an arm, longer
for a humanoid, where the time is one Blender mesh-importer call per visual.
"""

from __future__ import annotations

from pathlib import Path

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper

from ..catalog import index as catalog
from ..io import loader
from ..rig import builder


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


# --------------------------------------------------------------------------
# the builder
# --------------------------------------------------------------------------
class KINEMA_OT_build_robot(Operator):
    """Parse a robot description and build its rig.

    Not exposed in any menu: it is the engine behind the URDF file browser,
    which invokes it with the file already chosen.

    **This blocks.** The parse used to run on a worker thread and the build was
    chunked across modal timer ticks, which kept the viewport alive and let Esc
    cancel a long import. Blender's extension guidelines do not permit an
    extension to start threads, and responsiveness during the import was the
    only thing the thread bought -- so it is a wait cursor and a blocking call
    now, which is what Blender's own importers do. Expect a second or two for an
    arm and longer for a humanoid, where most of the time is one Blender
    mesh-importer call per visual.
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

    def execute(self, context: bpy.types.Context) -> set[str]:
        window_manager = context.window_manager
        # Blender's own "busy" signal: it puts up the wait cursor and takes it
        # down again. Nothing reports progress into it, because nothing can --
        # the UI does not redraw until this returns.
        window_manager.progress_begin(0, 1)
        try:
            return self._build(context)
        finally:
            window_manager.progress_end()

    def _build(self, context: bpy.types.Context) -> set[str]:
        result = loader.load_file(self.filepath)
        if result.error:
            self.report({"ERROR"}, result.error)
            return {"CANCELLED"}
        if not result.model.actuated_joints:
            self.report({"WARNING"}, f"'{result.model.name}' has no movable joints")

        # Driven by hand rather than through builder.build_rig(), only so that
        # `rig` still holds the partial result if a step raises -- otherwise a
        # failure half way leaves loose objects in the scene.
        rig = builder.RigBuildResult()
        steps = builder.build_rig_iter(result.model, _options_from(self), result=rig)
        try:
            while True:
                next(steps)
        except StopIteration as stop:
            rig = stop.value
        except Exception as exc:  # noqa: BLE001 - reported, never raised at the user
            steps.close()
            builder.discard_rig(rig)
            self.report({"ERROR"}, f"Could not build rig: {exc}")
            return {"CANCELLED"}

        if result.source is not None and rig.armature_object is not None:
            # Record where this came from: the solver reloads the description
            # later to build PyRoki's robot model, and a saved .blend must still
            # know.
            kind, value = result.source
            rig.armature_object[builder.PROP_SOURCE_KIND] = kind
            rig.armature_object[builder.PROP_SOURCE] = value

        for warning in rig.warnings[:3]:
            self.report({"WARNING"}, warning)
        if len(rig.warnings) > 3:
            self.report(
                {"WARNING"}, f"...and {len(rig.warnings) - 3} more mesh warnings"
            )
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
    show_all = bool(getattr(props, "catalog_show_all", False))
    # Both filters move together, or the toggle cannot keep its promise: an
    # entry with no resolved file path is *also* unsupported, so leaving
    # supported_only on would hide it however the toggle is set.
    entries = catalog.search(supported_only=not show_all, include_curated_out=show_all)
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
            {"INFO"}, f"{entry.label}: clone command copied — {entry.handoff_hint}"
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
    # See KINEMA_OT_build_robot: it owns the undo step, and REGISTER here would
    # give the file browser a redo panel that re-imports everything.
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

        # A plain call, not a deferred one: build_robot is no longer modal, so
        # there is nothing to escape execute()'s return-value rules for.
        return bpy.ops.kinema.build_robot(
            filepath=str(path), **setting_values(self)
        )


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
