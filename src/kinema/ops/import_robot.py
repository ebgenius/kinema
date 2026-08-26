"""Operators that turn a robot description into a Kinema rig.

Two entry points, because they answer two different questions:

* **Import from Catalog** -- "I want a UR5e." Picks from the 186-robot
  robot_descriptions catalog and downloads on demand.
* **Import URDF File** -- "I want *my* robot." Reads a local URDF, xacro or
  MJCF, resolving ``package://`` references by searching the file's own tree.

Both formats end at the same place: a RobotModel handed to the rig builder.
"""

from __future__ import annotations

from pathlib import Path

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper

from ..catalog import index as catalog
from ..rig import builder, kinematics

_URDF_SUFFIXES = {".urdf", ".xacro", ".xml"}


def _looks_like_mjcf(path: Path) -> bool:
    """True if the file's root element is <mujoco>.

    MJCF and URDF both commonly use .xml, so the extension decides nothing.
    Only the opening bytes are read -- some MJCF scenes are large.
    """
    if path.suffix.lower() in (".urdf", ".xacro"):
        return False
    try:
        with open(path, "rb") as handle:
            head = handle.read(4096).decode("utf-8", "ignore")
    except OSError:
        return False
    return "<mujoco" in head


def _online_ready(operator: Operator) -> bool:
    """Blender's offline mode is a user setting the guidelines require honouring."""
    if bpy.app.online_access:
        return True
    operator.report(
        {"ERROR"},
        "Blender is in offline mode; enable Preferences > System > Allow Online Access "
        "to download robot descriptions",
    )
    return False


def _build(operator: Operator, urdf, resolver, options, source=None) -> set[str]:
    """Shared tail: URDF -> model -> rig, with errors reported to the UI."""
    try:
        model = kinematics.model_from_urdf(urdf, mesh_resolver=resolver)
    except kinematics.UnsupportedJointError as exc:
        operator.report({"ERROR"}, str(exc))
        return {"CANCELLED"}
    except Exception as exc:  # noqa: BLE001
        operator.report({"ERROR"}, f"Could not read robot: {exc}")
        return {"CANCELLED"}

    if not model.actuated_joints:
        operator.report({"WARNING"}, f"'{model.name}' has no movable joints")

    result = builder.build_rig(model, options)

    # Record where this came from: the solver reloads the description later to
    # build PyRoki's robot model, and a saved .blend must still know.
    if source is not None and result.armature_object is not None:
        kind, value = source
        result.armature_object[builder.PROP_SOURCE_KIND] = kind
        result.armature_object[builder.PROP_SOURCE] = value

    for warning in result.warnings[:3]:
        operator.report({"WARNING"}, warning)
    if len(result.warnings) > 3:
        operator.report({"WARNING"}, f"...and {len(result.warnings) - 3} more mesh warnings")

    operator.report(
        {"INFO"},
        f"{model.name}: {len(result.joint_bones)} joints, "
        f"{len(result.mesh_objects)} meshes, TCP at '{result.tcp_link}'",
    )
    bpy.context.scene.kinema.last_import = model.name
    return {"FINISHED"}


def _build_mjcf(operator, context, path, options, source) -> set[str]:
    """Shared tail for MJCF: parse -> model -> rig."""
    from ..io import mjcf

    window = context.window
    window.cursor_set("WAIT")
    try:
        model = mjcf.model_from_mjcf(path)
    except mjcf.MjcfError as exc:
        operator.report({"ERROR"}, str(exc))
        return {"CANCELLED"}
    except Exception as exc:  # noqa: BLE001
        operator.report({"ERROR"}, f"Could not parse {Path(path).name}: {exc}")
        return {"CANCELLED"}
    finally:
        window.cursor_set("DEFAULT")

    result = builder.build_rig(model, options)
    if result.armature_object is not None:
        kind, value = source
        result.armature_object[builder.PROP_SOURCE_KIND] = kind
        result.armature_object[builder.PROP_SOURCE] = value
    for warning in result.warnings[:3]:
        operator.report({"WARNING"}, warning)
    operator.report(
        {"INFO"},
        f"{model.name}: {len(result.joint_bones)} joints, "
        f"{len(result.mesh_objects)} meshes, TCP at '{result.tcp_link}'",
    )
    context.scene.kinema.last_import = model.name
    return {"FINISHED"}


def import_settings() -> dict:
    """Property annotations shared by both import operators.

    Blender reads properties from a class's *own* ``__annotations__``; it does
    not walk base classes. So a mixin can share methods but not properties, and
    each operator merges this dict into its own annotations instead.
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


class KinemaImportSettings:
    """Behaviour shared by both import operators (methods only -- see above)."""

    def _options(self) -> builder.RigBuildOptions:
        return builder.RigBuildOptions(
            bone_length=self.bone_length or None,
            enforce_limits=self.enforce_limits,
            import_visuals=self.import_visuals,
            create_tcp=self.create_tcp,
        )

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        layout.use_property_split = True
        column = layout.column()
        column.prop(self, "import_visuals")
        column.prop(self, "create_tcp")
        column.prop(self, "enforce_limits")
        column.prop(self, "bone_length")


def _catalog_items(self, context):
    """Enum items for the catalog search popup.

    Built fresh on each invoke: the list is small, and caching Blender enum
    item strings is a well-known way to get garbage-collected labels.
    """
    entries = catalog.search(supported_only=True)
    if not entries:
        return [("", "Catalog unavailable", "robot_descriptions could not be loaded")]
    return [
        (
            entry.key,
            entry.label,
            f"{entry.key} · {entry.format_label} — {entry.description}",
        )
        for entry in entries
    ]


class KINEMA_OT_import_catalog(Operator, KinemaImportSettings):
    bl_idname = "kinema.import_catalog"
    bl_label = "Import Robot from Catalog"
    bl_description = "Pick a robot from the robot_descriptions catalog and build a rig"
    bl_options = {"REGISTER", "UNDO"}
    # Makes invoke_search_popup show a fuzzy-searchable list of every robot.
    bl_property = "robot_key"

    robot_key: EnumProperty(name="Robot", items=_catalog_items)
    __annotations__.update(import_settings())

    def invoke(self, context: bpy.types.Context, event) -> set[str]:
        context.window_manager.invoke_search_popup(self)
        return {"RUNNING_MODAL"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        key = self.robot_key
        entry = catalog.get(key)
        if entry is None:
            self.report({"ERROR"}, f"Unknown robot '{key}'")
            return {"CANCELLED"}

        # Only hit the network when the description is not already cached.
        from ..catalog import fetch

        cached = (fetch.cache_root() / key).exists()
        if not cached and not _online_ready(self):
            return {"CANCELLED"}

        window = context.window
        window.cursor_set("WAIT")
        mjcf_file = None
        try:
            if entry.has_urdf:
                urdf = catalog.load_urdf(key)
            else:
                mjcf_file = catalog.mjcf_path(key)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Could not load '{key}': {exc}")
            return {"CANCELLED"}
        finally:
            window.cursor_set("DEFAULT")

        if mjcf_file is not None:
            return _build_mjcf(
                self, context, mjcf_file, self._options(), ("catalog-mjcf", key)
            )

        # yourdfpy already knows how to resolve this description's own meshes.
        resolver = getattr(urdf, "_filename_handler", None)
        return _build(self, urdf, resolver, self._options(), source=("catalog", key))


class KINEMA_OT_import_urdf(Operator, ImportHelper, KinemaImportSettings):
    bl_idname = "kinema.import_urdf"
    bl_label = "Import URDF"
    bl_description = "Build a Kinema rig from a local URDF or xacro file"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".urdf"
    filter_glob: StringProperty(
        default="*.urdf;*.xacro;*.xml", options={"HIDDEN"}
    )
    __annotations__.update(import_settings())

    def execute(self, context: bpy.types.Context) -> set[str]:
        path = Path(self.filepath)
        if not path.is_file():
            self.report({"ERROR"}, f"No such file: {path}")
            return {"CANCELLED"}

        # An .xml here is almost always MJCF; a URDF would be .urdf. Sniff the
        # root tag rather than trusting the extension either way.
        if _looks_like_mjcf(path):
            return _build_mjcf(
                self, context, path, self._options(), ("mjcf", str(path))
            )

        try:
            import yourdfpy
        except ImportError:
            self.report({"ERROR"}, "yourdfpy is unavailable; check Kinema's dependencies")
            return {"CANCELLED"}

        from ..io.resolve import make_mesh_resolver

        resolver = make_mesh_resolver(path)

        window = context.window
        window.cursor_set("WAIT")
        try:
            if path.suffix.lower() == ".xacro" or path.name.endswith(".urdf.xacro"):
                urdf = self._load_xacro(path, resolver)
            else:
                urdf = yourdfpy.URDF.load(
                    str(path),
                    build_scene_graph=True,
                    load_meshes=False,
                    filename_handler=lambda name: resolver(name),
                )
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Could not parse {path.name}: {exc}")
            return {"CANCELLED"}
        finally:
            window.cursor_set("DEFAULT")

        return _build(self, urdf, resolver, self._options(), source=("file", str(path)))

    @staticmethod
    def _load_xacro(path: Path, resolver):
        """Render a xacro to URDF first; many ROS descriptions ship only xacro."""
        import yourdfpy
        from xacrodoc import XacroDoc

        doc = XacroDoc.from_file(str(path), resolve_packages=True)
        with doc.temp_urdf_file_path() as urdf_path:
            return yourdfpy.URDF.load(
                urdf_path,
                build_scene_graph=True,
                load_meshes=False,
                filename_handler=lambda name: resolver(name),
            )


def menu_draw(self, context: bpy.types.Context) -> None:
    self.layout.operator(KINEMA_OT_import_urdf.bl_idname, text="Robot URDF (.urdf/.xacro)")


classes = (KINEMA_OT_import_catalog, KINEMA_OT_import_urdf)


def register_props() -> None:
    bpy.types.TOPBAR_MT_file_import.append(menu_draw)


def unregister_props() -> None:
    bpy.types.TOPBAR_MT_file_import.remove(menu_draw)
