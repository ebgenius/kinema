"""File > Import > COLLADA (.dae) operator.

Blender 5.0 removed its own COLLADA importer, so this entry restores the menu
item. It is useful on its own -- independently of anything robotics-specific --
which is why it is a first-class operator rather than an internal helper of the
URDF importer.
"""

from __future__ import annotations

import bpy
from bpy.props import BoolProperty, CollectionProperty, StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper

from ..io import dae


class KINEMA_OT_import_dae(Operator, ImportHelper):
    bl_idname = "kinema.import_dae"
    bl_label = "Import COLLADA"
    bl_description = "Import a COLLADA .dae mesh (restores support removed in Blender 5.0)"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".dae"
    filter_glob: StringProperty(default="*.dae", options={"HIDDEN"})
    # ImportHelper gives a single path; `files` adds multi-select support.
    files: CollectionProperty(type=bpy.types.OperatorFileListElement, options={"HIDDEN"})
    directory: StringProperty(subtype="DIR_PATH", options={"HIDDEN"})

    apply_unit_scale: BoolProperty(
        name="Apply Unit Scale",
        description=(
            "Honour the file's <unit meter> tag. CAD exporters often write "
            "millimetres; without this the mesh arrives 1000x too large"
        ),
        default=True,
    )
    apply_up_axis: BoolProperty(
        name="Apply Up Axis",
        description=(
            "Rotate the file's up axis to Blender's Z-up. Disable only for files "
            "that declare Y-up but are authored Z-up"
        ),
        default=True,
    )

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        layout.use_property_split = True
        column = layout.column(heading="Transform")
        column.prop(self, "apply_unit_scale")
        column.prop(self, "apply_up_axis")

    def execute(self, context: bpy.types.Context) -> set[str]:
        from pathlib import Path

        directory = Path(self.directory) if self.directory else Path(self.filepath).parent
        names = [f.name for f in self.files if f.name] or [Path(self.filepath).name]

        imported = 0
        warnings: list[str] = []
        for name in names:
            try:
                result = dae.import_dae(
                    directory / name,
                    apply_unit_scale=self.apply_unit_scale,
                    apply_up_axis=self.apply_up_axis,
                )
            except dae.DaeImportError as exc:
                self.report({"ERROR"}, str(exc))
                return {"CANCELLED"}
            imported += len(result.objects)
            warnings += result.warnings

        for warning in warnings[:3]:
            self.report({"WARNING"}, warning)

        if not imported:
            self.report({"WARNING"}, "No geometry found")
            return {"CANCELLED"}

        self.report(
            {"INFO"},
            f"Imported {imported} object{'s' if imported != 1 else ''} "
            f"from {len(names)} file{'s' if len(names) != 1 else ''}",
        )
        return {"FINISHED"}


def menu_draw(self, context: bpy.types.Context) -> None:
    self.layout.operator(KINEMA_OT_import_dae.bl_idname, text="COLLADA (.dae)")


classes = (KINEMA_OT_import_dae,)


def register_props() -> None:
    bpy.types.TOPBAR_MT_file_import.append(menu_draw)


def unregister_props() -> None:
    bpy.types.TOPBAR_MT_file_import.remove(menu_draw)
