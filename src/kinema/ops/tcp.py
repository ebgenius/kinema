"""Edit TCP: place the tool-centre point in a dialog, seeing where it goes first.

Three ways to say where:

* **Offset** -- a location and roll/pitch/yaw from the parent joint's link frame,
  the numbers Set TCP has always taken.
* **3D Cursor** -- wherever the cursor is. Snap it to a vertex of the tool mesh first
  (Shift+S), and the TCP lands on that vertex.
* **Object Origin** -- an empty or the tool mesh itself, say one attached to the
  flange.

For the last two, the dialog works out the offset those place it at and shows it,
so what gets stored is always an offset. Either can also give the orientation, or
leave it at the offset's.

A TCP offset is the tool's pose relative to its link, which the arm's motion does
not change. So it is measured from where the link stands *now*, posed, against the
cursor or object where they stand now -- which is what lets a TCP be picked off a
tool mesh on a robot that is not at rest, on a posed Root included.

While the dialog is open the viewport shows where the TCP would go, next to where it
is (``ui/tcp_preview.py``). Nothing moves until OK, and OK goes through Set TCP, so a
TCP on the same link keeps its compiled solver and the arm stays where it is.
"""

from __future__ import annotations

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatVectorProperty, StringProperty
from bpy.types import Operator
from mathutils import Euler, Matrix, Vector

from ..rig import builder
from ..solver import manager
from ..ui import tcp_preview
from ..ui.panel import active_rig
from . import deferral

SOURCE_ITEMS = [
    ("OFFSET", "Offset", "Type a location and rotation from the parent joint's link frame"),
    ("CURSOR", "3D Cursor", "Put the TCP where the 3D cursor is. Snap the cursor to a "
     "tool vertex first with Shift+S"),
    ("OBJECT", "Object Origin", "Put the TCP on an object's origin: an empty, or the tool mesh"),
]


def _np4(matrix) -> np.ndarray:
    return np.array([[matrix[r][c] for c in range(4)] for r in range(4)])


def _rigid(matrix: Matrix) -> Matrix:
    """``matrix`` without its scale: a TCP is a frame, and an offset holds no scale."""
    location, rotation, _ = matrix.decompose()
    return Matrix.Translation(location) @ rotation.to_matrix().to_4x4()


def posed_link(rig, parent: str) -> Matrix | None:
    """``parent``'s link frame where it stands now, in armature space, or None."""
    pose_bone = rig.pose.bones.get(parent)
    if pose_bone is None or builder.PROP_JOINT_NAME not in pose_bone.bone:
        return None
    rest_link = builder.link_frame_of(pose_bone.bone)
    if rest_link is None:
        return None
    correction = pose_bone.bone.matrix_local.inverted_safe() @ rest_link
    return pose_bone.matrix @ correction


def source_frame(context, rig, source: str, object_name: str) -> Matrix | None:
    """Where the cursor or object stands, in the rig's armature space, scale removed."""
    if source == "CURSOR":
        world = context.scene.cursor.matrix
    elif source == "OBJECT":
        obj = bpy.data.objects.get(object_name)
        if obj is None:
            return None
        world = obj.matrix_world
    else:
        return None
    return _rigid(rig.matrix_world.inverted_safe() @ world)


def resolve(link: Matrix, target: Matrix | None, offset, rotation, use_rotation: bool):
    """``(offset, rotation)`` that put the TCP on ``target``, from the ``link`` frame.

    With no target, the typed values. Without ``use_rotation`` only the location is
    taken from the target, and the orientation stays the typed one.
    """
    offset, rotation = tuple(offset), tuple(rotation)
    if target is None:
        return offset, rotation
    relative = link.inverted_safe() @ target
    location = tuple(relative.translation)
    if use_rotation:
        rotation = tuple(relative.to_euler("XYZ"))
    return location, rotation


def tool_frame(link: Matrix, offset, rotation) -> Matrix:
    """The TCP's frame for an offset from ``link``: Set TCP's own composition."""
    turn = Euler(Vector(rotation), "XYZ").to_matrix().to_4x4()
    return link @ Matrix.Translation(Vector(offset)) @ turn


# --------------------------------------------------------------------------
# remembered fields
# --------------------------------------------------------------------------
#: rig -> the dialog's fields as it last drew them, keyed by identity, as the axis
#: dialog's are: a click in the viewport to look at the preview closes the dialog,
#: and Blender keeps an operator's fields only when it finishes.
_drafts: dict[int | str, dict] = {}
_FIELDS = ("parent", "source", "offset", "rotation", "object_name", "use_rotation")


def _remember(operator, rig) -> None:
    _drafts[manager.rig_identity(rig)] = {
        name: (tuple(value) if hasattr(value, "__len__") and not isinstance(value, str)
               else value)
        for name, value in ((name, getattr(operator, name)) for name in _FIELDS)
    }


def forget() -> None:
    """Drop every draft: a new file shares no rig with the old one."""
    _drafts.clear()


def _refresh_preview(operator, context) -> None:
    """Redraw the preview from the fields as they change, a slider drag included.

    The dialog's draw() only runs when Blender rebuilds the popup, which it does not
    do mid-drag; a property's update callback does run on every step. Reads the
    fields and nothing else, and hands the preview the dialog it is tracking, in
    case ``operator`` here is not the same Python object as the dialog's.
    """
    dialog = tcp_preview._state.get("operator")
    rig = active_rig(context)
    if dialog is None or rig is None:
        return
    link, offset, rotation, problem = KINEMA_OT_edit_tcp.placement(operator, context, rig)
    tcp_preview.show(dialog, None if problem else tool_frame(link, offset, rotation), link)


# --------------------------------------------------------------------------
# the operator
# --------------------------------------------------------------------------
class KINEMA_OT_edit_tcp(Operator):
    """Place the TCP by offset, on the 3D cursor or on an object, previewing it first"""

    bl_idname = "kinema.edit_tcp"
    bl_label = "Edit TCP"
    bl_options = {"REGISTER", "UNDO"}

    parent: StringProperty(
        name="Parent Joint", update=_refresh_preview,
        description="The joint bone the TCP rides; its link frame is what offsets are from",
    )
    source: EnumProperty(
        name="Place By", items=SOURCE_ITEMS, default="OFFSET", update=_refresh_preview,
    )
    offset: FloatVectorProperty(
        name="Location", size=3, subtype="TRANSLATION", unit="LENGTH", update=_refresh_preview,
        description="From the parent joint's link frame",
    )
    rotation: FloatVectorProperty(
        name="Rotation", size=3, subtype="EULER", update=_refresh_preview,
        description="Roll, pitch and yaw about the link frame's fixed X, Y and Z",
    )
    object_name: StringProperty(
        name="Object", description="The object whose origin the TCP goes on",
        update=_refresh_preview,
    )
    use_rotation: BoolProperty(
        name="Use Its Rotation", default=True, update=_refresh_preview,
        description="Take the orientation from the cursor or object too, not only the location",
    )

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return active_rig(context) is not None

    def placement(self, context, rig) -> tuple[Matrix | None, tuple, tuple, str | None]:
        """``(link, offset, rotation, problem)`` for the fields as they stand."""
        link = posed_link(rig, self.parent)
        if link is None:
            if not self.parent:
                return None, (), (), "Pick the joint the TCP rides"
            return None, (), (), f"'{self.parent}' is not a joint bone"
        target = source_frame(context, rig, self.source, self.object_name)
        if self.source == "OBJECT" and target is None:
            return link, (), (), "Pick the object to put the TCP on"
        offset, rotation = resolve(link, target, self.offset, self.rotation, self.use_rotation)
        return link, offset, rotation, None

    def invoke(self, context, event):
        rig = active_rig(context)
        draft = _drafts.get(manager.rig_identity(rig))
        if draft:
            for name in _FIELDS:
                if name in draft:
                    setattr(self, name, draft[name])
        else:
            self.parent = rig.kinema_tcp_parent
            self.offset = tuple(rig.kinema_tcp_offset)
            self.rotation = tuple(rig.kinema_tcp_rpy)
        tcp_preview.start(self, rig)
        return context.window_manager.invoke_props_dialog(self, width=400)

    def cancel(self, context):
        tcp_preview.stop()

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        rig = active_rig(context)
        if rig is None:
            return
        _remember(self, rig)
        layout.prop_search(self, "parent", rig.pose, "bones", icon="BONE_DATA")
        layout.row().prop(self, "source", expand=True)

        if self.source == "OFFSET":
            column = layout.column(align=True)
            column.prop(self, "offset")
            column.prop(self, "rotation")
        else:
            if self.source == "OBJECT":
                layout.prop_search(self, "object_name", bpy.data, "objects")
            else:
                layout.label(text="Snap the cursor to a tool vertex with Shift+S", icon="INFO")
            layout.prop(self, "use_rotation")
            if not self.use_rotation:
                layout.prop(self, "rotation")

        link, offset, rotation, problem = self.placement(context, rig)
        tcp_preview.show(self, None if problem else tool_frame(link, offset, rotation), link)
        if problem:
            layout.label(text=problem, icon="ERROR")
            return
        if self.source != "OFFSET":
            box = layout.box().column(align=True)
            box.label(text="Stored as an offset from the link frame:", icon="EMPTY_AXIS")
            box.label(text="Location  " + "  ".join(f"{v:+.4f}" for v in offset))
            box.label(
                text="Rotation  " + "  ".join(f"{np.degrees(v):+.2f}°" for v in rotation)
            )

    def execute(self, context):
        tcp_preview.stop()
        rig = active_rig(context)
        link, offset, rotation, problem = self.placement(context, rig)
        if problem:
            self.report({"ERROR"}, problem)
            return {"CANCELLED"}
        # Kept typed from now on, so reopening shows the numbers the TCP stands at.
        self.source, self.offset, self.rotation = "OFFSET", offset, rotation
        _remember(self, rig)
        rig.kinema_tcp_offset = offset
        rig.kinema_tcp_rpy = rotation
        return bpy.ops.kinema.set_tcp(bone=self.parent)


# Its redo panel is an edit in progress: see ops/deferral.py.
deferral.adjustable("KINEMA_OT_edit_tcp")

classes = (KINEMA_OT_edit_tcp,)
