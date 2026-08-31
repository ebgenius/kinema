"""Hang objects and collections off a rig's bones.

Dressing a robot -- a gripper on the flange, a cable harness down the forearm,
a tool on the wrist -- is bone parenting, which Blender already does well. Two
things make doing it by hand tedious enough to be worth an operator.

**Bone parenting measures from the bone's tail.** Ctrl+P > Bone puts the child
at the far end of the bone, so every offset the user then dials in is measured
from a frame that has nothing to do with the joint. Kinema cancels that with
``matrix_parent_inverse``, which makes the bone's *head* the parent frame. The
attachment's own location/rotation/scale then simply *is* the offset from the
joint -- editable in the sidebar or with G/R/S in the viewport, and keyframable
like any object channel. Nothing extra is stored, so nothing can drift out of
sync.

**The same model usually goes on several bones.** So what lands in the scene is
a linked copy: ``source.copy()`` shares its mesh and materials, and a collection
arrives as a collection-instance Empty, which is a linked instance by
construction. Fix the harness once and every link wearing it updates.

Two ways to take an attachment off, and they differ deliberately. *Replacing* a
bone's pick deletes the copy it supersedes -- it is a linked copy Kinema made,
holding no user data but a placement. *Detaching* with the X button unparents
and leaves the object in the scene, because that is an explicit act on
something the user may well want to keep.
"""

from __future__ import annotations

import bpy
from bpy.props import BoolProperty, EnumProperty, PointerProperty, StringProperty
from bpy.types import Operator
from mathutils import Matrix

from ..rig import builder
from ..ui.panel import active_rig

#: Guards the picker update callbacks against the writes they themselves cause,
#: the same shape as handlers._solving.
_attaching = False


# --------------------------------------------------------------------------
# attach / detach
# --------------------------------------------------------------------------
def attach(rig, bone_name: str, source) -> bpy.types.Object | None:
    """Put a linked copy of ``source`` on ``bone_name``. Returns the copy.

    ``source`` may be an Object or a Collection. Whatever the bone was already
    wearing is deleted -- one slot per bone, because a collection already
    covers "several things at once" without needing a nested list in the UI.
    """
    pose_bone = rig.pose.bones.get(bone_name)
    if pose_bone is None or source is None:
        return None

    superseded = builder.bone_attachment(rig, bone_name)

    if isinstance(source, bpy.types.Collection):
        copy = bpy.data.objects.new(f"{source.name}.{bone_name}", None)
        copy.instance_type = "COLLECTION"
        copy.instance_collection = source
        copy.empty_display_size = pose_bone.bone.length or 0.05
    else:
        # copy() and not a data copy: sharing .data is the whole point.
        copy = source.copy()
        copy.name = f"{source.name}.{bone_name}"
        # Everything that could drive the copy's transform has to go, because
        # here that transform *is* the offset from the bone -- an attachment
        # still being pulled around by a Copy Location or a driver is not
        # sitting at the offset the panel claims it is.
        copy.animation_data_clear()
        copy.constraints.clear()

    _target_collection(rig).objects.link(copy)

    copy[builder.PROP_ATTACHMENT] = bone_name
    copy[builder.PROP_ATTACH_SOURCE] = source.name

    copy.parent = rig
    copy.parent_type = "BONE"
    copy.parent_bone = bone_name
    # Blender parents to the bone's TAIL. Undo that translation so the head
    # frame is the parent frame, which makes the copy's own transform the
    # offset from the joint:
    #     matrix_world = rig.matrix_world @ pose_bone.matrix @ copy.matrix_basis
    # Reading the rest length is safe because the rig builder scale-locks joint
    # bones, so the posed tail sits exactly bone.length from the head.
    copy.matrix_parent_inverse = Matrix.Translation(
        (0.0, -(pose_bone.bone.length or 0.0), 0.0)
    )
    zero_offset(copy)

    # Last, so a failure above leaves the old attachment in place rather than
    # removing it and then not replacing it.
    if superseded is not None:
        bpy.data.objects.remove(superseded, do_unlink=True)
    return copy


def zero_offset(obj) -> None:
    """Sit ``obj`` exactly on its parent frame, whatever channels it inherited.

    Not simply ``rotation_euler = (0, 0, 0)``. A copy keeps its source's
    ``rotation_mode``, so zeroing the Euler channel leaves a quaternion or
    axis-angle source rotated -- and delta transforms are copied too and feed
    ``matrix_basis`` on top of everything else. Either one puts a "zero" offset
    somewhere other than the bone head.

    Clearing the deltas first is what makes the basis assignment sufficient:
    ``matrix_basis`` reads deltas but writing it only touches the ordinary
    channels, so a surviving delta would reappear immediately.
    """
    obj.delta_location = (0.0, 0.0, 0.0)
    obj.delta_rotation_euler = (0.0, 0.0, 0.0)
    obj.delta_rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
    obj.delta_scale = (1.0, 1.0, 1.0)
    # Assigning the basis zeroes whichever rotation channel is actually active.
    obj.matrix_basis = Matrix.Identity(4)


def detach(attachment) -> None:
    """Unparent ``attachment``, leaving it where it appears on screen."""
    if attachment is None:
        return
    # Read the world matrix before breaking the parent, then write it back:
    # clearing the parent alone would snap the object to its raw local
    # transform, which for an un-offset attachment is the world origin.
    world = attachment.matrix_world.copy()
    attachment.parent = None
    attachment.parent_type = "OBJECT"
    attachment.parent_bone = ""
    attachment.matrix_parent_inverse = Matrix.Identity(4)
    attachment.matrix_world = world
    for key in (builder.PROP_ATTACHMENT, builder.PROP_ATTACH_SOURCE):
        if key in attachment:
            del attachment[key]


def _target_collection(rig) -> bpy.types.Collection:
    """Where a new attachment is linked: beside the rig, else the scene root."""
    collections = rig.users_collection
    return collections[0] if collections else bpy.context.scene.collection


# --------------------------------------------------------------------------
# the per-bone pickers
# --------------------------------------------------------------------------
def _can_attach(pose_bone, obj) -> bool:
    """Reject sources that would make the rig its own ancestor.

    Parenting a rig -- or anything already hanging off it, its own link meshes
    included -- to one of its bones is a dependency cycle, which Blender
    reports as a broken depsgraph rather than refusing outright. Cheaper to
    keep it out of the picker than to explain it afterwards.
    """
    rig = pose_bone.id_data
    node = obj
    while node is not None:
        # == and not `is`: RNA hands out fresh Python wrappers, and equality is
        # what compares the underlying data.
        if node == rig:
            return False
        node = node.parent
    return True


def _can_instance(pose_bone, collection) -> bool:
    """Reject a collection that contains the rig -- the same cycle, one level up."""
    rig = pose_bone.id_data
    return rig.name not in collection.all_objects


def _picked_source(pose_bone):
    if pose_bone.kinema_attach_type == "COLLECTION":
        return pose_bone.kinema_attach_collection
    return pose_bone.kinema_attach_object


def _on_source_picked(pose_bone, context) -> None:
    """Attach (or detach) when a row's picker changes.

    The work is handed to a zero-delay timer rather than done here. Creating
    and deleting datablocks from inside a property update callback is the thing
    to avoid -- the callback can fire while Blender is mid-draw -- and
    ops/import_robot.py already uses this same one-shot-timer escape for the
    same class of reason.

    Only names cross into the timer, never the datablocks themselves: an undo
    between the pick and the tick would leave a captured reference dangling.
    """
    if _attaching:
        return
    rig = pose_bone.id_data
    rig_name = rig.name
    bone_name = pose_bone.name
    source = _picked_source(pose_bone)
    source_name = source.name if source is not None else None
    is_collection = pose_bone.kinema_attach_type == "COLLECTION"

    def apply() -> None:
        _apply_pick(rig_name, bone_name, source_name, is_collection)
        return None  # unregister the timer

    bpy.app.timers.register(apply, first_interval=0.0)


def _apply_pick(
    rig_name: str, bone_name: str, source_name: str | None, is_collection: bool
) -> None:
    """Do the work the picker asked for, one tick later.

    Routed through operators rather than calling :func:`attach` directly, so
    the change lands in the undo stack. Blender pushes an undo step for the
    property edit as soon as the button is released -- before this tick runs --
    so work done here outside an operator would survive the Ctrl+Z that undoes
    the pick, leaving an attachment no row admits to.

    The scene can also have moved on in between -- an undo, a file load, a
    deleted rig -- so nothing here assumes anything still exists.
    """
    rig = bpy.data.objects.get(rig_name)
    if not builder.is_kinema_rig(rig) or bone_name not in rig.pose.bones:
        return

    if source_name is None:
        # Cleared, not replaced. Deleted rather than orphaned: an empty picker
        # should leave an empty bone, and the X button is the way to keep the
        # object.
        if builder.bone_attachment(rig, bone_name) is not None:
            bpy.ops.kinema.detach_from_bone(
                rig=rig_name, bone=bone_name, keep=False
            )
        return

    library = bpy.data.collections if is_collection else bpy.data.objects
    if library.get(source_name) is None:
        return
    bpy.ops.kinema.attach_to_bone(
        rig=rig_name,
        bone=bone_name,
        source=source_name,
        is_collection=is_collection,
    )


def set_picker(pose_bone, source) -> None:
    """Point a row's pickers at ``source`` (or clear them) without re-attaching."""
    global _attaching
    _attaching = True
    try:
        if isinstance(source, bpy.types.Collection):
            pose_bone.kinema_attach_collection = source
            pose_bone.kinema_attach_object = None
        else:
            pose_bone.kinema_attach_object = source
            if source is None:
                pose_bone.kinema_attach_collection = None
    finally:
        _attaching = False


# --------------------------------------------------------------------------
# operators
# --------------------------------------------------------------------------
class KinemaAttachOperator(Operator):
    """Base for the attachment operators.

    ``rig`` is a name rather than a context lookup because these are also
    called from a timer, where the active object is whatever the user last
    clicked and need not be the rig the pick came from. It falls back to the
    context for the buttons in the panel, which have no reason to spell it out.
    """

    bl_options = {"REGISTER", "UNDO"}

    rig: StringProperty(name="Rig", default="", options={"SKIP_SAVE"})
    bone: StringProperty(name="Bone", default="", options={"SKIP_SAVE"})

    def target_rig(self, context: bpy.types.Context):
        if self.rig:
            named = bpy.data.objects.get(self.rig)
            return named if builder.is_kinema_rig(named) else None
        return active_rig(context)


class KINEMA_OT_attach_to_bone(KinemaAttachOperator):
    bl_idname = "kinema.attach_to_bone"
    bl_label = "Attach to Bone"
    bl_description = "Put a linked copy of an object or collection on this bone"

    source: StringProperty(name="Source", default="", options={"SKIP_SAVE"})
    is_collection: BoolProperty(name="Collection", default=False, options={"SKIP_SAVE"})

    def execute(self, context: bpy.types.Context) -> set[str]:
        rig = self.target_rig(context)
        if rig is None:
            self.report({"ERROR"}, "No Kinema rig to attach to")
            return {"CANCELLED"}
        if self.bone not in rig.pose.bones:
            self.report({"ERROR"}, f"'{rig.name}' has no bone named '{self.bone}'")
            return {"CANCELLED"}

        library = bpy.data.collections if self.is_collection else bpy.data.objects
        source = library.get(self.source)
        if source is None:
            self.report({"ERROR"}, f"'{self.source}' is not in this file")
            return {"CANCELLED"}

        copy = attach(rig, self.bone, source)
        if copy is None:
            self.report({"ERROR"}, f"Could not attach '{self.source}'")
            return {"CANCELLED"}

        # Keep the row honest when the operator is driven from a script rather
        # than from the picker that would already be showing this.
        set_picker(rig.pose.bones[self.bone], source)
        context.view_layer.update()
        self.report({"INFO"}, f"'{source.name}' attached to '{self.bone}'")
        return {"FINISHED"}


class KINEMA_OT_detach_from_bone(KinemaAttachOperator):
    bl_idname = "kinema.detach_from_bone"
    bl_label = "Detach"
    bl_description = (
        "Unparent this bone's attachment, leaving it in the scene exactly "
        "where it appears"
    )

    keep: BoolProperty(
        name="Keep the Object",
        description=(
            "Leave the attachment in the scene, unparented. Off deletes it, "
            "which is what clearing a row's picker does"
        ),
        default=True,
        options={"SKIP_SAVE"},
    )

    def execute(self, context: bpy.types.Context) -> set[str]:
        rig = self.target_rig(context)
        if rig is None:
            self.report({"ERROR"}, "No Kinema rig selected")
            return {"CANCELLED"}
        attachment = builder.bone_attachment(rig, self.bone)
        if attachment is None:
            self.report({"WARNING"}, f"Nothing is attached to '{self.bone}'")
            return {"CANCELLED"}

        name = attachment.name
        if self.keep:
            detach(attachment)
        else:
            bpy.data.objects.remove(attachment, do_unlink=True)

        pose_bone = rig.pose.bones.get(self.bone)
        if pose_bone is not None:
            set_picker(pose_bone, None)

        message = (
            f"'{name}' detached; it is still in the scene"
            if self.keep
            else f"'{name}' removed from '{self.bone}'"
        )
        self.report({"INFO"}, message)
        return {"FINISHED"}


class KINEMA_OT_reset_attachment_offset(KinemaAttachOperator):
    bl_idname = "kinema.reset_attachment_offset"
    bl_label = "Reset Offset"
    bl_description = "Sit the attachment exactly on the bone, with no offset"

    def execute(self, context: bpy.types.Context) -> set[str]:
        rig = self.target_rig(context)
        attachment = builder.bone_attachment(rig, self.bone) if rig else None
        if attachment is None:
            self.report({"WARNING"}, f"Nothing is attached to '{self.bone}'")
            return {"CANCELLED"}

        zero_offset(attachment)
        context.view_layer.update()
        self.report({"INFO"}, f"'{attachment.name}' reset onto the bone")
        return {"FINISHED"}


class KINEMA_OT_select_attachment(KinemaAttachOperator):
    bl_idname = "kinema.select_attachment"
    bl_label = "Select Attachment"
    bl_description = "Make this bone's attachment the active object, ready to move"

    def execute(self, context: bpy.types.Context) -> set[str]:
        rig = self.target_rig(context)
        attachment = builder.bone_attachment(rig, self.bone) if rig else None
        if attachment is None:
            self.report({"WARNING"}, f"Nothing is attached to '{self.bone}'")
            return {"CANCELLED"}
        if context.object is not None and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        for other in context.view_layer.objects:
            other.select_set(False)
        attachment.select_set(True)
        context.view_layer.objects.active = attachment
        return {"FINISHED"}


classes = (
    KINEMA_OT_attach_to_bone,
    KINEMA_OT_detach_from_bone,
    KINEMA_OT_reset_attachment_offset,
    KINEMA_OT_select_attachment,
)


def register_props() -> None:
    # On PoseBone rather than on the rig, so the picker can be drawn straight
    # into a UIList row over rig.pose.bones with no parallel collection to keep
    # in step with the bones.
    bpy.types.PoseBone.kinema_attach_type = EnumProperty(
        name="Attach",
        description="Whether this bone takes an object or a whole collection",
        items=[
            ("OBJECT", "Object", "Attach a linked copy of an object", "OBJECT_DATA", 0),
            (
                "COLLECTION",
                "Collection",
                "Attach an instance of a collection",
                "OUTLINER_COLLECTION",
                1,
            ),
        ],
        default="OBJECT",
    )
    bpy.types.PoseBone.kinema_attach_object = PointerProperty(
        name="Attachment",
        description="Object to attach to this bone, as a linked copy",
        type=bpy.types.Object,
        poll=_can_attach,
        update=_on_source_picked,
    )
    bpy.types.PoseBone.kinema_attach_collection = PointerProperty(
        name="Attachment",
        description="Collection to instance on this bone",
        type=bpy.types.Collection,
        poll=_can_instance,
        update=_on_source_picked,
    )


def unregister_props() -> None:
    del bpy.types.PoseBone.kinema_attach_type
    del bpy.types.PoseBone.kinema_attach_object
    del bpy.types.PoseBone.kinema_attach_collection
