"""Pose-level operators: rest pose, keying, and moving the TCP."""

from __future__ import annotations

import bpy
from bpy.props import BoolProperty, StringProperty
from bpy.types import Operator
from mathutils import Euler, Matrix, Vector

from .. import handlers
from ..rig import builder
from ..solver import manager
from ..ui.panel import active_rig


class KinemaRigOperator(Operator):
    """Base for operators that need a Kinema rig in the context."""

    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return active_rig(context) is not None


class KINEMA_OT_reset_pose(KinemaRigOperator):
    bl_idname = "kinema.reset_pose"
    bl_label = "Rest Pose"
    bl_description = "Return every joint to zero, the robot's URDF rest configuration"

    def execute(self, context: bpy.types.Context) -> set[str]:
        rig = active_rig(context)
        for pose_bone in builder.joint_bones(rig):
            pose_bone.rotation_euler = (0.0, 0.0, 0.0)
            pose_bone.location = (0.0, 0.0, 0.0)
        context.view_layer.update()
        self.report({"INFO"}, "Joints reset to rest pose")
        return {"FINISHED"}


class KINEMA_OT_key_joints(KinemaRigOperator):
    bl_idname = "kinema.key_joints"
    bl_label = "Key All Joints"
    bl_description = "Insert a keyframe on every joint channel at the current frame"

    def execute(self, context: bpy.types.Context) -> set[str]:
        rig = active_rig(context)
        frame = context.scene.frame_current
        keyed = 0
        for pose_bone in builder.joint_bones(rig):
            is_prismatic = (
                pose_bone.bone.get(builder.PROP_JOINT_TYPE, "revolute") == "prismatic"
            )
            # Key only the channel the joint actually uses; keying all nine
            # would fill the dope sheet with channels that can never move.
            path, index = (
                ("location", 1) if is_prismatic else ("rotation_euler", 1)
            )
            if pose_bone.keyframe_insert(data_path=path, index=index, frame=frame):
                keyed += 1
        self.report({"INFO"}, f"Keyed {keyed} joint{'s' if keyed != 1 else ''} at frame {frame}")
        return {"FINISHED"}


class KINEMA_OT_reset_link_meshes(KinemaRigOperator):
    bl_idname = "kinema.reset_link_meshes"
    bl_label = "Reset Meshes"
    bl_description = (
        "Put every link mesh back where the importer placed it, undoing a "
        "stray grab. Objects you attached yourself are left alone"
    )

    def execute(self, context: bpy.types.Context) -> set[str]:
        rig = active_rig(context)
        meshes = builder.link_meshes(rig)
        if not meshes:
            # Either there are no visuals, or the rig predates the marker.
            # Guessing is not an option: the visual origin and mesh scale that
            # produced the placement are recorded nowhere else.
            self.report(
                {"WARNING"},
                "No link meshes to reset. A rig imported before Kinema recorded "
                "them has to be re-imported",
            )
            return {"CANCELLED"}

        moved = 0
        for obj in meshes:
            rest = builder.link_rest_matrix(obj)
            if rest is None:
                continue
            if _restore_basis(obj, rest):
                moved += 1

        context.view_layer.update()
        if not moved:
            self.report({"INFO"}, "Link meshes were already in place")
            return {"FINISHED"}
        self.report({"INFO"}, f"Reset {moved} link mesh{'es' if moved != 1 else ''}")
        return {"FINISHED"}


def _restore_basis(obj, rest) -> bool:
    """Put ``obj`` back on ``rest``. Returns True if it actually moved.

    The delta channels are cleared first for the same reason attachments clear
    them: ``matrix_basis`` reads deltas, but writing it only touches the
    ordinary channels, so a delta left behind would reappear immediately.
    """
    if _matrices_match(obj.matrix_basis, rest) and _deltas_are_clear(obj):
        return False
    obj.delta_location = (0.0, 0.0, 0.0)
    obj.delta_rotation_euler = (0.0, 0.0, 0.0)
    obj.delta_rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
    obj.delta_scale = (1.0, 1.0, 1.0)
    obj.matrix_basis = rest
    return True


def _matrices_match(a, b, tolerance: float = 1e-6) -> bool:
    return all(
        abs(a[row][col] - b[row][col]) < tolerance
        for row in range(4)
        for col in range(4)
    )


def _deltas_are_clear(obj) -> bool:
    return (
        tuple(obj.delta_location) == (0.0, 0.0, 0.0)
        and tuple(obj.delta_rotation_euler) == (0.0, 0.0, 0.0)
        and tuple(obj.delta_scale) == (1.0, 1.0, 1.0)
    )


class KINEMA_OT_set_tcp(KinemaRigOperator):
    bl_idname = "kinema.set_tcp"
    bl_label = "Set TCP"
    bl_description = (
        "Place the tool-centre-point marker on a joint bone, offset from that "
        "joint's link frame by the rig's tool offset"
    )

    bone: StringProperty(
        name="Bone",
        description="Joint bone to place the TCP on. Empty uses the active bone",
        default="",
        options={"SKIP_SAVE"},
    )

    def execute(self, context: bpy.types.Context) -> set[str]:
        rig = active_rig(context)
        source = self._source_bone(context, rig, self.bone)
        if source is None:
            self.report(
                {"WARNING"},
                "Pick a bone in the Tool Centre Point panel, or make one active "
                "in Pose or Edit mode",
            )
            return {"CANCELLED"}
        # Joint bones only. They are the ones carrying a link correction, so
        # they are the only ones this can place a *tool frame* on -- and it
        # keeps the TCP off the IK control, which is the likeliest thing to be
        # selected in Pose mode once IK exists and which would leave the marker
        # riding the goal it is supposed to define.
        if builder.PROP_JOINT_NAME not in source.bone:
            self.report({"WARNING"}, f"'{source.name}' is not a joint bone")
            return {"CANCELLED"}

        previous_mode = context.object.mode if context.object else "OBJECT"
        previous_active = context.view_layer.objects.active
        context.view_layer.objects.active = rig
        if rig.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        # Everything past this point runs with the rig made active and forced
        # into Object mode, so every exit has to put those back -- not just the
        # successful one. Cancelling used to leave the user in a different mode
        # on a different object, which is a worse outcome than the failure it
        # was reporting.
        try:
            return self._place(context, rig, source)
        finally:
            if previous_active is not None:
                context.view_layer.objects.active = previous_active
            if previous_mode != "OBJECT" and context.object is not None:
                try:
                    bpy.ops.object.mode_set(mode=previous_mode)
                except RuntimeError:
                    pass

    def _place(self, context: bpy.types.Context, rig, source) -> set[str]:
        # Computed before Edit mode, not inside it: Bone.matrix_local is not
        # live while the armature is open for editing, and the whole placement
        # hangs off it.
        target = self._tool_frame(rig, source)
        if target is None:
            self.report({"ERROR"}, f"'{source.name}' records no link frame")
            return {"CANCELLED"}
        length = source.bone.length or 0.05

        bpy.ops.object.mode_set(mode="EDIT")
        try:
            edit_bones = rig.data.edit_bones
            reference = edit_bones.get(source.name)
            if reference is None:
                self.report({"ERROR"}, f"'{source.name}' vanished from the armature")
                return {"CANCELLED"}

            tcp = edit_bones.get(builder.TCP_BONE) or edit_bones.new(builder.TCP_BONE)
            # The same three lines the importer uses, against the same frame --
            # which is what makes re-setting the TCP land where it started
            # rather than rolling it onto the bone's own axes.
            origin = target.translation
            tcp.head = origin
            tcp.tail = origin + target.col[2].xyz.normalized() * length * 0.8
            tcp.align_roll(target.col[0].xyz)  # align_roll takes the desired Z
            tcp.parent = reference
            tcp.use_connect = False
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")

        self._finish_tcp(rig, source)
        self.report({"INFO"}, f"TCP placed on '{source.name}'")
        return {"FINISHED"}

    @staticmethod
    def _tool_frame(rig, source):
        """Where the tool should sit: the link frame, then the rig's offset.

        The offset is read in the *link* frame rather than the bone's, so the
        numbers match a URDF ``<origin rpy="...">`` for the same tool.

        Zero means the flange's own link frame -- **not** where the importer
        puts the marker, which is normally the deepest link, one or more fixed
        joints further out. That distance is what the importer seeds into the
        offset, so reproducing an import means applying the seeded value rather
        than clearing it.
        """
        link = builder.link_frame_of(source.bone)
        if link is None:
            return None
        offset = Matrix.Translation(
            Vector(getattr(rig, "kinema_tcp_offset", (0.0, 0.0, 0.0)))
        ) @ Euler(
            Vector(getattr(rig, "kinema_tcp_rpy", (0.0, 0.0, 0.0))), "XYZ"
        ).to_matrix().to_4x4()
        return link @ offset

    @staticmethod
    def _source_bone(context, rig, name: str = ""):
        """The bone to place the TCP on, in order of how explicit it is.

        No selection scan: ``Bone`` carries no selection flag at all in Blender
        5.x, and reading one is what made this operator raise an AttributeError
        every time it was used from Object mode.
        """
        if name:
            return rig.pose.bones.get(name)
        bone = context.active_pose_bone
        if bone is not None and bone.id_data is rig:
            return bone
        # In Edit mode this is an EditBone, which shares the bone's name.
        #
        # Checked against this rig's armature, not just resolved by name:
        # active_rig() will happily return a *selected* rig while a different
        # armature is the active object, and two robots in one scene very
        # often have a "joint3" each -- so a bare name lookup could move the
        # TCP on the rig the user is not looking at.
        active = getattr(context, "active_bone", None)
        if active is not None and active.id_data == rig.data:
            return rig.pose.bones.get(active.name)
        return None

    @staticmethod
    def _finish_tcp(rig, source) -> None:
        rig[builder.PROP_TCP_BONE] = builder.TCP_BONE
        # The URDF link, not the bone name -- the panel labels this "Link:", and
        # the importer writes a link here, so writing a bone name made the two
        # disagree depending on how the TCP had last been placed.
        rig[builder.PROP_TCP_LINK] = source.bone.get(
            builder.PROP_CHILD_LINK, source.name
        )
        rig.kinema_tcp_parent = source.name

        armature = rig.data
        tcp_collection = next(
            (c for c in armature.collections_all if c.name == builder.COLLECTION_TCP),
            None,
        )
        bone = armature.bones.get(builder.TCP_BONE)
        if tcp_collection is not None and bone is not None:
            tcp_collection.assign(bone)

        pose_bone = rig.pose.bones.get(builder.TCP_BONE)
        if pose_bone is not None:
            from ..rig import widgets

            pose_bone.custom_shape = widgets.ensure_widgets()["tcp"]
            pose_bone.use_custom_shape_bone_size = True
            # A marker, not a control: M5 adds a separate movable IK target.
            pose_bone.lock_location = (True, True, True)
            pose_bone.lock_rotation = (True, True, True)
            pose_bone.lock_rotation_w = True
            pose_bone.lock_scale = (True, True, True)

        # Re-parenting the TCP re-roots the IK chain, but the bone keeps its
        # name, so get_solver's staleness check sees nothing wrong and would
        # happily go on driving the old chain. The demos only escaped this by
        # calling set_tcp before add_ik.
        manager.invalidate(rig.name)
        handlers.reset(rig.name)


class KINEMA_OT_reset_tcp_offset(KinemaRigOperator):
    bl_idname = "kinema.reset_tcp_offset"
    bl_label = "Reset Tool Offset"
    bl_description = (
        "Zero the tool offset, putting the TCP exactly on the selected joint's "
        "own link frame"
    )

    apply: BoolProperty(
        name="Re-place the TCP",
        description="Move the marker straight away, rather than only clearing the fields",
        default=True,
        options={"SKIP_SAVE"},
    )

    def execute(self, context: bpy.types.Context) -> set[str]:
        rig = active_rig(context)
        rig.kinema_tcp_offset = (0.0, 0.0, 0.0)
        rig.kinema_tcp_rpy = (0.0, 0.0, 0.0)

        parent = rig.kinema_tcp_parent
        if self.apply and parent and parent in rig.pose.bones:
            return bpy.ops.kinema.set_tcp(bone=parent)

        self.report({"INFO"}, "Tool offset cleared")
        return {"FINISHED"}


classes = (
    KINEMA_OT_reset_pose,
    KINEMA_OT_key_joints,
    KINEMA_OT_reset_link_meshes,
    KINEMA_OT_set_tcp,
    KINEMA_OT_reset_tcp_offset,
)
