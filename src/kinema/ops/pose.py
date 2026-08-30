"""Pose-level operators: rest pose, keying, and moving the TCP."""

from __future__ import annotations

import bpy
from bpy.types import Operator
from mathutils import Vector

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


class KINEMA_OT_set_tcp(KinemaRigOperator):
    bl_idname = "kinema.set_tcp"
    bl_label = "Set TCP"
    bl_description = (
        "Move the tool-centre-point marker to the active bone, or create it there"
    )

    def execute(self, context: bpy.types.Context) -> set[str]:
        rig = active_rig(context)
        source = self._source_bone(context, rig)
        if source is None:
            self.report(
                {"ERROR"},
                "Select a bone to place the TCP on (enter Pose mode and click one)",
            )
            return {"CANCELLED"}
        if source.name == builder.TCP_BONE:
            self.report({"WARNING"}, "Select a joint bone, not the TCP itself")
            return {"CANCELLED"}

        previous_mode = context.object.mode if context.object else "OBJECT"
        previous_active = context.view_layer.objects.active

        context.view_layer.objects.active = rig
        bpy.ops.object.mode_set(mode="EDIT")
        try:
            edit_bones = rig.data.edit_bones
            reference = edit_bones[source.name]
            length = (reference.tail - reference.head).length or 0.05

            tcp = edit_bones.get(builder.TCP_BONE)
            if tcp is None:
                tcp = edit_bones.new(builder.TCP_BONE)
            # Sit on the source bone's head, pointing along its Z, matching the
            # ROS convention that a tool frame's +Z is its approach direction.
            rotation = reference.matrix.to_3x3()
            tcp.head = reference.head
            tcp.tail = reference.head + (rotation @ Vector((0.0, 0.0, 1.0))) * length * 0.8
            tcp.align_roll(rotation @ Vector((1.0, 0.0, 0.0)))
            tcp.parent = reference
            tcp.use_connect = False
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")

        self._finish_tcp(rig, source.name)

        if previous_active is not None:
            context.view_layer.objects.active = previous_active
        if previous_mode != "OBJECT" and context.object is not None:
            try:
                bpy.ops.object.mode_set(mode=previous_mode)
            except RuntimeError:
                pass

        self.report({"INFO"}, f"TCP placed on '{source.name}'")
        return {"FINISHED"}

    @staticmethod
    def _source_bone(context, rig):
        bone = context.active_pose_bone
        if bone is not None and bone.id_data is rig:
            return bone
        selected = [b for b in rig.pose.bones if b.bone.select]
        return selected[0] if selected else None

    @staticmethod
    def _finish_tcp(rig, source_name: str) -> None:
        rig[builder.PROP_TCP_BONE] = builder.TCP_BONE
        rig[builder.PROP_TCP_LINK] = source_name

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


classes = (KINEMA_OT_reset_pose, KINEMA_OT_key_joints, KINEMA_OT_set_tcp)
