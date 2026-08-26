"""IK operators: add a target, snap it, bake it down.

The IK control is an ordinary bone. It is not parented into the joint chain --
that would make its own position depend on the solve it drives -- so it hangs
off the Root bone and is keyframed like any other control.

Baking matters more than it sounds. A live handler is convenient but it is also
a dependency: a .blend that only animates correctly while Kinema is installed
and enabled is not a deliverable. Baking writes plain FK keyframes, after which
the file renders anywhere, on a farm, with the add-on absent.
"""

from __future__ import annotations

import bpy
from bpy.props import BoolProperty, IntProperty
from bpy.types import Operator

from .. import handlers
from ..rig import builder, widgets
from ..solver import manager
from ..ui.panel import active_rig

IK_SUFFIX = ".ik"


def _ik_bone_name(tcp_bone: str) -> str:
    return f"{tcp_bone}{IK_SUFFIX}"


class KinemaRigOperator(Operator):
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return active_rig(context) is not None


class KINEMA_OT_add_ik(KinemaRigOperator):
    bl_idname = "kinema.add_ik"
    bl_label = "Add IK Target"
    bl_description = "Create a keyframable IK control at the tool centre point"

    def execute(self, context: bpy.types.Context) -> set[str]:
        rig = active_rig(context)
        tcp_bone = rig.get(builder.PROP_TCP_BONE) or builder.TCP_BONE
        if tcp_bone not in rig.data.bones:
            self.report({"ERROR"}, "This rig has no TCP; create one first")
            return {"CANCELLED"}

        ik_name = _ik_bone_name(tcp_bone)
        previous_active = context.view_layer.objects.active
        context.view_layer.objects.active = rig
        if rig.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        bpy.ops.object.mode_set(mode="EDIT")
        try:
            edit_bones = rig.data.edit_bones
            tcp = edit_bones[tcp_bone]
            ik = edit_bones.get(ik_name) or edit_bones.new(ik_name)
            ik.head = tcp.head
            ik.tail = tcp.tail
            ik.roll = tcp.roll
            # Parented to Root, never into the chain: an IK goal that moved
            # with the arm it drives would chase its own tail.
            ik.parent = edit_bones.get(builder.ROOT_BONE)
            ik.use_connect = False
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")

        pose_bone = rig.pose.bones[ik_name]
        pose_bone.rotation_mode = "QUATERNION"
        pose_bone.custom_shape = widgets.ensure_widgets()["ik_target"]
        pose_bone.use_custom_shape_bone_size = True

        armature = rig.data
        for collection in armature.collections_all:
            if collection.name == builder.COLLECTION_IK:
                collection.assign(armature.bones[ik_name])
                collection.is_visible = True

        rig[builder.PROP_IK_BONE] = ik_name

        # Snap the goal onto the tool's *current* pose before switching live IK
        # on. A new bone starts at its rest position, so enabling the solver
        # first would immediately drag the arm back to the URDF rest pose and
        # throw away whatever the user had posed.
        context.view_layer.update()
        pose_bone.matrix = rig.pose.bones[tcp_bone].matrix.copy()
        context.view_layer.update()

        manager.invalidate(rig.name)
        handlers.reset(rig.name)
        handlers.register_handlers()

        # Pay JAX's one-off JIT compilation here, behind a wait cursor, rather
        # than letting it land on the user's first drag of the control.
        compile_seconds = self._warm_up(context, rig, ik_name)

        rig.kinema_ik_enabled = True
        if previous_active is not None:
            context.view_layer.objects.active = previous_active

        message = f"IK target '{ik_name}' created"
        if compile_seconds > 1.0:
            message += f" (solver compiled in {compile_seconds:.1f}s)"
        self.report({"INFO"}, message)
        return {"FINISHED"}

    @staticmethod
    def _warm_up(context, rig, ik_name: str) -> float:
        import time

        solver = manager.get_solver(rig, ik_name)
        if solver is None:
            return 0.0
        started = time.perf_counter()
        context.window.cursor_set("WAIT")
        try:
            # Solving for where the tool already is: compiles the kernel without
            # moving the robot.
            solver.solve(rig, getattr(rig, "kinema_solver_mode", manager.MODE_PYROKI))
        except Exception:  # noqa: BLE001 - warmup is best-effort
            pass
        finally:
            context.window.cursor_set("DEFAULT")
        return time.perf_counter() - started


class KINEMA_OT_remove_ik(KinemaRigOperator):
    bl_idname = "kinema.remove_ik"
    bl_label = "Remove IK Target"
    bl_description = "Delete the IK control and return the rig to plain FK"

    def execute(self, context: bpy.types.Context) -> set[str]:
        rig = active_rig(context)
        ik_name = rig.get(builder.PROP_IK_BONE)
        if not ik_name:
            self.report({"WARNING"}, "This rig has no IK target")
            return {"CANCELLED"}

        context.view_layer.objects.active = rig
        if rig.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.mode_set(mode="EDIT")
        try:
            edit_bones = rig.data.edit_bones
            if ik_name in edit_bones:
                edit_bones.remove(edit_bones[ik_name])
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")

        if builder.PROP_IK_BONE in rig:
            del rig[builder.PROP_IK_BONE]
        rig.kinema_ik_enabled = False
        manager.invalidate(rig.name)
        handlers.reset(rig.name)
        self.report({"INFO"}, "IK target removed")
        return {"FINISHED"}


class KINEMA_OT_snap_ik(KinemaRigOperator):
    bl_idname = "kinema.snap_ik"
    bl_label = "Snap IK to Tool"
    bl_description = (
        "Move the IK control onto the tool's current position, so switching "
        "from FK posing to IK does not jerk the arm"
    )

    def execute(self, context: bpy.types.Context) -> set[str]:
        rig = active_rig(context)
        ik_name = rig.get(builder.PROP_IK_BONE)
        tcp_name = rig.get(builder.PROP_TCP_BONE) or builder.TCP_BONE
        if not ik_name or ik_name not in rig.pose.bones:
            self.report({"ERROR"}, "This rig has no IK target")
            return {"CANCELLED"}

        context.view_layer.update()
        rig.pose.bones[ik_name].matrix = rig.pose.bones[tcp_name].matrix.copy()
        context.view_layer.update()
        handlers.reset(rig.name)
        self.report({"INFO"}, "IK target snapped to the tool")
        return {"FINISHED"}


class KINEMA_OT_bake_ik(KinemaRigOperator):
    bl_idname = "kinema.bake_ik"
    bl_label = "Bake IK to Keyframes"
    bl_description = (
        "Solve every frame and key the joints, so the animation plays back "
        "without Kinema installed"
    )

    frame_start: IntProperty(name="Start", default=1)
    frame_end: IntProperty(name="End", default=250)
    step: IntProperty(name="Step", default=1, min=1, max=10)
    disable_live_ik: BoolProperty(
        name="Disable Live IK",
        description="Turn the live solver off afterwards, so the baked keys are what plays",
        default=True,
    )
    clear_existing: BoolProperty(
        name="Clear Existing Keys",
        description="Remove existing joint keyframes in the range before baking",
        default=True,
    )

    def invoke(self, context: bpy.types.Context, event) -> set[str]:
        scene = context.scene
        self.frame_start = scene.frame_start
        self.frame_end = scene.frame_end
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context: bpy.types.Context) -> set[str]:
        rig = active_rig(context)
        ik_name = rig.get(builder.PROP_IK_BONE)
        if not ik_name or ik_name not in rig.pose.bones:
            self.report({"ERROR"}, "This rig has no IK target to bake")
            return {"CANCELLED"}

        solver = manager.get_solver(rig, ik_name)
        if solver is None:
            self.report({"ERROR"}, "Could not prepare a solver for this rig")
            return {"CANCELLED"}

        scene = context.scene
        original_frame = scene.frame_current
        mode = getattr(rig, "kinema_solver_mode", manager.MODE_PYROKI)
        if mode == manager.MODE_OFF:
            mode = manager.MODE_PYROKI

        joint_bones = [rig.pose.bones[name] for name in solver.chain.bone_names]
        if self.clear_existing:
            self._clear_keys(rig, solver.chain.bone_names)

        frames = range(self.frame_start, self.frame_end + 1, self.step)
        failed = 0
        window = context.window_manager
        window.progress_begin(0, len(frames))
        try:
            for index, frame in enumerate(frames):
                scene.frame_set(frame)
                result = solver.solve(rig, mode)
                if result is None or not result.converged:
                    failed += 1
                for pose_bone, revolute in zip(
                    joint_bones, solver.chain.is_revolute, strict=True
                ):
                    path, channel = (
                        ("rotation_euler", 1) if revolute else ("location", 1)
                    )
                    pose_bone.keyframe_insert(data_path=path, index=channel, frame=frame)
                window.progress_update(index)
        finally:
            window.progress_end()
            scene.frame_set(original_frame)

        if self.disable_live_ik:
            rig.kinema_ik_enabled = False
        handlers.reset(rig.name)

        message = f"Baked {len(frames)} frames"
        if failed:
            message += f" ({failed} did not converge -- target may be out of reach)"
        self.report({"WARNING"} if failed else {"INFO"}, message)
        return {"FINISHED"}

    @staticmethod
    def _clear_keys(rig, bone_names: list[str]) -> None:
        action = rig.animation_data.action if rig.animation_data else None
        if action is None:
            return
        wanted = {f'pose.bones["{name}"]' for name in bone_names}
        # Blender 4.4+ stores curves in slotted layers/strips rather than
        # directly on the action, so walk whichever structure this build uses.
        for curves in _iter_fcurve_containers(action):
            for curve in list(curves):
                if any(curve.data_path.startswith(prefix) for prefix in wanted):
                    curves.remove(curve)


def _iter_fcurve_containers(action):
    """Yield every F-curve collection in an action, across Blender versions."""
    legacy = getattr(action, "fcurves", None)
    if legacy is not None:
        yield legacy
        return
    for layer in getattr(action, "layers", ()):
        for strip in getattr(layer, "strips", ()):
            for channelbag in getattr(strip, "channelbags", ()):
                yield channelbag.fcurves


classes = (
    KINEMA_OT_add_ik,
    KINEMA_OT_remove_ik,
    KINEMA_OT_snap_ik,
    KINEMA_OT_bake_ik,
)
