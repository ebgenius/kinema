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

        # Snap the goal onto the tip's *current* pose before switching live IK
        # on. A new bone starts at its rest position, so enabling the solver
        # first would immediately drag the arm back to the URDF rest pose and
        # throw away whatever the user had posed. The tip is usually the TCP,
        # but the rig may already be aimed at a bone further up the chain.
        context.view_layer.update()
        pose_bone.matrix = rig.pose.bones[manager.tip_bone(rig)].matrix.copy()
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
        tip_name = manager.tip_bone(rig)
        if not ik_name or ik_name not in rig.pose.bones:
            self.report({"ERROR"}, "This rig has no IK target")
            return {"CANCELLED"}
        if tip_name not in rig.pose.bones:
            self.report({"ERROR"}, f"This rig has no bone named '{tip_name}'")
            return {"CANCELLED"}

        context.view_layer.update()
        rig.pose.bones[ik_name].matrix = rig.pose.bones[tip_name].matrix.copy()
        context.view_layer.update()
        handlers.reset(rig.name)
        self.report({"INFO"}, f"IK target snapped to '{tip_name}'")
        return {"FINISHED"}


class KINEMA_OT_set_ik_tip(KinemaRigOperator):
    bl_idname = "kinema.set_ik_tip"
    bl_label = "Set IK Target Bone"
    bl_description = (
        "Aim the solver at this bone. Everything above it in the chain stays "
        "free; everything below it stops being solved. If the target is already "
        "animated, this keys the change on the current frame"
    )

    #: Index into builder.joint_bones(rig); -1 restores the TCP marker.
    index: IntProperty(name="Bone", default=-1, options={"SKIP_SAVE"})
    snap: BoolProperty(
        name="Snap the Control",
        description="Move the IK control onto the new tip, so the arm does not jump",
        default=True,
        options={"SKIP_SAVE"},
    )

    def execute(self, context: bpy.types.Context) -> set[str]:
        rig = active_rig(context)
        joints = builder.joint_bones(rig)
        if self.index < -1 or self.index >= len(joints):
            self.report({"ERROR"}, "That bone is not part of this rig's chain")
            return {"CANCELLED"}

        tip_name = manager.tip_bone_for(rig, self.index)

        # Where the new tip is *now*, read before the tip changes. Writing
        # kinema_ik_tip drops the cached goal, so the very next depsgraph
        # update solves the new chain against the old goal and moves the arm --
        # and a snapshot taken after that would capture the disturbed pose and
        # bake the jerk in, which is exactly what this operator promises not to
        # do.
        context.view_layer.update()
        landing = (
            rig.pose.bones[tip_name].matrix.copy()
            if tip_name in rig.pose.bones
            else None
        )

        # No manager.invalidate() here: get_solver compares the tip and rebuilds
        # on its own, and invalidating would also throw away the compiled PyRoki
        # kernels this rig has already paid for -- the exact thing that makes
        # scrubbing a keyframed tip affordable.
        rig.kinema_ik_tip = self.index
        # On an already-animated tip a plain write does not survive: the next
        # animation evaluation puts the curve's value back, so the radio button
        # would appear to do nothing at all. Changing the target on a frame that
        # is already keyed means keying it there.
        if _tip_curves(rig):
            rig.keyframe_insert(data_path=TIP_PATH, frame=context.scene.frame_current)
        _force_constant_tip_keys(rig)

        ik_name = rig.get(builder.PROP_IK_BONE)
        if self.snap and landing is not None and ik_name and ik_name in rig.pose.bones:
            rig.pose.bones[ik_name].matrix = landing
        handlers.reset(rig.name)
        context.view_layer.update()

        self.report({"INFO"}, f"IK now targets '{tip_name}'")
        return {"FINISHED"}


class KINEMA_OT_key_ik_tip(KinemaRigOperator):
    bl_idname = "kinema.key_ik_tip"
    bl_label = "Key IK Target Bone"
    bl_description = (
        "Keyframe which bone IK aims at, so a shot can hand the goal from one "
        "bone to another part-way through"
    )

    def execute(self, context: bpy.types.Context) -> set[str]:
        rig = active_rig(context)
        frame = context.scene.frame_current
        if not rig.keyframe_insert(data_path=TIP_PATH, frame=frame):
            self.report({"WARNING"}, "Could not key the IK target bone")
            return {"CANCELLED"}

        # This, rather than the property's decorator dot, is the offered way to
        # key the tip -- because it is the only one that can guarantee the
        # channel steps rather than ramps.
        _force_constant_tip_keys(rig)
        self.report({"INFO"}, f"IK target bone keyed at frame {frame}")
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

        if manager.get_solver(rig, ik_name) is None:
            self.report({"ERROR"}, "Could not prepare a solver for this rig")
            return {"CANCELLED"}

        scene = context.scene
        original_frame = scene.frame_current
        mode = getattr(rig, "kinema_solver_mode", manager.MODE_PYROKI)
        if mode == manager.MODE_OFF:
            mode = manager.MODE_PYROKI

        frames = range(self.frame_start, self.frame_end + 1, self.step)
        failed = 0
        window = context.window_manager

        # Handlers suspended, not kinema_ik_enabled switched off: that property
        # is animatable, so a rig with a curve on it would have live IK restored
        # by every frame_set. frame_set fires the frame-change handler, which
        # solves and writes the pose -- and then this loop solves the same frame
        # again. With a keyed tip the two can disagree, and whichever ran last
        # wins.
        with handlers.suspended():
            bones = self._bones_over_range(rig, scene, frames, ik_name)
            if not bones:
                self.report({"ERROR"}, "No joints to bake on this rig")
                return {"CANCELLED"}
            if self.clear_existing:
                self._clear_keys(rig, bones)

            window.progress_begin(0, len(frames))
            try:
                for index, frame in enumerate(frames):
                    scene.frame_set(frame)
                    # Per frame, not once: the tip is keyframable, so the chain
                    # can change mid-bake and a solver captured up front would go
                    # on driving the joints of a chain that is no longer live.
                    solver = manager.get_solver(rig, ik_name)
                    result = solver.solve(rig, mode) if solver is not None else None
                    if result is None or not result.converged:
                        failed += 1
                    # Every joint active anywhere in the range gets a key on every
                    # frame, not just the ones this frame's chain solved. A joint
                    # that drops out of the chain still has a value, and a channel
                    # that stopped being keyed half way through would hold its
                    # last key instead -- a different pose than the one baked.
                    for name in bones:
                        _key_joint_value(rig.pose.bones[name], frame)
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
    def _bones_over_range(rig, scene, frames, ik_name: str) -> list[str]:
        """Every joint that is on the IK chain at any frame in the range.

        With a static tip that is just the current chain. With a keyed one the
        chain changes part-way through, and both the channels to clear and the
        channels to key are the union over the range -- clearing only today's
        chain would leave a stale curve on a joint that becomes active later.

        Scanning costs a pass of frame_set, so it is skipped entirely when the
        tip is not animated, which is the common case.
        """
        if not _tip_curves(rig):
            solver = manager.get_solver(rig, ik_name)
            return list(solver.chain.bone_names) if solver is not None else []

        # Seeded empty, not from the current frame: the frame the user happens
        # to be sitting on need not be inside the bake range, and if it targets
        # a different tip its joints would have their existing curves cleared
        # and new keys written despite never being active in the range.
        seen: dict[str, None] = {}
        original = scene.frame_current
        try:
            for frame in frames:
                scene.frame_set(frame)
                # Chain extraction is pure matrix maths off the armature; it
                # does not build a PyRoki solver, so this pass stays cheap.
                solver = manager.get_solver(rig, ik_name)
                if solver is not None:
                    seen.update(dict.fromkeys(solver.chain.bone_names))
        finally:
            scene.frame_set(original)
        # Rig order, so the baked channels read the way the chain does.
        order = {bone.name: i for i, bone in enumerate(builder.joint_bones(rig))}
        return sorted(seen, key=lambda name: order.get(name, len(order)))

    @staticmethod
    def _clear_keys(rig, bone_names: list[str]) -> None:
        wanted = {f'pose.bones["{name}"]' for name in bone_names}
        # Blender 4.4+ stores curves in slotted layers/strips rather than
        # directly on the action, and only this rig's slot is ours to clear --
        # a rig sharing the action almost certainly has bones by the same names.
        for curves in _own_fcurve_containers(rig):
            for curve in list(curves):
                if any(curve.data_path.startswith(prefix) for prefix in wanted):
                    curves.remove(curve)


#: Data path of the keyframable IK tip, on the rig object.
TIP_PATH = "kinema_ik_tip"


def _key_joint_value(pose_bone, frame: int) -> None:
    """Key the one channel this joint actually uses.

    Read off the bone rather than a chain's ``is_revolute``, so it works for a
    joint that is not on the chain currently being solved. Keying all nine
    channels would fill the dope sheet with curves that can never move.
    """
    is_prismatic = (
        pose_bone.bone.get(builder.PROP_JOINT_TYPE, "revolute") == "prismatic"
    )
    path, channel = ("location", 1) if is_prismatic else ("rotation_euler", 1)
    pose_bone.keyframe_insert(data_path=path, index=channel, frame=frame)


def _tip_curves(rig) -> list:
    """Every F-curve animating this rig's IK tip.

    A list, not a generator: both callers ask whether the tip is animated at
    all, and a generator object is truthy even when it yields nothing.
    """
    return [
        curve
        for container in _own_fcurve_containers(rig)
        for curve in container
        if curve.data_path == TIP_PATH
    ]


def _own_fcurve_containers(rig):
    """F-curve collections belonging to ``rig``, and to no other ID.

    One Action can hold channelbags for several slots, so two rigs may share
    it. Walking all of them and matching on data path alone would find another
    rig's ``kinema_ik_tip`` -- enough to auto-key this rig off someone else's
    animation and rewrite the other slot's interpolation -- and, worse for
    baking, another rig's identically named pose bones.
    """
    anim = rig.animation_data
    action = anim.action if anim else None
    if action is None:
        return
    if getattr(action, "layers", None):
        slot = getattr(anim, "action_slot", None)
        handle = getattr(slot, "handle", None)
        if handle is None:
            # A slotted action with no slot assigned animates nothing here.
            # Yielding the whole action instead would hand back every other
            # rig's curves, which is the failure this exists to prevent.
            return
        yield from _iter_fcurve_containers(action, handle)
        return
    # Pre-4.4 layout: an action belongs to one ID, so all of it is ours.
    yield from _iter_fcurve_containers(action)


def _force_constant_tip_keys(rig) -> None:
    """Make the tip channel step between values instead of ramping through them.

    The tip is an index, not a quantity. Interpolated at all, a hand-off from
    the TCP (-1) to joint3 (2) passes through 0 and 1 on the frames between the
    keys, so the solver drives two chains nobody asked for -- and each link it
    has not seen before pays its own JAX compile on the way past. Blender
    inserts keys with the user's default interpolation, Bezier out of the box,
    so the curve is normalised wherever Kinema touches it.
    """
    for curve in _tip_curves(rig):
        for point in curve.keyframe_points:
            point.interpolation = "CONSTANT"
        curve.update()


def _iter_fcurve_containers(action, slot_handle=None):
    """Yield every F-curve collection in an action, across Blender versions.

    ``slot_handle`` restricts the walk to one slot's channelbags, which is what
    keeps a shared action's other occupants out of the result.

    Layers are checked first, and that order is load-bearing. Blender 5.2 still
    exposes ``action.fcurves`` on a slotted action as a compatibility view, so
    testing for it first -- which this used to do -- takes the legacy path for
    every modern action and makes the slot filter below unreachable.
    """
    layers = getattr(action, "layers", None)
    if layers:
        for layer in layers:
            for strip in getattr(layer, "strips", ()):
                for channelbag in getattr(strip, "channelbags", ()):
                    if (
                        slot_handle is not None
                        and getattr(channelbag, "slot_handle", None) != slot_handle
                    ):
                        continue
                    yield channelbag.fcurves
        return
    legacy = getattr(action, "fcurves", None)
    if legacy is not None:
        yield legacy


classes = (
    KINEMA_OT_add_ik,
    KINEMA_OT_remove_ik,
    KINEMA_OT_snap_ik,
    KINEMA_OT_set_ik_tip,
    KINEMA_OT_key_ik_tip,
    KINEMA_OT_bake_ik,
)
