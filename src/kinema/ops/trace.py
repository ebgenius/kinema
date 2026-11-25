"""Trace Tool: the tool's path as playback shows it, kept in an empty to compare (#87).

Comparing a job before and after Optimize Motion, or after a retime, means
seeing where the tool went, and how fast. **Trace Tool**, in Motions, plays the
scene's frames as playback does and keys an empty with the tool's pose on each:

* **The tool's own frame**, its Z out of the tool, not the bone's, and its
  rotation as quaternions, which turn smoothly where Euler angles can jump.
* **Its speed and acceleration**, linear and angular, as properties keyed on
  every frame (``rig/trace.py`` measures them), to compare in the Graph Editor
  in real units.
* **Its path in the viewport,** a Blender motion path in the trace's colour,
  with a dot on every frame, shown whether the trace is selected or not.

A trace is a snapshot. It is in world space and parented to nothing, so it keeps
its keys and its path when the rig, the job or the robot's placement changes.
Several sit side by side, each its own colour, listed under the button with the
rig they trace; removing one removes its empty and its keys.

Over the scene's frame range, or its preview range when that is on, as playback
plays them: a frame either side is played too, since a rate is measured from a
frame's neighbours.
"""

from __future__ import annotations

import bpy
import numpy as np
from bpy.props import FloatVectorProperty, PointerProperty, StringProperty
from bpy.types import Operator
from mathutils import Matrix

from ..rig import builder
from ..rig import trace as rates_of
from ..ui.panel import active_rig
from .outcome import Outcome, reported

#: The colours traces take in turn, so two side by side tell apart.
COLOURS = (
    (1.0, 0.55, 0.1),
    (0.15, 0.75, 1.0),
    (0.95, 0.3, 0.75),
    (0.55, 0.9, 0.2),
    (1.0, 0.9, 0.2),
    (0.6, 0.45, 1.0),
)
#: The trace's empty, as large as a waypoint's marker is for the same robot.
MARKER_SCALE = 0.08


def frames(scene) -> tuple[int, int]:
    """The frames playback plays: the preview range while it is on, else the scene's."""
    if scene.use_preview_range:
        return scene.frame_preview_start, scene.frame_preview_end
    return scene.frame_start, scene.frame_end


def tool_poses(context, rig, first: int, last: int) -> list[np.ndarray]:
    """The tool's pose in world space on each frame from ``first`` to ``last``, as
    playback shows it: the frame set with the handlers live, so live IK solves it."""
    scene = context.scene
    tool = builder.tool_bone_name(rig)
    poses = []
    # The subframe too: tracing leaves the scene at the time it found it.
    original, subframe = scene.frame_current, scene.frame_subframe
    window = context.window_manager
    window.progress_begin(first, last)
    try:
        for frame in range(first, last + 1):
            scene.frame_set(frame)
            context.view_layer.update()
            pose = rig.matrix_world @ rig.pose.bones[tool].matrix @ builder.BONE_TO_TOOL
            location, rotation, _ = pose.decompose()
            # Without the rig's scale: a frame, as the panel's tool is.
            poses.append(np.array(
                Matrix.Translation(location) @ rotation.to_matrix().to_4x4(), dtype=float
            ))
            window.progress_update(frame)
    finally:
        window.progress_end()
        scene.frame_set(original, subframe=subframe)
    return poses


def traces_of(scene, rig) -> list:
    """The traces of ``rig`` in ``scene``, oldest first."""
    found = [obj for obj in scene.objects if obj.kinema_trace_of == rig]
    return sorted(found, key=lambda obj: obj.get("kinema_trace_index", 0))


def trace_tool(context, rig) -> Outcome:
    """Trace ``rig``'s tool over the frames playback plays: what Trace Tool does."""
    scene = context.scene
    if builder.tool_bone_name(rig) not in rig.pose.bones:
        return Outcome(False, "ERROR", "This rig has no tool bone to trace")
    first, last = frames(scene)
    if last <= first:
        return Outcome(False, "ERROR", "A trace needs at least two frames to measure")
    poses = tool_poses(context, rig, first - 1, last + 1)
    fps = scene.render.fps / scene.render.fps_base
    measured = rates_of.rates(poses, fps)

    index = 1 + max((obj.get("kinema_trace_index", 0) for obj in traces_of(scene, rig)), default=0)
    colour = COLOURS[(index - 1) % len(COLOURS)]
    empty = bpy.data.objects.new(f"{rig.name}.Trace {index}", None)
    empty.empty_display_type = "ARROWS"
    extent = max(rig.dimensions) if any(rig.dimensions) else 1.0
    empty.empty_display_size = max(extent * MARKER_SCALE, 0.01)
    empty.rotation_mode = "QUATERNION"
    empty.kinema_trace_of = rig
    empty["kinema_trace_index"] = index
    _collection(scene, rig).objects.link(empty)

    _key(empty, first, poses[1:-1], measured)
    _path(context, empty, first, last)
    # Its update colours the empty and both sides of its path.
    empty.kinema_trace_colour = colour
    return Outcome(
        True, "INFO", f"Traced '{rig.name}' over frames {first}-{last} as '{empty.name}'"
    )


def _key(empty, first: int, poses, measured) -> None:
    """Key ``empty`` with ``poses`` and the rates, one key a frame from ``first``."""
    for name, unit in rates_of.UNITS.items():
        empty[name] = 0.0
        ui = empty.id_properties_ui(name)
        ui.update(description=f"The traced tool's {name.lower()}, in {unit}", min=0.0)
        if unit in ("m/s", "m/s²"):
            ui.update(subtype="VELOCITY" if unit == "m/s" else "ACCELERATION")
    action = bpy.data.actions.new(empty.name)
    empty.animation_data_create()
    empty.animation_data.action = action

    locations, rotations = [], []
    previous = None
    for pose in poses:
        location, rotation, _ = Matrix(pose.tolist()).decompose()
        if previous is not None:
            # A matrix's quaternion has either sign: keep each to the last one's,
            # so the curves turn the short way between frames.
            rotation.make_compatible(previous)
        previous = rotation
        locations.append(location)
        rotations.append(rotation)
    channels = [("location", i, [v[i] for v in locations]) for i in range(3)]
    channels += [("rotation_quaternion", i, [q[i] for q in rotations]) for i in range(4)]
    channels += [(f'["{name}"]', 0, measured[name]) for name in rates_of.UNITS]
    for path, index, values in channels:
        curve = action.fcurve_ensure_for_datablock(empty, path, index=index)
        curve.keyframe_points.add(len(values))
        co = np.empty(2 * len(values))
        co[0::2] = np.arange(first, first + len(values))
        co[1::2] = values
        curve.keyframe_points.foreach_set("co", co.tolist())
        # A key a frame: straight between them, with nothing made up in between.
        curve.keyframe_points.foreach_set("interpolation", [_LINEAR] * len(values))
        curve.update()


#: LINEAR's value in a keyframe's interpolation enum, as foreach_set takes it.
_LINEAR = bpy.types.Keyframe.bl_rna.properties["interpolation"].enum_items["LINEAR"].value


def _collection(scene, rig):
    """Where a trace of ``rig`` goes: a collection of the rig's in ``scene``, else the
    scene's own. A rig linked into two scenes can have its first collection in the
    other one, where the trace would be missing from this scene and its list."""
    here = scene.collection.children_recursive
    for collection in rig.users_collection:
        if collection == scene.collection or collection in here:
            return collection
    return scene.collection


def _path(context, empty, first: int, last: int) -> None:
    """``empty``'s motion path over its frames. Worked out once: its keys never change."""
    settings = empty.animation_visualization.motion_path
    settings.type = "RANGE"
    settings.frame_start, settings.frame_end = first, last
    # Every frame is a key, so Blender's highlighted and numbered keys would bury
    # the path: a plain dot a frame shows the speed by how far apart they are.
    settings.show_frame_numbers = False
    settings.show_keyframe_highlight = False
    settings.show_keyframe_numbers = False
    with context.temp_override(
        active_object=empty, object=empty,
        selected_objects=[empty], selected_editable_objects=[empty],
    ):
        bpy.ops.object.paths_calculate(display_type="RANGE", range="MANUAL")


def _recoloured(empty, _context) -> None:
    """A trace's colour, on its empty and its path either side of the current frame,
    so it reads as one trace however its colour was changed."""
    colour = tuple(empty.kinema_trace_colour)
    empty.color = (*colour, 1.0)
    path = empty.motion_path
    if path is not None:
        path.use_custom_color = True
        path.color = path.color_post = colour


def remove_trace(obj) -> None:
    """Delete a trace's empty, and its keys where nothing else uses them."""
    action = obj.animation_data.action if obj.animation_data else None
    bpy.data.objects.remove(obj)
    if action is not None and not action.users:
        bpy.data.actions.remove(action)


class KINEMA_OT_trace_tool(Operator):
    bl_idname = "kinema.trace_tool"
    bl_label = "Trace Tool"
    bl_description = (
        "Play the frames as playback does and keep the tool's path in an empty: its "
        "pose, speed and acceleration on every frame, to compare with another trace"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return active_rig(context) is not None

    def execute(self, context: bpy.types.Context) -> set[str]:
        return reported(self, trace_tool(context, active_rig(context)))


class KINEMA_OT_remove_trace(Operator):
    bl_idname = "kinema.remove_trace"
    bl_label = "Remove Trace"
    bl_description = "Delete this trace's empty and its keys"
    bl_options = {"REGISTER", "UNDO", "INTERNAL"}

    name: StringProperty(name="Trace", description="The trace's empty")

    def execute(self, context: bpy.types.Context) -> set[str]:
        obj = bpy.data.objects.get(self.name)
        if obj is None or obj.kinema_trace_of is None:
            self.report({"ERROR"}, f"No trace called '{self.name}'")
            return {"CANCELLED"}
        remove_trace(obj)
        return {"FINISHED"}


def draw_traces(layout, context, rig) -> None:
    """The rig's traces, one row each: its name, its path's colour, and remove."""
    from ..ui.panel import section

    found = traces_of(context.scene, rig)
    if not found:
        return
    body = section(layout, context, "show_traces", "Traces", "CURVE_PATH")
    if body is None:
        return
    column = body.column(align=True)
    for obj in found:
        row = column.row(align=True)
        if obj.motion_path is not None:
            colour = row.row(align=True)
            colour.ui_units_x = 1.5
            colour.prop(obj, "kinema_trace_colour", text="")
        row.prop(obj, "name", text="")
        row.operator("kinema.remove_trace", text="", icon="X").name = obj.name


def register_props() -> None:
    bpy.types.Object.kinema_trace_of = PointerProperty(
        type=bpy.types.Object,
        name="Trace Of",
        description="The rig whose tool this empty traces",
    )
    bpy.types.Object.kinema_trace_colour = FloatVectorProperty(
        name="Colour",
        description="The trace's colour: its path's, either side of the current frame, "
        "and its empty's",
        subtype="COLOR",
        size=3,
        min=0.0,
        max=1.0,
        default=COLOURS[0],
        update=_recoloured,
    )


def unregister_props() -> None:
    del bpy.types.Object.kinema_trace_colour
    del bpy.types.Object.kinema_trace_of


classes = (KINEMA_OT_trace_tool, KINEMA_OT_remove_trace)
