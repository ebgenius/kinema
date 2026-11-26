"""Trace Tool. Needs a real ``bpy``; the trace after Optimize Motion needs JAX too.

The claims worth holding onto:
- a trace is the tool as playback shows it, on every frame, in the tool's own
  frame: to 0.001 mm, before and after Optimize Motion;
- its rotation keys are quaternions that never change sign between frames;
- its speed and acceleration curves agree with its own positions;
- it keeps its keys and its path when the job changes;
- each trace has its own colour, and removing one removes its empty and its keys.

The job is ``test_optimize``'s: arm6 at 30 fps, Joint, Joint, Linear, Joint over
frames 1, 61, 121 and 166, every joint at 1.5 rad/s and 6 rad/s².
"""

from __future__ import annotations

import importlib
import math

import numpy as np
import pytest

from ..conftest import requires_bpy
from . import test_optimize
from .test_optimize import TIGHT, _job
from .test_sidebar import _drawn

# test_optimize's fixtures, under their own names: its arm6, limited as its jobs need.
arm6, builder, modules = test_optimize.arm6, test_optimize.builder, test_optimize.modules

pytestmark = requires_bpy

#: The linear move's frames: arrived at on its first, left on its last.
LINE = (TIGHT[1], TIGHT[2])


@pytest.fixture
def trace(addon):
    return importlib.import_module(f"{addon.__name__}.ops.trace")


@pytest.fixture
def job(arm6, builder):
    import bpy

    _job(builder, arm6, TIGHT)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = TIGHT[0], TIGHT[-1]
    return arm6


def _traced(trace, rig):
    import bpy

    before = set(bpy.data.objects)
    assert "FINISHED" in bpy.ops.kinema.trace_tool()
    (made,) = set(bpy.data.objects) - before
    assert made.kinema_trace_of == rig
    return made


def _played(builder, rig, frames):
    """The tool's world pose on each of ``frames`` as playback shows it."""
    import bpy

    poses = {}
    tool = rig.pose.bones[builder.tool_bone_name(rig)]
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        poses[frame] = np.array(rig.matrix_world @ tool.matrix @ builder.BONE_TO_TOOL)
    return poses


def _trace_poses(empty, frames):
    import bpy

    poses = {}
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        poses[frame] = np.array(empty.matrix_world)
    return poses


def _assert_matches_playback(builder, rig, empty):
    frames = range(TIGHT[0], TIGHT[-1] + 1)
    played, traced = _played(builder, rig, frames), _trace_poses(empty, frames)
    for frame in frames:
        a, b = played[frame], traced[frame]
        assert np.linalg.norm(a[:3, 3] - b[:3, 3]) < 1e-6, frame
        # The tool's axes, its Z out of the tool, not the bone's: each within
        # 1e-5 rad of playback's. Compared directly, since an acos of float32
        # matrices can't tell anything under about 5e-4 rad from none.
        assert np.abs(a[:3, :3] - b[:3, :3]).max() < 1e-5, frame


def _keys(empty, path, index=0):
    anim = empty.animation_data
    bag = anim.action.layers[0].strips[0].channelbag(anim.action_slot)
    curve = bag.fcurves.find(path, index=index)
    assert curve is not None, path
    return {int(p.co[0]): p.co[1] for p in curve.keyframe_points}


class TestTracing:
    def test_a_trace_is_the_tool_as_playback_shows_it(self, job, builder, trace):
        empty = _traced(trace, job)

        _assert_matches_playback(builder, job, empty)
        assert empty.rotation_mode == "QUATERNION"
        assert empty.parent is None
        frames = list(range(TIGHT[0], TIGHT[-1] + 1))
        path = empty.motion_path
        assert path is not None and len(path.points) == len(frames)
        assert path.use_custom_color
        assert tuple(path.color) == pytest.approx(tuple(path.color_post))
        shown = empty.animation_visualization.motion_path
        assert not (shown.show_keyframe_highlight or shown.show_keyframe_numbers)

    def test_its_quaternions_keep_their_sign_where_the_tools_would_flip(
        self, job, builder, trace
    ):
        """The robot turned 150° about Z: twice along the job, the quaternion of the
        tool's world pose changes sign between frames, though the tool barely turns."""
        import bpy
        from mathutils import Matrix

        job.rotation_euler[2] = math.radians(150.0)
        bpy.context.view_layer.update()
        frames = list(range(TIGHT[0], TIGHT[-1] + 1))
        played = _played(builder, job, frames)
        raw = [Matrix(played[f].tolist()).to_quaternion() for f in frames]
        assert any(a.dot(b) < 0.0 for a, b in zip(raw, raw[1:], strict=False)), (
            "precondition: the tool's quaternion flips sign somewhere"
        )

        empty = _traced(trace, job)
        keyed = np.array([
            [_keys(empty, "rotation_quaternion", i)[f] for i in range(4)] for f in frames
        ])
        assert (np.sum(keyed[1:] * keyed[:-1], axis=1) > 0.99).all(), (
            "a quaternion key changed sign between frames"
        )
        _assert_matches_playback(builder, job, empty)

    def test_its_speeds_agree_with_its_positions(self, job, trace):
        import bpy

        empty = _traced(trace, job)
        fps = bpy.context.scene.render.fps
        frames = range(TIGHT[0], TIGHT[-1] + 1)
        where = np.array([[_keys(empty, "location", i)[f] for i in range(3)] for f in frames])
        speed, acceleration = _keys(empty, '["Speed"]'), _keys(empty, '["Acceleration"]')
        assert sorted(speed) == list(frames) == sorted(acceleration)
        # From the frame before, and the frames either side: each frame inside.
        for row, frame in enumerate(frames[1:-1], 1):
            moved = np.linalg.norm(where[row] - where[row - 1]) * fps
            assert speed[frame] == pytest.approx(moved, abs=1e-5), frame
            change = np.linalg.norm(where[row + 1] - 2.0 * where[row] + where[row - 1]) * fps**2
            assert acceleration[frame] == pytest.approx(change, abs=1e-3), frame
        # The line's steady speed, and the hard change entering it.
        along = [speed[f] for f in range(LINE[0] + 1, LINE[1] + 1)]
        # Steady to live IK's tolerance on the line, a few microns a frame.
        assert max(along) - min(along) < 1e-3
        assert along[0] == pytest.approx(0.129, abs=1e-3)
        assert acceleration[LINE[0]] > 5.0
        turning = _keys(empty, '["Angular Speed"]')
        assert max(turning.values()) > 0.1, "precondition: the tool turns on the joint moves"

    def test_it_keeps_its_keys_when_the_job_changes(self, job, builder, trace):
        import bpy

        empty = _traced(trace, job)
        kept = {i: _keys(empty, "location", i) for i in range(3)}
        job.kinema_motions[-1].move = "LINEAR"
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        played = _played(builder, job, [140])[140]
        assert np.linalg.norm(played[:3, 3] - _trace_poses(empty, [140])[140][:3, 3]) > 1e-3, (
            "precondition: the job changed where the tool is"
        )
        assert {i: _keys(empty, "location", i) for i in range(3)} == kept
        assert empty.motion_path is not None

    def test_each_trace_has_its_own_colour_and_removing_one_removes_it(self, job, trace):
        import bpy

        first, second = _traced(trace, job), _traced(trace, job)
        assert first.name != second.name
        assert tuple(first.motion_path.color) != tuple(second.motion_path.color)
        assert trace.traces_of(bpy.context.scene, job) == [first, second]

        name, action = first.name, first.animation_data.action.name
        assert "FINISHED" in bpy.ops.kinema.remove_trace(name=name)
        assert name not in bpy.data.objects
        assert action not in bpy.data.actions
        assert trace.traces_of(bpy.context.scene, job) == [second]
        third = _traced(trace, job)
        assert tuple(third.motion_path.color) != tuple(second.motion_path.color)

    def test_a_new_colour_is_the_whole_traces(self, job, trace):
        """Set from the panel's row: the path either side of the current frame, and the
        empty, all take it."""
        import bpy

        empty = _traced(trace, job)
        bpy.context.scene.frame_set(80)
        empty.kinema_trace_colour = (0.2, 0.3, 0.4)
        path = empty.motion_path
        for shown in (path.color, path.color_post, empty.color[:3]):
            assert tuple(shown) == pytest.approx((0.2, 0.3, 0.4))

    def test_tracing_leaves_the_scene_at_the_time_it_found_it(self, job, trace):
        import bpy

        scene = bpy.context.scene
        scene.frame_set(50, subframe=0.25)
        _traced(trace, job)
        assert (scene.frame_current, scene.frame_subframe) == (50, pytest.approx(0.25))

    def test_a_trace_goes_in_the_scene_it_was_made_in(self, job, trace):
        """The robot linked into a second scene through a collection of that scene's:
        traced there, the trace is in that scene and its list."""
        import bpy

        other = bpy.data.scenes.new("Other")
        there = bpy.data.collections.new("There")
        try:
            other.render.fps = bpy.context.scene.render.fps
            other.frame_start, other.frame_end = TIGHT[0], TIGHT[-1]
            other.collection.children.link(there)
            there.objects.link(job)
            assert job.users_collection[0] not in other.collection.children_recursive, (
                "precondition: the robot's first collection is in the first scene"
            )
            with bpy.context.temp_override(scene=other, view_layer=other.view_layers[0]):
                assert "FINISHED" in bpy.ops.kinema.trace_tool()
            (made,) = trace.traces_of(other, job)
            assert made.users_collection[0] == there
        finally:
            # clean_scene purges objects, not scenes or collections.
            bpy.data.scenes.remove(other)
            bpy.data.collections.remove(there)

    @pytest.mark.parametrize("how", ["excluded", "hidden", "disabled"])
    def test_a_trace_goes_in_a_collection_the_view_layer_shows(self, job, trace, how):
        """The robot's first collection excluded from the view layer, hidden in it, or
        disabled in viewports, the robot shown through another: the trace goes in that
        one, where it shows."""
        import bpy

        view_layer = bpy.context.view_layer
        first = job.users_collection[0]
        shown = bpy.data.collections.new("Shown")
        try:
            bpy.context.scene.collection.children.link(shown)
            shown.objects.link(job)
            layer = next(
                child for child in view_layer.layer_collection.children
                if child.collection == first
            )
            if how == "excluded":
                layer.exclude = True
            elif how == "hidden":
                layer.hide_viewport = True
            else:
                first.hide_viewport = True
            assert job.users_collection[0] == first, (
                "precondition: the excluded collection is the robot's first"
            )
            assert job.visible_get(), "precondition: the robot still shows"
            made = _traced(trace, job)
            assert made.users_collection[0] == shown
            assert made.visible_get()
        finally:
            for child in view_layer.layer_collection.children:
                child.exclude = child.hide_viewport = False
            first.hide_viewport = False
            for obj in list(shown.objects):
                shown.objects.unlink(obj)
            bpy.data.collections.remove(shown)

    def test_a_trace_whose_robot_is_gone_can_still_be_removed(self, job, trace):
        import bpy

        made = _traced(trace, job)
        name = made.name
        bpy.data.objects.remove(job)
        assert made.kinema_trace_of is None, "precondition: the trace's robot is gone"
        assert "FINISHED" in bpy.ops.kinema.remove_trace(name=name)
        assert name not in bpy.data.objects

    def test_remove_trace_refuses_an_object_that_isnt_one(self, job):
        import bpy

        assert "kinema_trace_index" not in job
        with pytest.raises(RuntimeError):
            bpy.ops.kinema.remove_trace(name=job.name)
        assert job.name in bpy.data.objects

    def test_a_trace_after_optimizing_matches_playback_and_eases_into_the_line(
        self, job, builder, trace
    ):
        before = _traced(trace, job)
        import bpy

        assert "FINISHED" in bpy.ops.kinema.optimize_motion()
        after = _traced(trace, job)

        _assert_matches_playback(builder, job, after)
        speed = {name: _keys(e, '["Speed"]') for name, e in (("b", before), ("a", after))}
        for frame in range(LINE[0] + 1, LINE[1] + 1):
            assert speed["a"][frame] == pytest.approx(speed["b"][frame], abs=1e-3), frame
        entering = {name: _keys(e, '["Acceleration"]')[LINE[0]] for name, e in
                    (("b", before), ("a", after))}
        assert entering["b"] > 5.0 and entering["a"] < 1.0



def test_the_motions_panel_offers_the_button_and_lists_the_rigs_traces(job, trace, addon):
    import bpy

    panel = importlib.import_module(f"{addon.__name__}.ui.panel")
    assert ("operator", "kinema.trace_tool") in _drawn(panel.KINEMA_PT_motions, bpy.context)
    assert ("prop", "name") not in _drawn(panel.KINEMA_PT_motions, bpy.context)

    _traced(trace, job), _traced(trace, job)
    drawn = _drawn(panel.KINEMA_PT_motions, bpy.context)
    assert drawn.count(("prop", "name")) == 2
    assert drawn.count(("prop", "kinema_trace_colour")) == 2
    assert drawn.count(("operator", "kinema.remove_trace")) == 2
