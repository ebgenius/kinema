"""The job as NLA strips (#96): ops/strips.py. Needs a real ``bpy``.

The claims:

* Generate Motion lays the job out as one strip per motion, on two tracks of
  their own taking turns, over a "Kinema Job" track holding the pauses, the
  bridging joint moves, and Before and After strips for what the action holds
  outside the job -- together tiling its frames;
* a motion's strip moved or scaled in the NLA editor retimes the motion, on the
  next operator that reads the job;
* it plays exactly as the job in one active action did: every joint, every
  frame, hand keys before and after the job, and a channel only hand keys
  animate, which a strip per motion would drop to its default;
* hand keys made in the active action since are moved into the job, so nothing
  plays on top of it;
* the preference off puts the job back in the active action;
* Bake IK writes a "Kinema Baked" strip over the baked range and leaves the job's
  action as it was; muting that track goes back to the live job;
* over a track of the user's, named as the job's track is, the strips play the
  same and leave it alone, and hand curves move with their modifiers and
  samples;
* a refused bake changes nothing, a bake whatever the preference goes on a
  strip, rebaking part of a bake keeps the rest, and regenerating mutes the
  bakes it replaces;
* in tweak mode, Generate leaves the job's action alone.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from ..conftest import requires_bpy
from .test_waypoints import Q0, REACH_A, REACH_B, _q, _teach

pytestmark = requires_bpy

# Past the action at both ends, where the first and last strips have to hold.
FRAMES = range(-30, 151)


@pytest.fixture
def builder(addon):
    return importlib.import_module(f"{addon.__name__}.rig.builder")


@pytest.fixture
def strips(addon):
    return importlib.import_module(f"{addon.__name__}.ops.strips")


@pytest.fixture
def prefs(addon):
    found = importlib.import_module(f"{addon.__name__}.prefs").get_prefs()
    was = found.job_as_strips
    yield found
    found.job_as_strips = was


@pytest.fixture
def arm6(addon, fixture_dir, clean_scene, builder):
    import bpy

    assert "FINISHED" in bpy.ops.kinema.build_robot(filepath=str(fixture_dir / "arm6.urdf"))
    rig = next(o for o in bpy.data.objects if builder.is_kinema_rig(o))
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.kinema.add_ik()
    rig.kinema_ik_enabled = False
    return rig


def _job(builder, rig):
    """home, a joint move to a, a line to b, a pause at b over 41-61, a joint move
    back home over 61-81, and hand keys: joint1 before the job and after it, and
    the wrist's scale, which nothing in the job keys, across it."""
    import bpy

    home = _teach(builder, rig, 1, Q0, "JOINT", "home")
    _teach(builder, rig, 21, REACH_A, "JOINT", "a")
    b = _teach(builder, rig, 41, REACH_B, "LINEAR", "b")
    motion = rig.kinema_motions.add()
    motion.name, motion.start, motion.end = "back", b.uid, home.uid
    motion.frame_start, motion.frame_end, motion.move = 61, 81, "JOINT"
    first, wrist = builder.joint_bones(rig)[0], builder.joint_bones(rig)[-1]
    for frame, value in ((-10, 0.2), (120, -0.4)):
        first.rotation_euler[1] = value
        first.keyframe_insert("rotation_euler", index=1, frame=frame)
    for frame, value in ((-10, 1.0), (130, 1.5)):
        wrist.scale[0] = value
        wrist.keyframe_insert("scale", index=0, frame=frame)
    bpy.context.view_layer.update()
    return wrist


def _played(builder, rig, wrist) -> dict:
    """The joints and the wrist's scale on every frame, as playback has them."""
    import bpy

    out = {}
    for frame in FRAMES:
        bpy.context.scene.frame_set(frame)
        out[frame] = np.append(_q(builder, rig), wrist.scale[0])
    return out


def _worst(a: dict, b: dict, frames=None) -> float:
    return max(float(np.max(np.abs(a[f] - b[f]))) for f in (frames or a))


def _keys(action) -> dict:
    """Every key of ``action``, by channel."""
    return {
        (c.data_path, c.array_index): [tuple(p.co) for p in c.keyframe_points]
        for bag in action.layers[0].strips[0].channelbags for c in bag.fcurves
    }


def _track(rig, name):
    return [t for t in rig.animation_data.nla_tracks if t.name == name]


def _job_strips(strips, rig) -> list:
    """Every strip of the job, on all its tracks, in frame order."""
    return sorted(
        (s for t in strips._job_tracks(rig) for s in t.strips), key=lambda s: s.frame_start
    )


def _mute_job(strips, rig, mute=True) -> None:
    for track in strips._job_tracks(rig):
        track.mute = mute


class TestLayout:
    def test_one_strip_per_motion_tiling_the_job(self, arm6, builder, strips, prefs):
        import bpy

        prefs.job_as_strips = True
        _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()

        anim = arm6.animation_data
        assert anim.action is None, "nothing left to play on top of the job"
        job = arm6.kinema_job_action
        assert job is not None and job[strips.PROP_JOB] == arm6.name
        tracks = strips._job_tracks(arm6)
        assert [t.name for t in tracks] == [
            strips.JOB_TRACK, strips.MOTIONS_TRACK, strips.MOTIONS_TRACK
        ]
        assert list(anim.nla_tracks).index(tracks[0]) < list(anim.nla_tracks).index(tracks[1])
        base, first, second = ([s.name for s in t.strips] for t in tracks)
        # The pause at b the plan adds is the job's track's; the motions take turns.
        assert base == [strips.BEFORE, "hold at b", strips.AFTER], base
        assert (first, second) == (["a", "back"], ["b"]), (first, second)
        laid = [(s.name, s.frame_start, s.frame_end) for s in _job_strips(strips, arm6)]
        names = [name for name, _, _ in laid]
        assert names == [strips.BEFORE, "a", "b", "hold at b", "back", strips.AFTER], names
        for (_, _, end), (_, start, _) in zip(laid, laid[1:], strict=False):
            assert end == start, f"a gap or overlap at {end} / {start}: {laid}"
        assert (laid[1][1], laid[-2][2]) == (1.0, 81.0)
        held = [s.extrapolation for s in tracks[0].strips]
        assert held == ["HOLD", "HOLD_FORWARD", "HOLD_FORWARD"], held
        assert {s.extrapolation for t in tracks[1:] for s in t.strips} == {"NOTHING"}

    def test_regenerating_replaces_the_strips(self, arm6, builder, strips, prefs):
        import bpy

        prefs.job_as_strips = True
        _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        before = [(t.name, [s.name for s in t.strips]) for t in strips._job_tracks(arm6)]
        count = len(arm6.animation_data.nla_tracks)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        after = [(t.name, [s.name for s in t.strips]) for t in strips._job_tracks(arm6)]
        assert after == before and len(arm6.animation_data.nla_tracks) == count


    def test_a_track_showing_another_slot_of_the_job_is_left(
        self, arm6, builder, strips, prefs
    ):
        """One action can hold several slots. A track of the user's showing the job's
        action through another slot isn't the job's: regenerating, and putting the
        job back in the active action, leave it alone."""
        import bpy

        prefs.job_as_strips = True
        _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        job = arm6.kinema_job_action
        other = job.slots.new("OBJECT", "Other")
        mine = arm6.animation_data.nla_tracks.new()
        mine.name = "Mine"
        strips._new_strip(mine, "mine", 300.0, 320.0, job, other)

        def kept():
            # Looked up each time: a track removed under a reference to it
            # would crash Blender rather than fail.
            return [[s.name for s in t.strips] for t in _track(arm6, "Mine")]

        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        assert kept() == [["mine"]]
        prefs.job_as_strips = False
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        assert kept() == [["mine"]]

    def test_put_back_leaves_the_nla_switch(self, arm6, builder, prefs):
        """The active action plays with the NLA off too, so putting the job back
        there leaves the switch as the user set it."""
        import bpy

        prefs.job_as_strips = True
        _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        arm6.animation_data.use_nla = False
        prefs.job_as_strips = False
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        assert arm6.animation_data.use_nla is False

    def test_strips_without_a_slot_give_no_curves(self, addon, arm6, builder, strips, prefs):
        """With no slot to tell this rig's curves from another's, none of the job's
        are this rig's: the same rule as for the active action."""
        import bpy

        ik = importlib.import_module(f"{addon.__name__}.ops.ik")
        prefs.job_as_strips = True
        _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        assert list(ik.own_fcurve_containers(arm6)), "precondition: the job's curves"
        for strip in _job_strips(strips, arm6):
            strip.action_slot = None
        assert list(ik.own_fcurve_containers(arm6)) == []


class TestPlayback:
    @pytest.mark.parametrize("layered", [False, True], ids=["alone", "over-a-track"])
    def test_strips_play_as_the_active_action_did_and_back(
        self, arm6, builder, strips, prefs, layered
    ):
        """The same job generated into the active action, then laid out as strips,
        then put back: the same on every frame, hand keys and all.

        Over a track of the user's -- named as Kinema names its own -- the same
        again, and the user's track stays."""
        import bpy

        if layered:
            first, wrist = builder.joint_bones(arm6)[0], builder.joint_bones(arm6)[-1]
            first.rotation_euler[1] = 0.3
            first.keyframe_insert("rotation_euler", index=1, frame=1)
            wrist.scale[0] = 2.0
            wrist.keyframe_insert("scale", index=0, frame=1)
            anim = arm6.animation_data
            base, anim.action = anim.action, None
            mine = anim.nla_tracks.new()
            mine.name = strips.JOB_TRACK
            mine.strips.new("mine", 1, base)
        wrist = _job(builder, arm6)
        prefs.job_as_strips = False
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        assert arm6.kinema_job_action is None
        as_action = _played(builder, arm6, wrist)
        assert as_action[50][-1] != pytest.approx(1.0), "precondition: the scale animates"

        prefs.job_as_strips = True
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        assert strips.laid_out(arm6)
        as_strips = _played(builder, arm6, wrist)
        assert _worst(as_action, as_strips) < 1e-5, "strips play differently"

        prefs.job_as_strips = False
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        assert not strips.laid_out(arm6)
        assert arm6.animation_data.action is not None
        again = _played(builder, arm6, wrist)
        assert _worst(as_action, again) < 1e-5, "put back, it plays differently"
        if layered:
            assert [s.name for s in mine.strips] == ["mine"]

    def test_hand_keys_made_since_go_into_the_job(self, arm6, builder, strips, prefs):
        """Keyed by hand after Generate, a key lands in a new active action on top of
        the strips. The next Generate moves it into the job's action, and it plays."""
        import bpy

        prefs.job_as_strips = True
        wrist = _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        first = builder.joint_bones(arm6)[0]
        first.rotation_euler[1] = 0.9
        first.keyframe_insert("rotation_euler", index=1, frame=125)
        hand = arm6.animation_data.action
        assert hand is not None and hand != arm6.kinema_job_action

        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        assert arm6.animation_data.action is None
        bpy.context.scene.frame_set(125)
        assert _q(builder, arm6)[0] == pytest.approx(0.9, abs=1e-5)
        bpy.context.scene.frame_set(50)
        assert wrist.scale[0] != pytest.approx(1.0)


    def test_a_joint_keyed_by_hand_is_read_where_it_plays(
        self, addon, arm6, builder, prefs
    ):
        """Hand keys on top of the strips play over the job's on the same channel,
        so a lookup of a joint's curve -- the Motion Limits check's -- finds the
        hand curve, not the one it hides."""
        import bpy

        ik = importlib.import_module(f"{addon.__name__}.ops.ik")
        prefs.job_as_strips = True
        _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        first = builder.joint_bones(arm6)[0]
        assert ik.joint_curves(arm6)[first.name].id_data == arm6.kinema_job_action
        for frame, value in ((100, 0.0), (101, 1.5)):
            first.rotation_euler[1] = value
            first.keyframe_insert("rotation_euler", index=1, frame=frame)
        hand = arm6.animation_data.action
        assert ik.joint_curves(arm6)[first.name].id_data == hand
        # Over strips that animate the channel, the active action's influence
        # applies: at none, the job plays.
        anim = arm6.animation_data
        anim.action_influence = 0.0
        assert ik.joint_curves(arm6)[first.name].id_data == arm6.kinema_job_action
        # Where the job's strips don't play, the active action plays alone,
        # whatever its influence.
        anim.use_nla = False
        assert ik.joint_curves(arm6)[first.name].id_data == hand
        anim.use_nla = True
        for track in [t for t in anim.nla_tracks if t.name.startswith("Kinema")]:
            track.mute = True
        assert ik.joint_curves(arm6)[first.name].id_data == hand

    def test_a_muted_job_keys_nothing_in_a_fingerprint(
        self, addon, arm6, builder, strips, prefs
    ):
        """What hangs from the robot's bone is fingerprinted by what keys that bone's
        chain. The job's strips do, until their track is muted: then the joints'
        values as they stand place it, and posing one has to show."""
        import bpy

        refs = importlib.import_module(f"{addon.__name__}.ops.references")
        prefs.job_as_strips = True
        _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        first = builder.joint_bones(arm6)[0]
        channel = (first.path_from_id("rotation_euler"), 1)
        _, keyed = refs._animation(arm6, {channel[0]})
        assert channel in keyed, "precondition: the job's strips key the joint"
        _mute_job(strips, arm6)
        _, keyed = refs._animation(arm6, {channel[0]})
        assert channel not in keyed

    def test_hand_keys_onto_a_sampled_job_curve_play(self, arm6, builder, prefs):
        """Keys on a curve of samples never play, so a job curve baked to samples
        becomes keys before hand keys join it, and both play."""
        import bpy

        prefs.job_as_strips = True
        wrist = _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        curve = next(
            c for bag in arm6.kinema_job_action.layers[0].strips[0].channelbags
            for c in bag.fcurves if c.data_path.endswith("scale") and c.array_index == 0
        )
        bpy.context.scene.frame_set(20)
        sampled = wrist.scale[0]
        curve.convert_to_samples(-10, 131)
        wrist.scale[0] = 2.5
        wrist.keyframe_insert("scale", index=0, frame=50)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        bpy.context.scene.frame_set(50)
        assert wrist.scale[0] == pytest.approx(2.5)
        bpy.context.scene.frame_set(20)
        assert wrist.scale[0] == pytest.approx(sampled, abs=1e-4)

    def test_hand_curves_move_with_modifiers_and_samples(self, arm6, builder, prefs):
        """A hand curve that plays as more than its keys -- cycled by a modifier,
        or baked to samples -- plays the same once moved into the job."""
        import bpy

        prefs.job_as_strips = True
        wrist = _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        for frame, value in ((100, 0.5), (110, 1.5)):
            wrist.scale[1] = value
            wrist.keyframe_insert("scale", index=1, frame=frame)
        for frame, value in ((100, 0.7), (104, 1.3)):
            wrist.scale[2] = value
            wrist.keyframe_insert("scale", index=2, frame=frame)
        hand = arm6.animation_data.action
        curves = {
            c.array_index: c for bag in hand.layers[0].strips[0].channelbags
            for c in bag.fcurves
        }
        curves[1].modifiers.new("CYCLES")
        curves[2].convert_to_samples(100, 104)
        frames = (90, 102, 113, 127, 137, 150)

        def scales():
            out = []
            for frame in frames:
                bpy.context.scene.frame_set(frame)
                out.append((wrist.scale[1], wrist.scale[2]))
            return np.array(out)

        on_top = scales()
        assert on_top[3][0] != pytest.approx(1.5), "precondition: it cycles past 110"
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        assert arm6.animation_data.action is None
        assert np.max(np.abs(scales() - on_top)) < 1e-5


def _motion(rig, name):
    return next(m for m in rig.kinema_motions if m.name == name)


def _strip(strips, rig, name):
    return next(s for s in _job_strips(strips, rig) if s.name == name)


class TestRetime:
    def test_a_moved_strip_retimes_its_motion(self, arm6, builder, strips, prefs):
        """A motion's strip moved later in the NLA editor plays the motion there at
        once; the next Generate takes the frames into the Motions list and lays the
        job out at them, playing the same."""
        import bpy

        prefs.job_as_strips = True
        _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        old = {}
        for frame in range(61, 82):
            bpy.context.scene.frame_set(frame)
            old[frame] = _q(builder, arm6)

        _strip(strips, arm6, "back").frame_start_ui = 71
        for frame in range(61, 82):
            bpy.context.scene.frame_set(frame + 10)
            assert np.max(np.abs(_q(builder, arm6) - old[frame])) < 1e-4, f"moved, at {frame}"

        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        back = _motion(arm6, "back")
        assert (back.frame_start, back.frame_end) == (71, 91)
        assert tuple(back.laid_frames) == (71, 91)
        strip = _strip(strips, arm6, "back")
        assert (strip.frame_start, strip.frame_end) == (71.0, 91.0)
        assert (strip.action_frame_start, strip.action_frame_end) == (71.0, 91.0)
        # Regenerated, the move's keys on joint 1 sit nearer its hand key at 120,
        # and a Bezier key's handles follow its neighbours, so in between joint 1
        # bends a little differently. It is where it was at the move's ends, and
        # every other joint plays as it did throughout.
        for frame in range(61, 82):
            bpy.context.scene.frame_set(frame + 10)
            q = _q(builder, arm6)
            assert np.max(np.abs(q[1:] - old[frame][1:])) < 1e-4, f"regenerated, at {frame}"
            if frame in (61, 81):
                assert q[0] == pytest.approx(old[frame][0], abs=1e-4), frame

    def test_any_operator_reading_the_job_catches_up(self, arm6, builder, strips, prefs):
        """Scaled in the NLA editor, a motion's strip retimes it as soon as an
        operator reads the job -- the Motion Check here -- before any Generate."""
        import bpy

        prefs.job_as_strips = True
        _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        strip = _strip(strips, arm6, "back")
        strip.scale = 2.0
        assert (strip.frame_start, strip.frame_end) == (61.0, 101.0), "precondition: scaled"
        bpy.ops.kinema.check_motion()
        back = _motion(arm6, "back")
        assert (back.frame_start, back.frame_end) == (61, 101)

    def test_retimed_in_both_places_the_motions_list_wins(self, arm6, builder, strips, prefs):
        import bpy

        prefs.job_as_strips = True
        _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        _strip(strips, arm6, "back").frame_start_ui = 71
        back = _motion(arm6, "back")
        back.frame_start, back.frame_end = 65, 85
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        assert (back.frame_start, back.frame_end) == (65, 85)
        strip = _strip(strips, arm6, "back")
        assert (strip.frame_start, strip.frame_end) == (65.0, 85.0)

    def test_a_moved_pause_is_put_back(self, arm6, builder, strips, prefs):
        """A pause takes its frames from the motions around it, so its strip moved
        retimes nothing, and the next Generate puts it back."""
        import bpy

        prefs.job_as_strips = True
        _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        frames = {m.name: (m.frame_start, m.frame_end) for m in arm6.kinema_motions}
        _strip(strips, arm6, "hold at b").frame_start_ui = 46
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        assert {m.name: (m.frame_start, m.frame_end) for m in arm6.kinema_motions} == frames
        strip = _strip(strips, arm6, "hold at b")
        assert (strip.frame_start, strip.frame_end) == (41.0, 61.0)

    def test_a_renamed_strip_still_retimes_its_motion(self, arm6, builder, strips, prefs):
        """A motion's strip is known by the frames it shows, not its name."""
        import bpy

        prefs.job_as_strips = True
        _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        strip = _strip(strips, arm6, "back")
        strip.name = "something else"
        strip.frame_start_ui = 71
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        back = _motion(arm6, "back")
        assert (back.frame_start, back.frame_end) == (71, 91)


    def test_a_strip_moved_twice_retimes_twice(self, arm6, builder, strips, prefs):
        """Caught up by the Motion Check, then moved again: the second move is the
        NLA editor's too, not an edit in the Motions list."""
        import bpy

        prefs.job_as_strips = True
        _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        _strip(strips, arm6, "back").frame_start_ui = 71
        bpy.ops.kinema.check_motion()
        back = _motion(arm6, "back")
        assert (back.frame_start, back.frame_end) == (71, 91), "precondition: caught up"
        _strip(strips, arm6, "back").frame_start_ui = 76
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        assert (back.frame_start, back.frame_end) == (76, 96)

    def test_a_strip_moved_back_where_it_was_laid_out_retimes(
        self, arm6, builder, strips, prefs
    ):
        """Caught up, then moved back to the frames it was laid out at: the strip
        plays what its action shows again, and that is a move too."""
        import bpy

        prefs.job_as_strips = True
        _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        _strip(strips, arm6, "back").frame_start_ui = 71
        bpy.ops.kinema.check_motion()
        back = _motion(arm6, "back")
        assert (back.frame_start, back.frame_end) == (71, 91), "precondition: caught up"
        _strip(strips, arm6, "back").frame_start_ui = 61
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        assert (back.frame_start, back.frame_end) == (61, 81)

    def test_a_file_laid_out_before_retiming_still_retimes(
        self, arm6, builder, strips, prefs
    ):
        """A file laid out before motions kept their laid-out frames has none: its
        motions' own frames are taken for them, as they were when it was saved."""
        import bpy

        prefs.job_as_strips = True
        _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        for motion in arm6.kinema_motions:
            motion.laid_frames = motion.synced_frames = (0, 0)
        # Caught up twice before any Generate: the strip is still the motion's
        # once its frames have changed.
        _strip(strips, arm6, "back").frame_start_ui = 71
        bpy.ops.kinema.check_motion()
        back = _motion(arm6, "back")
        assert (back.frame_start, back.frame_end) == (71, 91)
        _strip(strips, arm6, "back").frame_start_ui = 76
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        assert (back.frame_start, back.frame_end) == (76, 96)

    def test_a_motion_added_since_doesnt_take_anothers_strip(
        self, arm6, builder, strips, prefs
    ):
        """A motion added in the list with the frames of one laid out has none laid
        out itself, so the strip there is still the laid-out one's."""
        import bpy

        prefs.job_as_strips = True
        _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        added = arm6.kinema_motions.add()
        added.name, added.frame_start, added.frame_end = "added", 61, 81
        _strip(strips, arm6, "back").frame_start_ui = 71
        _, retimes = strips.catch_up(arm6)
        assert [(m.name, frames) for m, frames, _ in retimes] == [("back", (71, 91))]

    def test_a_refused_retime_changes_nothing(self, arm6, builder, strips, prefs):
        """A strip moved onto another motion's frames: the job is refused for the
        overlap, and the Motions list keeps its frames -- a cancelled operator
        leaves no undo step to take a change back."""
        import bpy

        prefs.job_as_strips = True
        _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        _strip(strips, arm6, "back").frame_start_ui = 35
        assert bpy.ops.kinema.generate_motion() == {"CANCELLED"}
        back = _motion(arm6, "back")
        assert (back.frame_start, back.frame_end) == (61, 81)
        assert tuple(back.synced_frames) == (61, 81)

    def test_a_copied_strip_retimes_nothing(self, arm6, builder, strips, prefs):
        """A strip duplicated in the NLA editor shows what the motion's does, so which
        one is the motion's can't be told: neither retimes it."""
        import bpy

        prefs.job_as_strips = True
        _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        original = _strip(strips, arm6, "back")
        track = next(t for t in strips._job_tracks(arm6) if original.name in t.strips)
        strips._new_strip(
            track, "back copy", 100.0, 120.0, arm6.kinema_job_action, original.action_slot,
            action_start=61.0,
        ).extrapolation = "NOTHING"
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        back = _motion(arm6, "back")
        assert (back.frame_start, back.frame_end) == (61, 81)

    def test_a_job_of_one_motion_keeps_one_motion_track(self, arm6, builder, strips, prefs):
        """Only the motion tracks the job uses, so generating again adds none, and an
        empty one -- no one's -- is never left behind."""
        import bpy

        prefs.job_as_strips = True
        _teach(builder, arm6, 1, Q0, "JOINT", "home")
        _teach(builder, arm6, 21, REACH_A, "JOINT", "a")
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        names = [t.name for t in arm6.animation_data.nla_tracks]
        assert names == [strips.JOB_TRACK, strips.MOTIONS_TRACK], names
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        assert [t.name for t in arm6.animation_data.nla_tracks] == names

    def test_new_motion_tracks_keep_the_job_muted(self, arm6, builder, strips, prefs):
        import bpy

        prefs.job_as_strips = True
        _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        anim = arm6.animation_data
        base, *others = strips._job_tracks(arm6)
        for track in others:
            anim.nla_tracks.remove(track)
        base.mute = True
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        assert [t.mute for t in strips._job_tracks(arm6)] == [True, True, True]

    def test_cover_reads_the_jobs_own_track(self, arm6, builder, strips, prefs):
        """A motion's strip moved out past After holds nothing past itself, so a key
        added in between still needs After stretched to it."""
        import bpy

        prefs.job_as_strips = True
        _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        end = bpy.context.scene.frame_end
        _strip(strips, arm6, "back").frame_start_ui = end + 50
        job = arm6.kinema_job_action
        curve = next(
            c for bag in job.layers[0].strips[0].channelbags for c in bag.fcurves
            if c.data_path.endswith("scale") and c.array_index == 0
        )
        curve.keyframe_points.insert(end + 30, 3.0)
        strips._cover(arm6, strips._slot(arm6))
        base = strips._job_tracks(arm6)[0]
        assert max(s.frame_end for s in base.strips) >= end + 30

    def test_a_bake_over_a_moved_strip_is_refused(self, arm6, builder, strips, prefs):
        """The bake plays a copy of the action in the strips' place, at the frames the
        job was laid out at: with a strip moved since, it asks for Generate first."""
        import bpy

        prefs.job_as_strips = True
        _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        _strip(strips, arm6, "back").frame_start_ui = 71
        with pytest.raises(RuntimeError, match="moved or scaled"):
            bpy.ops.kinema.bake_ik(frame_start=61, frame_end=91)
        assert not _track(arm6, strips.BAKED_TRACK)

    def test_the_job_holds_past_both_its_ends(self, arm6, builder, strips, prefs):
        """With nothing animated outside the job, the job's track still has a strip
        past each end, holding: a motion's strip holds nothing."""
        import bpy

        prefs.job_as_strips = True
        _teach(builder, arm6, 1, Q0, "JOINT", "home")
        _teach(builder, arm6, 21, REACH_A, "JOINT", "a")
        scene = bpy.context.scene
        assert scene.frame_start == 1, "precondition: the job starts with the scene"
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        for inside, outside in ((1, -20), (21, scene.frame_end + 40)):
            scene.frame_set(inside)
            q = _q(builder, arm6)
            scene.frame_set(outside)
            assert np.max(np.abs(_q(builder, arm6) - q)) < 1e-5, outside

    def test_motion_tracks_go_under_the_bakes(self, arm6, builder, strips, prefs):
        """A job laid out on its one track, as before the motions had their own,
        gets theirs straight above it, under the bakes."""
        import bpy

        prefs.job_as_strips = True
        _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        assert "FINISHED" in bpy.ops.kinema.bake_ik(frame_start=1, frame_end=81)
        anim = arm6.animation_data
        for track in strips._job_tracks(arm6)[1:]:
            anim.nla_tracks.remove(track)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        names = [t.name for t in anim.nla_tracks]
        assert names == [
            strips.JOB_TRACK, strips.MOTIONS_TRACK, strips.MOTIONS_TRACK, strips.BAKED_TRACK
        ], names


class TestTweakMode:
    def test_generate_in_tweak_mode_leaves_the_job(self, arm6, builder, strips, prefs):
        """Tweaking a strip of the user's, Generate writes into that strip's action,
        as it did before strips -- and the job's own action keeps its keys."""
        import bpy

        prefs.job_as_strips = True
        _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        job = arm6.kinema_job_action
        before = _keys(job)
        anim = arm6.animation_data
        first = builder.joint_bones(arm6)[0]
        first.rotation_euler[1] = 0.1
        first.keyframe_insert("rotation_euler", index=1, frame=1)
        mine, anim.action = anim.action, None
        track = anim.nla_tracks.new()
        strip = track.strips.new("mine", 1, mine)
        anim.nla_tracks.active = track
        track.select = strip.select = True
        anim.use_tweak_mode = True
        assert anim.use_tweak_mode and anim.action == mine, "precondition: tweaking"

        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        anim.use_tweak_mode = False
        assert _keys(job) == before, "Generate in tweak mode changed the job's keys"


class TestBake:
    def test_bake_goes_on_its_own_track_and_leaves_the_job(
        self, arm6, builder, strips, prefs
    ):
        import bpy

        prefs.job_as_strips = True
        wrist = _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        job = arm6.kinema_job_action
        before = {
            (c.data_path, c.array_index): [tuple(p.co) for p in c.keyframe_points]
            for bag in job.layers[0].strips[0].channelbags for c in bag.fcurves
        }
        played = _played(builder, arm6, wrist)

        assert "FINISHED" in bpy.ops.kinema.bake_ik(frame_start=1, frame_end=81)
        after = {
            (c.data_path, c.array_index): [tuple(p.co) for p in c.keyframe_points]
            for bag in job.layers[0].strips[0].channelbags for c in bag.fcurves
        }
        assert after == before, "the bake changed the job's own keys"
        anim = arm6.animation_data
        assert anim.action is None
        baked = next(t for t in anim.nla_tracks if t.name == strips.BAKED_TRACK)
        assert [(s.frame_start, s.frame_end) for s in baked.strips] == [(1.0, 81.0)]
        mode = arm6.kinema_solver_mode
        arm6.kinema_solver_mode = "OFF"
        assert _worst(played, _played(builder, arm6, wrist)) < 1e-4, "it plays as baked"

        # Muted, the live job plays again, solver and all.
        arm6.kinema_solver_mode = mode
        baked.mute = True
        assert _worst(played, _played(builder, arm6, wrist)) < 1e-4, "the live job is there"

    def test_a_refused_bake_changes_nothing(self, arm6, builder, strips, prefs):
        """Refused, Bake leaves the hand keys where they were and the job as it was:
        a cancelled operator leaves no undo step to take anything back."""
        import bpy

        prefs.job_as_strips = True
        _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        first = builder.joint_bones(arm6)[0]
        first.rotation_euler[1] = 0.9
        first.keyframe_insert("rotation_euler", index=1, frame=125)
        anim = arm6.animation_data
        hand, job = anim.action, arm6.kinema_job_action
        locked = next(
            c for bag in job.layers[0].strips[0].channelbags for c in bag.fcurves
            if c.data_path.endswith("rotation_euler")
        )
        locked.lock = True
        before = _keys(job)

        with pytest.raises(RuntimeError, match="Not baked"):
            bpy.ops.kinema.bake_ik(frame_start=1, frame_end=81)
        assert anim.action == hand
        assert _keys(job) == before
        assert not _track(arm6, strips.BAKED_TRACK)

    def test_hand_keys_past_the_job_still_play_after_a_bake(
        self, arm6, builder, strips, prefs
    ):
        """Bake moves hand keys into the job, as Generate does, so the job's strips
        reach as far as they do -- by strips added at either end. The strips there
        stay as the user left them: one muted, one renamed as Kinema names its own
        outside the job, and the first set not to hold. The job's own last key on
        a subframe still leaves room after it."""
        import bpy

        prefs.job_as_strips = True
        wrist = _job(builder, arm6)
        scene = bpy.context.scene
        wrist.scale[0] = 1.2
        wrist.keyframe_insert("scale", index=0, frame=scene.frame_end + 0.5)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        laid = _job_strips(strips, arm6)
        motion = next(s for s in laid if s.name == "b")
        motion.name, motion.mute = strips.BEFORE + " b", True
        named = next(s for s in laid if s.name == "a")
        named.name = strips.AFTER
        laid[0].extrapolation = "NOTHING"
        laid = [(s.name, s.frame_start, s.frame_end, s.mute, s.extrapolation) for s in laid]
        # Past both of the scene's ends, where the job's strips stop.
        late, early = scene.frame_end + 50, scene.frame_start - 100
        for frame, value in ((late, 3.0), (early, 0.25)):
            wrist.scale[0] = value
            wrist.keyframe_insert("scale", index=0, frame=frame)
        assert laid[-1][2] < late, "precondition: past the strips"
        assert laid[0][1] > early, "precondition: before the strips"
        assert "FINISHED" in bpy.ops.kinema.bake_ik(frame_start=1, frame_end=81)
        assert arm6.animation_data.action is None
        now = [
            (s.name, s.frame_start, s.frame_end, s.mute, s.extrapolation)
            for s in _job_strips(strips, arm6)
        ]
        assert now[1:-1] == laid and len(now) == len(laid) + 2, now
        for frame, value in ((late, 3.0), (early, 0.25)):
            scene.frame_set(frame)
            assert wrist.scale[0] == pytest.approx(value)

    def test_a_sampled_hand_curve_keeps_the_jobs_keys(self, arm6, builder, strips, prefs):
        """Samples and keys don't mix on one curve. A sampled hand curve on a channel
        the job keys becomes keys, and the job's keys outside it stay."""
        import bpy

        prefs.job_as_strips = True
        wrist = _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        for frame, value in ((100, 0.7), (104, 1.3)):
            wrist.scale[0] = value
            wrist.keyframe_insert("scale", index=0, frame=frame)
        hand = arm6.animation_data.action
        next(
            c for bag in hand.layers[0].strips[0].channelbags for c in bag.fcurves
        ).convert_to_samples(100, 104)
        bpy.context.scene.frame_set(102)
        sampled = wrist.scale[0]

        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        curve = next(
            c for bag in arm6.kinema_job_action.layers[0].strips[0].channelbags
            for c in bag.fcurves if c.data_path.endswith("scale") and c.array_index == 0
        )
        frames = [p.co[0] for p in curve.keyframe_points]
        assert -10.0 in frames and 130.0 in frames, frames
        bpy.context.scene.frame_set(102)
        assert wrist.scale[0] == pytest.approx(sampled)

    def test_the_preference_off_still_bakes_onto_a_strip(self, arm6, builder, strips, prefs):
        """The preference says how the next Generate lays the job out; until then a
        job on the NLA is baked as one, and its keys are left alone."""
        import bpy

        prefs.job_as_strips = True
        _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        job = arm6.kinema_job_action
        before = _keys(job)
        prefs.job_as_strips = False
        assert "FINISHED" in bpy.ops.kinema.bake_ik(frame_start=1, frame_end=81)
        assert _keys(job) == before
        assert len(_track(arm6, strips.BAKED_TRACK)[0].strips) == 1

    def test_a_muted_job_track_stays_muted(self, arm6, builder, strips, prefs):
        import bpy

        prefs.job_as_strips = True
        _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        tracks = strips._job_tracks(arm6)
        tracks[1].mute = True
        assert "FINISHED" in bpy.ops.kinema.bake_ik(frame_start=1, frame_end=81)
        assert [t.mute for t in tracks] == [False, True, False]

    def test_rebaking_part_of_a_bake_keeps_the_rest(self, arm6, builder, strips, prefs):
        """A bake over part of another goes above it, and the other plays on either
        side, as the user left it. One before an earlier one fits in beside it, and
        one over all of them leaves none of them, or their actions, behind."""
        import bpy

        prefs.job_as_strips = True
        wrist = _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        played = _played(builder, arm6, wrist)
        assert "FINISHED" in bpy.ops.kinema.bake_ik(frame_start=60, frame_end=80)
        _track(arm6, strips.BAKED_TRACK)[0].strips[0].blend_in = 2.0
        for first, last in ((20, 40), (30, 70)):
            assert "FINISHED" in bpy.ops.kinema.bake_ik(frame_start=first, frame_end=last)
        laid = [
            [(s.frame_start, s.frame_end) for s in t.strips]
            for t in _track(arm6, strips.BAKED_TRACK)
        ]
        assert laid == [[(20.0, 40.0), (60.0, 80.0)], [(30.0, 70.0)]], laid
        lower = _track(arm6, strips.BAKED_TRACK)[0]
        assert lower.strips[1].blend_in == pytest.approx(2.0), "a bake the user set is reset"
        mode = arm6.kinema_solver_mode
        arm6.kinema_solver_mode = "OFF"
        assert _worst(played, _played(builder, arm6, wrist), range(20, 81)) < 1e-4
        arm6.kinema_solver_mode = mode

        old = {s.action.name for t in _track(arm6, strips.BAKED_TRACK) for s in t.strips}
        assert "FINISHED" in bpy.ops.kinema.bake_ik(frame_start=10, frame_end=90)
        laid = [
            [(s.frame_start, s.frame_end) for s in t.strips]
            for t in _track(arm6, strips.BAKED_TRACK)
        ]
        assert laid == [[(10.0, 90.0)]], laid
        assert not old & {a.name for a in bpy.data.actions}, "bakes nothing shows are kept"

    def test_one_frame_onto_a_strip_is_refused(self, arm6, builder, strips, prefs):
        """A strip needs a length, so a one-frame bake is refused before anything
        changes, rather than half way through."""
        import bpy

        prefs.job_as_strips = True
        _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        assert "FINISHED" in bpy.ops.kinema.bake_ik(frame_start=1, frame_end=80)
        with pytest.raises(RuntimeError, match="two frames or more"):
            bpy.ops.kinema.bake_ik(frame_start=30, frame_end=30)
        laid = [[(s.frame_start, s.frame_end) for s in t.strips]
                for t in _track(arm6, strips.BAKED_TRACK)]
        assert laid == [[(1.0, 80.0)]]

    def test_the_checks_see_the_hand_keys_merged(self, arm6, builder, strips, prefs):
        """A hand curve with a modifier, keyed only inside the range, would be
        cleared whole on its own; merged with the job's keys outside the range it
        is kept, modifier and all, and would reshape the bake. The checks run on
        the merged copy, so it is refused -- and the refusal changes nothing."""
        import bpy

        prefs.job_as_strips = True
        _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        first = builder.joint_bones(arm6)[0]
        first.rotation_euler[1] = 0.5
        first.keyframe_insert("rotation_euler", index=1, frame=50)
        anim = arm6.animation_data
        hand, job = anim.action, arm6.kinema_job_action
        curve = next(
            c for bag in hand.layers[0].strips[0].channelbags for c in bag.fcurves
        )
        curve.modifiers.new("NOISE")
        before = _keys(job)

        with pytest.raises(RuntimeError, match="with modifiers"):
            bpy.ops.kinema.bake_ik(frame_start=1, frame_end=81)
        assert anim.action == hand
        assert _keys(job) == before

    def test_regenerating_mutes_the_bake_it_replaces(self, arm6, builder, strips, prefs):
        """A bake shows the job as it was baked; once the job is generated again it
        no longer does, so it is muted rather than left playing over it."""
        import bpy

        prefs.job_as_strips = True
        _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        assert "FINISHED" in bpy.ops.kinema.bake_ik(frame_start=1, frame_end=81)
        baked = _track(arm6, strips.BAKED_TRACK)[0]
        assert not baked.strips[0].mute
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        assert baked.strips[0].mute

    def test_a_bake_never_goes_onto_a_muted_track(self, arm6, builder, strips, prefs):
        """Muted, a track of bakes plays the live job; a new bake beside the muted
        one goes on a track that plays, and the muted one stays as it was."""
        import bpy

        prefs.job_as_strips = True
        _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        assert "FINISHED" in bpy.ops.kinema.bake_ik(frame_start=1, frame_end=30)
        _track(arm6, strips.BAKED_TRACK)[0].mute = True
        assert "FINISHED" in bpy.ops.kinema.bake_ik(frame_start=60, frame_end=80)
        laid = [
            (t.mute, [(s.frame_start, s.frame_end) for s in t.strips])
            for t in _track(arm6, strips.BAKED_TRACK)
        ]
        assert laid == [(True, [(1.0, 30.0)]), (False, [(60.0, 80.0)])], laid

    def test_strips_sharing_a_bake_are_cleared_once(self, arm6, builder, strips, prefs):
        """A baked strip duplicated in the NLA editor shares its action. Both under a
        new bake, they go, and the action with them, once."""
        import bpy

        prefs.job_as_strips = True
        _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        assert "FINISHED" in bpy.ops.kinema.bake_ik(frame_start=1, frame_end=30)
        track = _track(arm6, strips.BAKED_TRACK)[0]
        shared = track.strips[0].action
        name = shared.name
        copy = strips._new_strip(track, "copy", 40.0, 50.0, shared, track.strips[0].action_slot)
        copy.extrapolation = "NOTHING"  # as a duplicate of the bake has it
        assert "FINISHED" in bpy.ops.kinema.bake_ik(frame_start=1, frame_end=60)
        laid = [[(s.frame_start, s.frame_end) for s in t.strips]
                for t in _track(arm6, strips.BAKED_TRACK)]
        assert laid == [[(1.0, 60.0)]], laid
        assert name not in bpy.data.actions

    def test_a_bake_that_holds_past_its_frames_stays(self, arm6, builder, strips, prefs):
        """An earlier bake set to hold past its frames plays outside them, so one
        wholly under a new bake is kept, still holding there."""
        import bpy

        prefs.job_as_strips = True
        _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        assert "FINISHED" in bpy.ops.kinema.bake_ik(frame_start=20, frame_end=40)
        _track(arm6, strips.BAKED_TRACK)[0].strips[0].extrapolation = "HOLD_FORWARD"
        assert "FINISHED" in bpy.ops.kinema.bake_ik(frame_start=10, frame_end=50)
        laid = [[(s.frame_start, s.frame_end, s.extrapolation) for s in t.strips]
                for t in _track(arm6, strips.BAKED_TRACK)]
        assert laid == [[(20.0, 40.0, "HOLD_FORWARD")], [(10.0, 50.0, "NOTHING")]], laid

    @pytest.mark.parametrize("state", ["nla-off", "soloed"])
    def test_a_bake_that_wouldnt_play_is_refused(self, arm6, builder, strips, prefs, state):
        """With the NLA off the strip wouldn't play, and with a track soloed the copy
        being baked wouldn't either: refused before anything changes."""
        import bpy

        prefs.job_as_strips = True
        _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        first = builder.joint_bones(arm6)[0]
        first.rotation_euler[1] = 0.9
        first.keyframe_insert("rotation_euler", index=1, frame=125)
        anim = arm6.animation_data
        hand, job = anim.action, arm6.kinema_job_action
        before = _keys(job)
        if state == "nla-off":
            anim.use_nla = False
        else:
            _track(arm6, strips.JOB_TRACK)[0].is_solo = True

        with pytest.raises(RuntimeError, match="NLA is off" if state == "nla-off" else "soloed"):
            bpy.ops.kinema.bake_ik(frame_start=1, frame_end=81)
        assert anim.action == hand
        assert _keys(job) == before
        assert not _track(arm6, strips.BAKED_TRACK)

    def test_keys_past_a_manual_frame_range_are_covered(self, arm6, builder, strips, prefs):
        """A manual frame range on the job's action doesn't stop keys past it from
        playing, so the strips reach them too."""
        import bpy

        prefs.job_as_strips = True
        wrist = _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        job = arm6.kinema_job_action
        job.use_frame_range = True
        job.frame_start, job.frame_end = 1, 81
        frame = bpy.context.scene.frame_end + 50
        wrist.scale[0] = 3.0
        wrist.keyframe_insert("scale", index=0, frame=frame)
        assert "FINISHED" in bpy.ops.kinema.bake_ik(frame_start=1, frame_end=81)
        bpy.context.scene.frame_set(frame)
        assert wrist.scale[0] == pytest.approx(3.0)

    def test_a_bake_past_the_jobs_keys_holds_them(self, arm6, builder, strips, prefs):
        """Past the job's last key its strips hold, over a track of the user's under
        them. The copy being baked holds too, whatever the active action's
        extrapolation is set to, so those frames bake what plays live."""
        import bpy

        first = builder.joint_bones(arm6)[0]
        first.rotation_euler[1] = 0.3
        first.keyframe_insert("rotation_euler", index=1, frame=1)
        anim = arm6.animation_data
        base, anim.action = anim.action, None
        anim.nla_tracks.new().strips.new("mine", 1, base)
        prefs.job_as_strips = True
        _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        frames = range(135, 161)
        live = {}
        for frame in frames:
            bpy.context.scene.frame_set(frame)
            live[frame] = _q(builder, arm6)[0]
        assert live[140] != pytest.approx(0.3), "precondition: the job holds over the track"
        anim.action_extrapolation = "NOTHING"
        assert "FINISHED" in bpy.ops.kinema.bake_ik(frame_start=120, frame_end=160)
        assert anim.action_extrapolation == "NOTHING"
        arm6.kinema_solver_mode = "OFF"
        for frame in frames:
            bpy.context.scene.frame_set(frame)
            assert _q(builder, arm6)[0] == pytest.approx(live[frame], abs=1e-4), frame
