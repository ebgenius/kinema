"""The job as NLA strips (#96): ops/strips.py. Needs a real ``bpy``.

The claims:

* Generate Motion lays the job out on a "Kinema Job" track, one strip per motion
  -- holds and bridging joint moves included -- tiling its frames, with Before
  and After strips for what the action holds outside the job;
* it plays exactly as the job in one active action did: every joint, every
  frame, hand keys before and after the job, and a channel only hand keys
  animate, which a strip per motion would drop to its default;
* hand keys made in the active action since are moved into the job, so nothing
  plays on top of it;
* the preference off puts the job back in the active action;
* Bake IK writes a "Kinema Baked" strip over the baked range and leaves the job's
  action as it was; muting that track goes back to the live job.
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


def _worst(a: dict, b: dict) -> float:
    return max(float(np.max(np.abs(a[f] - b[f]))) for f in a)


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
        track = next(t for t in anim.nla_tracks if t.name == strips.JOB_TRACK)
        laid = [(s.name, s.frame_start, s.frame_end) for s in track.strips]
        names = [name for name, _, _ in laid]
        assert names[0] == strips.BEFORE and names[-1] == strips.AFTER, names
        # Each motion by its name, and the pause at b the plan adds, by its own.
        assert names[1:-1] == ["a", "b", "hold at b", "back"], names
        assert all(s.action == job for s in track.strips)
        for (_, _, end), (_, start, _) in zip(laid, laid[1:], strict=False):
            assert end == start, f"a gap or overlap at {end} / {start}: {laid}"
        assert (laid[1][1], laid[-2][2]) == (1.0, 81.0)
        assert track.strips[0].extrapolation == "HOLD"
        assert track.strips[-1].extrapolation == "HOLD_FORWARD"

    def test_regenerating_replaces_the_strips(self, arm6, builder, strips, prefs):
        import bpy

        prefs.job_as_strips = True
        _job(builder, arm6)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        count = len(next(t for t in arm6.animation_data.nla_tracks
                         if t.name == strips.JOB_TRACK).strips)
        assert "FINISHED" in bpy.ops.kinema.generate_motion()
        tracks = [t for t in arm6.animation_data.nla_tracks if t.name == strips.JOB_TRACK]
        assert len(tracks) == 1 and len(tracks[0].strips) == count


class TestPlayback:
    def test_strips_play_as_the_active_action_did_and_back(
        self, arm6, builder, strips, prefs
    ):
        """The same job generated into the active action, then laid out as strips,
        then put back: the same on every frame, hand keys and all."""
        import bpy

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
