"""The job as NLA strips: one per motion, all showing one action.

Generate Motion writes the job's keys into one action, as it always has, and
this lays that action out on the rig's NLA as strips: one per motion -- a hold
between two motions, and a joint move bridging a gap, count as motions here --
named after it and covering its frames, and a "Before" and an "After" strip for
whatever the action holds outside the job, hand keys included. They tile the
timeline end to end and all show the same action, each its own frames, so the
rig plays exactly as one action would: every channel, at every frame.

Why one action and not one per motion: a channel only the hand keys animate --
a gripper finger -- would otherwise drop to its default during each motion,
since a frame plays one strip per track. And why the last strip holds forward:
past a track's last strip, a channel it animates drops to its default too, so
the robot would snap to its rest pose after the job.

Keys are written with the action made active again and the NLA off, which plays
the same: :func:`editing`. Hand keys made in the active action since the last
Generate are moved into the job's action then, so nothing plays on top of it.
Bake IK writes into a copy, shown as a strip on a track above over the baked
range: :func:`baking`. It bakes what plays, the user's own tracks included, and
muting its track goes back to the live job.

The add-on preference "Job as NLA Strips" turns all of this off. A rig laid out
as strips is then put back in its active action on the next Generate.
"""

from __future__ import annotations

from contextlib import contextmanager

import bpy

#: The tracks Kinema owns. Strips on them are replaced on every Generate.
JOB_TRACK = "Kinema Job"
BAKED_TRACK = "Kinema Baked"
BEFORE, AFTER = "Before", "After"
#: Marks an action as a rig's job (its value is the rig's name), and as a bake.
PROP_JOB = "kinema_job"
PROP_BAKED = "kinema_baked"


def enabled(context=None) -> bool:
    """Whether jobs are laid out as NLA strips: the add-on preference, on by default."""
    from ..prefs import get_prefs

    prefs = get_prefs(context)
    return bool(getattr(prefs, "job_as_strips", True)) if prefs is not None else True


def _track(anim, name):
    return next((t for t in anim.nla_tracks if t.name == name), None)


def job_action(rig):
    """The action holding ``rig``'s job, laid out as strips or not; None if none."""
    action = getattr(rig, "kinema_job_action", None)
    if action is not None:
        return action
    anim = rig.animation_data
    return anim.action if anim is not None else None


def laid_out(rig) -> bool:
    """Whether ``rig``'s job is on its NLA as strips now."""
    anim = rig.animation_data
    track = _track(anim, JOB_TRACK) if anim is not None else None
    return track is not None and len(track.strips) > 0


def _slot(rig):
    """The slot the job's strips show, or the active one."""
    anim = rig.animation_data
    track = _track(anim, JOB_TRACK) if anim is not None else None
    if track is not None and len(track.strips):
        slot = getattr(track.strips[0], "action_slot", None)
        if slot is not None:
            return slot
    return getattr(anim, "action_slot", None) if anim is not None else None


def _curves(action, slot):
    """The F-curve collections of ``action`` for ``slot``."""
    from .ik import _iter_fcurve_containers

    handle = getattr(slot, "handle", None)
    return list(_iter_fcurve_containers(action, handle))


def _merge(source, source_slot, target, slot) -> None:
    """Copy ``source``'s curves for ``source_slot`` into ``target``'s for ``slot``, key
    by key: a key at the same frame on the same channel replaces the one there."""
    bag = None
    if getattr(target, "layers", None):
        bag = target.layers[0].strips[0].channelbag(slot, ensure=True)
    destinations = bag.fcurves if bag is not None else target.fcurves
    for container in _curves(source, source_slot):
        for curve in list(container):
            into = destinations.find(curve.data_path, index=curve.array_index)
            if into is None:
                group = curve.group.name if curve.group is not None else ""
                into = destinations.new(curve.data_path, index=curve.array_index,
                                        group_name=group)
            for point in curve.keyframe_points:
                frame = float(point.co[0])
                for old in [k for k in into.keyframe_points if abs(k.co[0] - frame) < 1e-4]:
                    into.keyframe_points.remove(old)
                added = into.keyframe_points.insert(frame, point.co[1], options={"FAST"})
                added.interpolation = point.interpolation
                added.easing = point.easing
                added.handle_left_type, added.handle_right_type = "FREE", "FREE"
                added.handle_left = point.handle_left
                added.handle_right = point.handle_right
            into.update()


def _same_slot(action, slot):
    """``action``'s slot with ``slot``'s identifier, or None."""
    identifier = getattr(slot, "identifier", None)
    if identifier is None or not getattr(action, "slots", None):
        return None
    return next((s for s in action.slots if s.identifier == identifier), None)


def _activate(rig, action, slot) -> None:
    anim = rig.animation_data
    anim.action = action
    if slot is not None and getattr(anim, "action_slot", None) != slot:
        try:
            anim.action_slot = slot
        except (AttributeError, TypeError, RuntimeError):
            pass


@contextmanager
def editing(rig, context=None):
    """The job's action active and the NLA off, to write keys into; laid out after.

    Yields whether it was laid out as strips. With the preference off, or the rig
    in NLA tweak mode, the active action is what is written, as before strips --
    and a rig laid out as strips is first put back in its active action.
    """
    anim = rig.animation_data_create()
    if anim.use_tweak_mode:
        yield False
        return
    job = rig.kinema_job_action
    slot = _slot(rig)
    if not enabled(context):
        if job is not None:
            # Back to the active action, as it was before strips.
            _clear_tracks(anim)
            hand, hand_slot = anim.action, getattr(anim, "action_slot", None)
            _activate(rig, job, slot)
            if hand is not None and hand != job:
                _merge(hand, hand_slot, job, slot)
            rig.kinema_job_action = None
            anim.use_nla = True
        yield False
        return
    hand, hand_slot = anim.action, getattr(anim, "action_slot", None)
    if job is None:
        # The first time: the active action, job and hand keys alike, is the job's.
        job = hand or bpy.data.actions.new(f"{rig.name} Job")
        rig.kinema_job_action = job
        hand = None
    job[PROP_JOB] = rig.name
    _activate(rig, job, slot or hand_slot)
    slot = getattr(anim, "action_slot", None) or slot
    if hand is not None and hand != job:
        # Keyed by hand since the last Generate: into the job, so it plays. The
        # hand action itself is left unassigned, not deleted.
        _merge(hand, hand_slot, job, slot)
    anim.use_nla = False
    try:
        yield True
    finally:
        anim.use_nla = True


def _clear_tracks(anim) -> None:
    for name in (JOB_TRACK,):
        track = _track(anim, name)
        if track is not None:
            anim.nla_tracks.remove(track)


def lay_out(rig, plan) -> None:
    """Lay the job's action out as strips, one per span of ``plan``, and Before and
    After for what the action holds outside the job. The active action is then
    left empty. Does nothing with the preference off or in NLA tweak mode."""
    anim = rig.animation_data
    job = getattr(rig, "kinema_job_action", None)
    if anim is None or job is None or anim.use_tweak_mode or not enabled() or not plan:
        return
    slot = getattr(anim, "action_slot", None) if anim.action == job else _slot(rig)
    track = _track(anim, JOB_TRACK)
    if track is None:
        track = anim.nla_tracks.new()
        track.name = JOB_TRACK
    for strip in list(track.strips):
        track.strips.remove(strip)
    first, last = int(plan[0].start_frame), int(plan[-1].end_frame)
    start, end = (float(v) for v in job.frame_range)
    tiles = []
    if start < first:
        tiles.append((BEFORE, start, float(first)))
    for span in plan:
        if span.end_frame > span.start_frame:
            tiles.append((span.label, float(span.start_frame), float(span.end_frame)))
    if end > last:
        tiles.append((AFTER, float(last), end))
    anim.action = None
    for index, (name, a, b) in enumerate(tiles):
        strip = track.strips.new(name, int(a), job)
        if slot is not None and hasattr(strip, "action_slot"):
            strip.action_slot = slot
        strip.use_sync_length = False
        strip.action_frame_start = a
        strip.action_frame_end = b
        strip.frame_start_ui = a
        strip.scale = 1.0
        strip.blend_type = "REPLACE"
        strip.extrapolation = "HOLD" if index == 0 else "HOLD_FORWARD"
    anim.use_nla = True


class Bake:
    """What :func:`baking` yields: ``into_strip`` says whether the bake goes into a
    copy of the job's action, and the bake sets ``keep`` once it has finished, so
    a refused one leaves no strip."""

    def __init__(self, into_strip: bool):
        self.into_strip, self.keep = into_strip, False


@contextmanager
def baking(rig, first: int, last: int, context=None):
    """Bake IK into a copy of the job's action, shown over ``first``-``last`` on a
    track above the job. With no job laid out as strips, Bake keys the active
    action as it always has: ``into_strip`` is False."""
    anim = rig.animation_data
    if anim is None or anim.use_tweak_mode or not enabled(context) or not laid_out(rig):
        yield Bake(False)
        return
    job = rig.kinema_job_action
    slot = _slot(rig)
    hand, hand_slot = anim.action, getattr(anim, "action_slot", None)
    if hand is not None and hand != job:
        _merge(hand, hand_slot, job, slot)
    baked = job.copy()
    baked.name = f"{rig.name} Baked"
    if PROP_JOB in baked:
        del baked[PROP_JOB]
    baked[PROP_BAKED] = rig.name
    # A copy has slots of its own: the one standing for the same rig.
    slot = _same_slot(baked, slot)
    # The copy plays in the job's place: the job's track muted, the copy active
    # on top, at full strength. The NLA stays on, so any track of the user's
    # still plays under it, and what is baked is what plays.
    job_track = _track(anim, JOB_TRACK)
    job_track.mute = True
    blend, influence = anim.action_blend_type, anim.action_influence
    anim.action_blend_type, anim.action_influence = "REPLACE", 1.0
    _activate(rig, baked, slot)
    state = Bake(True)
    try:
        yield state
    finally:
        anim.action = None
        anim.action_blend_type, anim.action_influence = blend, influence
        job_track.mute = False
        if not state.keep:
            bpy.data.actions.remove(baked)
    if not state.keep:
        return
    track = _track(anim, BAKED_TRACK)
    if track is None:
        track = anim.nla_tracks.new()
        track.name = BAKED_TRACK
    for strip in list(track.strips):
        # A bake over the same frames replaces the one there.
        if strip.frame_start < last and strip.frame_end > first:
            track.strips.remove(strip)
    strip = track.strips.new(f"Baked {first}-{last}", int(first), baked)
    if slot is not None and hasattr(strip, "action_slot"):
        strip.action_slot = slot
    strip.use_sync_length = False
    strip.action_frame_start = float(first)
    strip.action_frame_end = float(last)
    strip.frame_start_ui = float(first)
    strip.blend_type = "REPLACE"
    strip.extrapolation = "NOTHING"
