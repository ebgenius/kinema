"""The job as NLA strips: one per motion, all showing one action.

Generate Motion writes the job's keys into one action, as it always has, and
this lays that action out on the rig's NLA as strips, all showing that action,
each its own frames:

* one per motion, named after it and covering its frames, on two tracks of
  their own, "Kinema Motions", taking turns -- so any one can be moved or
  scaled in the NLA editor without running into the next, which a strip on
  the same track refuses to do;
* under them, on the "Kinema Job" track, one for each pause between motions
  and each joint move bridging a gap, and a "Before" and an "After" for
  whatever the action holds outside the job, hand keys included, out to the
  scene's ends at least.

With nothing moved, every frame is played by one strip, so the rig plays
exactly as one action would: every channel, at every frame of the scene. They
replace what is under them at full strength, as the active action plays any
channel the tracks under it don't animate, whatever its blend and influence.

Why one action and not one per motion: a channel only the hand keys animate --
a gripper finger -- would otherwise drop to its default during each motion,
since a frame plays one strip per track. Why the motions hold nothing past
their frames: the track under them plays there. And why the job's track
holds forward, with a strip out past each end of the job: past a track's last
strip, a channel it animates drops to its default, so the robot would snap to
its rest pose.

A motion's strip moved or scaled in the NLA editor retimes the motion: the
next operator that reads the job takes its new frames into the Motions list
(:func:`catch_up`), and the next Generate lays the job out at them. Until then
the stretch a strip left plays the job's track under it, which holds the pose
before.

Keys are written with the action made active again and the NLA off, which plays
the same: :func:`editing`. Hand keys made in the active action since the last
Generate are moved into the job's action then, so nothing plays on top of it.
Bake IK writes into a copy, shown as a strip on a track above over the baked
range: :func:`baking`. It bakes what plays, the user's own tracks included, and
muting its track goes back to the live job. A bake over an earlier one goes
above it, and regenerating mutes the bakes over the job's frames, which no
longer show it.

The tracks are known by what their strips show -- the job's action, or a bake --
not by their names, so a track of the user's with the same name is left alone.
A motion's strip is known by the frames of the action it shows, which are the
frames its motion was laid out at.

The add-on preference "Job as NLA Strips" turns all of this off. A rig laid out
as strips is then put back in its active action on the next Generate.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import bpy

#: The names Kinema gives its tracks.
JOB_TRACK = "Kinema Job"
MOTIONS_TRACK = "Kinema Motions"
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


def _job_tracks(rig) -> list:
    """The tracks of ``rig``'s job, bottom to top: those whose strips all show its
    action. The job's own track comes first, the motions' above it."""
    job = getattr(rig, "kinema_job_action", None)
    anim = rig.animation_data
    if job is None or anim is None:
        return []
    return [
        t for t in anim.nla_tracks
        if len(t.strips) and all(s.action == job for s in t.strips)
    ]


def _plays(anim, track) -> bool:
    """Whether ``track`` plays: the NLA on, and it unmuted -- or, with a track
    soloed, it the soloed one, muted or not."""
    soloed = any(t.is_solo for t in anim.nla_tracks)
    return anim.use_nla and (track.is_solo if soloed else not track.mute)


def job_plays(rig) -> bool:
    """Whether any of the job's strips play now."""
    anim = rig.animation_data
    return any(_plays(anim, track) for track in _job_tracks(rig))


def _baked_tracks(rig) -> list:
    """Every track of bakes, bottom to top: a bake over another goes on a track of
    its own above it."""
    anim = rig.animation_data
    if anim is None:
        return []
    return [
        t for t in anim.nla_tracks
        if len(t.strips) and all(s.action is not None and PROP_BAKED in s.action for s in t.strips)
    ]


def _new_track(anim, name):
    track = anim.nla_tracks.new()
    track.name = name
    return track


def job_action(rig):
    """The action holding ``rig``'s job, laid out as strips or not; None if none."""
    action = getattr(rig, "kinema_job_action", None)
    if action is not None:
        return action
    anim = rig.animation_data
    return anim.action if anim is not None else None


def laid_out(rig) -> bool:
    """Whether ``rig``'s job is on its NLA as strips now."""
    return bool(_job_tracks(rig))


def _slot(rig):
    """The slot the job's strips show, or the active one while the job is the
    active action; None if neither. A slot belongs to one action, so the active
    one of any other action is no answer."""
    for track in _job_tracks(rig)[:1]:
        slot = getattr(track.strips[0], "action_slot", None)
        if slot is not None:
            return slot
    anim = rig.animation_data
    job = getattr(rig, "kinema_job_action", None)
    if anim is not None and job is not None and anim.action == job:
        return getattr(anim, "action_slot", None)
    return None


def _curves(action, slot):
    """The F-curve collections of ``action`` for ``slot``. A slotted action with
    no slot animates nothing here, so it has none."""
    from .ik import _iter_fcurve_containers

    handle = getattr(slot, "handle", None)
    if handle is None and getattr(action, "layers", None):
        return []
    return list(_iter_fcurve_containers(action, handle))


def _key(point) -> tuple:
    """A keyframe, as plain values that outlive its curve."""
    return (
        float(point.co[0]), float(point.co[1]), point.interpolation, point.easing,
        tuple(point.handle_left), tuple(point.handle_right),
    )


def _put(curve, key) -> None:
    """Key ``curve`` with ``key``, replacing a key at the same frame."""
    frame, value, interpolation, easing, left, right = key
    for old in [k for k in curve.keyframe_points if abs(k.co[0] - frame) < 1e-4]:
        curve.keyframe_points.remove(old)
    added = curve.keyframe_points.insert(frame, value, options={"FAST"})
    added.interpolation = interpolation
    added.easing = easing
    added.handle_left_type, added.handle_right_type = "FREE", "FREE"
    added.handle_left = left
    added.handle_right = right


def _group(bag, curve, name) -> None:
    if bag is None or not name or curve.group is not None:
        return
    group = bag.groups.get(name) if hasattr(bag.groups, "get") else None
    if group is None:
        group = bag.groups.new(name)
    try:
        curve.group = group
    except (AttributeError, TypeError, RuntimeError):
        pass


def _merge(source, source_slot, target, slot) -> None:
    """Move ``source``'s curves for ``source_slot`` into ``target``'s for ``slot``.

    Key by key: a key at the same frame on the same channel replaces the one
    there, as keying the one action would. A curve ``target`` has nothing on, or
    one that plays as more than its keys -- modifiers, or samples in place of
    keys -- comes whole, with its own extrapolation, the target's keys added
    back between its own where it has keys to add them to.
    """
    bag = None
    if getattr(target, "layers", None):
        if slot is None:
            return
        bag = target.layers[0].strips[0].channelbag(slot, ensure=True)
    destinations = bag.fcurves if bag is not None else target.fcurves
    for container in _curves(source, source_slot):
        for curve in list(container):
            into = destinations.find(curve.data_path, index=curve.array_index)
            if into is not None and not len(into.keyframe_points) and len(into.sampled_points):
                # Keys on a curve of samples never play, so the target's samples
                # become keys, one a frame, before any join them.
                frames = [p.co[0] for p in into.sampled_points]
                into.convert_to_keyframes(int(min(frames)), int(max(frames)) + 1)
            group = curve.group.name if curve.group is not None else ""
            whole = into is None or len(curve.modifiers) or not len(curve.keyframe_points)
            if whole and hasattr(destinations, "new_from_fcurve"):
                kept = []
                if into is not None:
                    group = into.group.name if into.group is not None else group
                    kept = [_key(p) for p in into.keyframe_points]
                    destinations.remove(into)
                into = destinations.new_from_fcurve(curve)
                _group(bag, into, group)
                if kept and not len(into.keyframe_points) and len(into.sampled_points):
                    # Samples and keys don't mix on one curve: the samples become
                    # keys, one a frame, so the target's keys can join them.
                    frames = [p.co[0] for p in into.sampled_points]
                    into.convert_to_keyframes(int(min(frames)), int(max(frames)) + 1)
                if len(into.keyframe_points):
                    taken = [float(p.co[0]) for p in into.keyframe_points]
                    for key in kept:
                        if all(abs(key[0] - f) >= 1e-4 for f in taken):
                            _put(into, key)
            else:
                if into is None:
                    into = destinations.new(
                        curve.data_path, index=curve.array_index, group_name=group
                    )
                for point in curve.keyframe_points:
                    _put(into, _key(point))
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
            # Back to the active action, as it was before strips. The NLA
            # switch is left as it is: the active action plays either way.
            for track in _job_tracks(rig):
                anim.nla_tracks.remove(track)
            hand, hand_slot = anim.action, getattr(anim, "action_slot", None)
            _activate(rig, job, slot)
            slot = getattr(anim, "action_slot", None) or slot
            if hand is not None and hand != job:
                _merge(hand, hand_slot, job, slot)
            rig.kinema_job_action = None
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
    was_nla = anim.use_nla
    anim.use_nla = False
    try:
        yield True
    finally:
        # Strips play only with the NLA on; with none laid out, as it was.
        anim.use_nla = True if laid_out(rig) else was_nla


def _new_strip(track, name, a: float, b: float, action, slot, action_start=None):
    """A strip on ``track`` over frames ``a``-``b``, showing ``action`` from
    ``action_start`` (``a`` unless given), at full strength.

    Blender gives a new strip its action's frame range and refuses it where that
    would overlap a neighbour, so the action's range is set to the strip's for
    the moment it goes in, and put back after.
    """
    lo = a if action_start is None else action_start
    hi = lo + (b - a)
    saved = (action.use_frame_range, action.frame_start, action.frame_end)
    action.use_frame_range = True
    # End, start, end: either one alone could be refused for passing the other.
    action.frame_end, action.frame_start, action.frame_end = hi, lo, hi
    try:
        strip = track.strips.new(name, int(a), action)
    finally:
        action.frame_end, action.frame_start, action.frame_end = saved[2], saved[1], saved[2]
        action.use_frame_range = saved[0]
    if slot is not None and hasattr(strip, "action_slot"):
        strip.action_slot = slot
    strip.use_sync_length = False
    strip.action_frame_start, strip.action_frame_end = lo, hi
    strip.frame_start_ui = a
    strip.scale = 1.0
    strip.blend_type = "REPLACE"
    return strip


def _reach(job) -> tuple[float, float]:
    """The frames the job's strips cover: what its action holds, and out to the
    scene's ends too -- a curve can play on past its keys, cycled or extrapolated
    in a line, and past a strip's frames it would hold.

    Out to whole frames: a strip goes in at a whole frame before it is moved to
    its own, so one that starts where another ends on a subframe would first
    land inside it, and be refused.
    """
    # Its keys as well as its range: a manual range doesn't stop the keys past it
    # from playing.
    keys, shown = job.curve_frame_range, job.frame_range
    start, end = min(float(keys[0]), float(shown[0])), max(float(keys[1]), float(shown[1]))
    scene = bpy.context.scene
    if scene is not None:
        start, end = min(start, float(scene.frame_start)), max(end, float(scene.frame_end))
    return float(math.floor(start)), float(math.ceil(end))


def _tracks_for_job(rig) -> tuple:
    """The job's track and the motions' two, emptied: the ones there are, kept
    where they are among the rig's tracks, and the missing ones made above them."""
    anim = rig.animation_data
    tracks = _job_tracks(rig)
    for track in tracks[3:]:
        anim.nla_tracks.remove(track)
    tracks = tracks[:3]
    for track in tracks:
        for strip in list(track.strips):
            track.strips.remove(strip)
    while len(tracks) < 3:
        # Above the job's last track, so the bakes and the user's tracks keep
        # their place above or under the job.
        track = anim.nla_tracks.new(prev=tracks[-1]) if tracks else anim.nla_tracks.new()
        track.name = JOB_TRACK if not tracks else MOTIONS_TRACK
        tracks.append(track)
    return tuple(tracks)


def _tile(rig, spans, slot) -> None:
    """The job laid out: each motion of ``spans`` -- ``(name, start, end, is a
    motion)``, in frame order -- on the motions' tracks, taking turns, and the
    rest on the job's track, with a Before and an After out past what the
    action holds, a frame past the job's ends at least. The active action is
    then left empty."""
    anim = rig.animation_data
    job = rig.kinema_job_action
    base, *motions = _tracks_for_job(rig)
    start, end = _reach(job)
    first, last = spans[0][1], spans[-1][2]
    # A frame past each end at least, so the job's track holds back before the
    # job and forward after it -- a motion's strip holds nothing.
    start, end = min(start, first - 1.0), max(end, last + 1.0)
    fills = [(BEFORE, start, first)]
    fills += [(name, a, b) for name, a, b, motion in spans if not motion]
    fills.append((AFTER, last, end))
    anim.action = None
    for index, (name, a, b) in enumerate(fills):
        strip = _new_strip(base, name, a, b, job, slot)
        strip.extrapolation = "HOLD" if index == 0 else "HOLD_FORWARD"
    for index, (name, a, b) in enumerate(
        (name, a, b) for name, a, b, motion in spans if motion
    ):
        strip = _new_strip(motions[index % 2], name, a, b, job, slot)
        strip.extrapolation = "NOTHING"
    anim.use_nla = True


def lay_out(rig, plan) -> int:
    """Lay the job's action out as strips, one per span of ``plan``, and Before and
    After for what the action holds outside the job. The active action is then
    left empty. Does nothing with the preference off or in NLA tweak mode.

    A bake over the job's frames shows the job as it was, so it is muted: the
    number muted is returned, for the caller to say so.
    """
    anim = rig.animation_data
    job = getattr(rig, "kinema_job_action", None)
    if anim is None or job is None or anim.use_tweak_mode or not enabled() or not plan:
        return 0
    from ..rig.waypoints import MADE_MOTION

    slot = getattr(anim, "action_slot", None) if anim.action == job else _slot(rig)
    spans = [
        (span.label, float(span.start_frame), float(span.end_frame), span.made == MADE_MOTION)
        for span in plan if span.end_frame > span.start_frame
    ]
    if not spans:
        return 0
    _tile(rig, spans, slot)
    for motion in getattr(rig, "kinema_motions", ()):
        # What its strip shows of the action: how catch_up knows it.
        motion.laid_frames = (motion.frame_start, motion.frame_end)
    first, last = spans[0][1], spans[-1][2]
    muted = 0
    for strip in (s for t in _baked_tracks(rig) for s in t.strips):
        if not strip.mute and strip.frame_start <= last and strip.frame_end >= first:
            strip.mute = True
            muted += 1
    return muted


def muted_text(count: int) -> str:
    """What a report says about the bakes :func:`lay_out` muted."""
    several = count != 1
    return (
        f"Muted {count} baked strip{'s' if several else ''} that no longer "
        f"show{'' if several else 's'} the job; unmute {'them' if several else 'it'} "
        f"in the NLA editor to play the old bake"
    )


def _cover(rig, slot) -> None:
    """Strips added at either end, where what the job's action holds now reaches
    past its strips: hand keys moved into it can. The strips there are left as
    they are, however the user has set them."""
    tracks = _job_tracks(rig)
    track = tracks[0]
    job = rig.kinema_job_action
    start, end = _reach(job)
    every = [s for t in tracks for s in t.strips]
    latest = max(float(s.frame_end) for s in every)
    laid = sorted(track.strips, key=lambda s: s.frame_start)
    first = laid[0]
    if end > latest:
        strip = _new_strip(track, AFTER, latest, end, job, slot)
        strip.extrapolation = "HOLD_FORWARD"
    if start < min(float(s.frame_start) for s in every):
        strip = _new_strip(track, BEFORE, start, float(first.frame_start), job, slot)
        strip.extrapolation = "HOLD"
        if first.extrapolation == "HOLD":
            # Only the first strip holds backwards, which it no longer is. Any
            # other setting the user gave it stays.
            first.extrapolation = "HOLD_FORWARD"


def bakes_to_strip(rig) -> bool:
    """Whether Bake IK goes into a strip: with the job laid out as strips, out of
    tweak mode. Whatever the preference: it says how the next Generate lays the
    job out, and until then the job is where it is."""
    anim = rig.animation_data
    return anim is not None and not anim.use_tweak_mode and laid_out(rig)


def bake_refusal(rig, first: int, last: int) -> str:
    """Why a bake onto a strip can't go ahead, in words, or "" if it can. Asked
    before anything changes."""
    if not bakes_to_strip(rig):
        return ""
    anim = rig.animation_data
    if last == first:
        return (
            f"A job on the NLA bakes onto a strip, which needs two frames or more: "
            f"bake {first}-{first + 1}"
        )
    if not anim.use_nla:
        # The strip it bakes onto wouldn't play, and nor does the job now.
        return "The robot's NLA is off, so the job's strips don't play: turn it on to bake"
    if any(track.is_solo for track in anim.nla_tracks):
        # A soloed track plays alone, without the copy the bake plays and keys.
        return "An NLA track of the robot's is soloed: un-solo it to bake"
    return ""


def catch_up(rig) -> list:
    """Take the frames of motions whose strips were moved or scaled in the NLA
    editor into the Motions list. Returns a line for each, for a report.

    A motion's strip is the one showing the frames of the action the motion was
    laid out at. Where the strip now plays other frames, and the motion's own
    frames are still the ones it was laid out at, the motion takes the strip's;
    where both changed, the Motions list's are kept, and said so. A strip that
    isn't a motion's -- a pause, a bridging joint move, Before or After -- takes
    its frames from the motions around it, so a move of it goes, and is said.
    """
    anim = rig.animation_data
    if anim is None or anim.use_tweak_mode or rig.library is not None:
        return []
    tracks = _job_tracks(rig)
    if not tracks:
        return []
    motions = {
        tuple(int(v) for v in motion.laid_frames): motion
        for motion in getattr(rig, "kinema_motions", ())
        if motion.laid_frames[1] > motion.laid_frames[0]
    }
    notes = []
    for strip in sorted((s for t in tracks for s in t.strips), key=lambda s: s.frame_start):
        if (
            abs(strip.frame_start - strip.action_frame_start) < 1e-3
            and abs(strip.frame_end - strip.action_frame_end) < 1e-3
        ):
            continue
        shown = (round(strip.action_frame_start), round(strip.action_frame_end))
        plays = (round(strip.frame_start), round(strip.frame_end))
        motion = motions.get(shown)
        if motion is None:
            notes.append(
                f"The NLA strip '{strip.name}' isn't a motion's, so it follows the "
                f"motions around it: the next Generate puts it back"
            )
            continue
        current = (motion.frame_start, motion.frame_end)
        if current == plays:
            continue
        if current != shown:
            notes.append(
                f"'{motion.name}' was retimed in the Motions list and in the NLA editor: "
                f"the Motions list's frames {current[0]}-{current[1]} are kept"
            )
            continue
        start, end = plays[0], max(plays[1], plays[0] + 1)
        motion.frame_start, motion.frame_end = start, end
        notes.append(
            f"'{motion.name}' retimed in the NLA editor, to frames {start}-{end}: "
            f"Generate Motion lays the job out there"
        )
    return notes


class Bake:
    """What :func:`baking` yields: ``into_strip`` says whether the bake goes into a
    copy of the job's action, and the bake sets ``keep`` once it has finished, so
    an unfinished one leaves no strip."""

    def __init__(self, into_strip: bool):
        self.into_strip, self.keep = into_strip, False


@contextmanager
def baking(rig, first: int, last: int, context=None):
    """Bake IK into a copy of the job's action, shown over ``first``-``last`` on a
    track above the job. With no job laid out as strips, Bake keys the active
    action as it always has: ``into_strip`` is False.

    Hand keys go into the copy, which then plays as the job and the hand keys
    did, and the bake's checks and solves see what is baked. Only a bake that
    finishes moves them into the job too, as Generate would: one refused or
    failed leaves the hand action active and the job as it was.
    """
    if not bakes_to_strip(rig):
        yield Bake(False)
        return
    anim = rig.animation_data
    job = rig.kinema_job_action
    slot = _slot(rig)
    hand, hand_slot = anim.action, getattr(anim, "action_slot", None)
    if hand == job:
        hand = None
    baked = job.copy()
    baked.name = f"{rig.name} Baked"
    if PROP_JOB in baked:
        del baked[PROP_JOB]
    baked[PROP_BAKED] = rig.name
    # A copy has slots of its own: the one standing for the same rig.
    baked_slot = _same_slot(baked, slot)
    if hand is not None:
        _merge(hand, hand_slot, baked, baked_slot)
    # The copy plays in the job's place: the job's track muted, the copy active
    # on top, at full strength. The NLA stays on, so any track of the user's
    # still plays under it, and what is baked is what plays.
    job_tracks = _job_tracks(rig)
    was_muted = [track.mute for track in job_tracks]
    for track in job_tracks:
        track.mute = True
    blend, influence = anim.action_blend_type, anim.action_influence
    extrapolation = anim.action_extrapolation
    anim.action_blend_type, anim.action_influence = "REPLACE", 1.0
    # Held past its keys, as the job's strips hold, so frames there bake what
    # plays live rather than whatever pose was left.
    anim.action_extrapolation = "HOLD"
    _activate(rig, baked, baked_slot)
    state = Bake(True)
    try:
        yield state
    finally:
        anim.action = None
        anim.action_blend_type, anim.action_influence = blend, influence
        anim.action_extrapolation = extrapolation
        for track, mute in zip(job_tracks, was_muted, strict=True):
            track.mute = mute
        if not state.keep:
            bpy.data.actions.remove(baked)
            if hand is not None:
                _activate(rig, hand, hand_slot)
    if state.keep:
        if hand is not None:
            _merge(hand, hand_slot, job, slot)
            _cover(rig, slot)
        _place_bake(rig, baked, baked_slot, float(first), float(last))


def _place_bake(rig, baked, slot, first: float, last: float) -> None:
    """Show ``baked`` over ``first``-``last``, above any earlier bake under those
    frames: on the top track of bakes where it has room and plays, or on a new
    one above.

    An earlier bake is left as it is, settings and all, and plays wherever the
    new one doesn't reach. One wholly under it that holds nothing past its frames
    can never play again, so it goes, with its action, and so does a track of
    bakes left empty.
    """
    anim = rig.animation_data
    gone = []
    for track in _baked_tracks(rig):
        for strip in list(track.strips):
            if (
                strip.frame_start >= first and strip.frame_end <= last
                and strip.extrapolation == "NOTHING"
            ):
                # Once each: strips duplicated in the NLA editor share one.
                if all(strip.action != action for action in gone):
                    gone.append(strip.action)
                track.strips.remove(strip)
        if not len(track.strips):
            anim.nla_tracks.remove(track)
    tracks = _baked_tracks(rig)
    top = tracks[-1] if tracks else None
    # A muted track would hide the new bake, and unmuting it would bring back the
    # bakes the user muted there.
    if top is None or top.mute or any(
        s.frame_start < last and s.frame_end > first for s in top.strips
    ):
        top = _new_track(anim, BAKED_TRACK)
    strip = _new_strip(top, f"Baked {first:g}-{last:g}", first, last, baked, slot)
    strip.extrapolation = "NOTHING"
    for action in gone:
        if action is not None and PROP_BAKED in action and action.users == 0:
            bpy.data.actions.remove(action)
