"""The job as NLA strips: one per motion, all showing one action.

Generate Motion writes the job's keys into one action, as it always has, and
this lays that action out on the rig's NLA as strips: one per motion -- a hold
between two motions, and a joint move bridging a gap, count as motions here --
named after it and covering its frames, and a "Before" and an "After" strip for
whatever the action holds outside the job, hand keys included, out to the
scene's ends at least. They tile the timeline end to end and all show the same
action, each its own frames, so the rig plays exactly as one action would:
every channel, at every frame of the scene. They replace what is under them at
full strength, as the active action plays any channel the tracks under it don't
animate, whatever its blend and influence.

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
muting its track goes back to the live job. Regenerating mutes the bakes over
the job's frames, which no longer show it.

The tracks are known by what their strips show -- the job's action, or a bake --
not by their names, so a track of the user's with the same name is left alone.

The add-on preference "Job as NLA Strips" turns all of this off. A rig laid out
as strips is then put back in its active action on the next Generate.
"""

from __future__ import annotations

from contextlib import contextmanager

import bpy

#: The names Kinema gives its tracks.
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


def _owned(anim, shows):
    """The first track whose strips all show an action ``shows`` accepts."""
    if anim is None:
        return None
    for track in anim.nla_tracks:
        if len(track.strips) and all(
            s.action is not None and shows(s.action) for s in track.strips
        ):
            return track
    return None


def _job_track(rig):
    job = getattr(rig, "kinema_job_action", None)
    if job is None:
        return None
    return _owned(rig.animation_data, lambda action: action == job)


def _baked_track(rig):
    return _owned(rig.animation_data, lambda action: PROP_BAKED in action)


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
    return _job_track(rig) is not None


def _slot(rig):
    """The slot the job's strips show, or the active one while the job is the
    active action; None if neither. A slot belongs to one action, so the active
    one of any other action is no answer."""
    track = _job_track(rig)
    if track is not None:
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
            group = curve.group.name if curve.group is not None else ""
            whole = into is None or len(curve.modifiers) or not len(curve.keyframe_points)
            if whole and hasattr(destinations, "new_from_fcurve"):
                kept = []
                if into is not None:
                    group = into.group.name if into.group is not None else group
                    frames = {float(p.co[0]) for p in curve.keyframe_points}
                    kept = [
                        _key(p) for p in into.keyframe_points
                        if all(abs(p.co[0] - f) >= 1e-4 for f in frames)
                    ]
                    destinations.remove(into)
                into = destinations.new_from_fcurve(curve)
                _group(bag, into, group)
                if len(into.keyframe_points):
                    for key in kept:
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
            track = _job_track(rig)
            if track is not None:
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


def _tile(rig, spans, slot) -> None:
    """The job's track, its strips replaced by one per ``(name, start, end)`` of
    ``spans`` -- in frame order, with Before and After added for what the job's
    action holds outside them -- each showing those frames of the action. The
    active action is then left empty."""
    anim = rig.animation_data
    job = rig.kinema_job_action
    track = _job_track(rig) or _new_track(anim, JOB_TRACK)
    for strip in list(track.strips):
        track.strips.remove(strip)
    start, end = (float(v) for v in job.frame_range)
    # Out to the scene's ends too: a curve can play on past its keys -- cycled,
    # or extrapolated in a line -- and past a strip's frames it would hold.
    scene = bpy.context.scene
    if scene is not None:
        start, end = min(start, float(scene.frame_start)), max(end, float(scene.frame_end))
    tiles = []
    if start < spans[0][1]:
        tiles.append((BEFORE, start, spans[0][1]))
    tiles += spans
    if end > spans[-1][2]:
        tiles.append((AFTER, spans[-1][2], end))
    anim.action = None
    # In frame order: a strip comes in at its action's whole length, and is cut
    # to its frames after. Only the empty track after the last one has room.
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
    slot = getattr(anim, "action_slot", None) if anim.action == job else _slot(rig)
    spans = [
        (span.label, float(span.start_frame), float(span.end_frame))
        for span in plan if span.end_frame > span.start_frame
    ]
    if not spans:
        return 0
    _tile(rig, spans, slot)
    first, last = spans[0][1], spans[-1][2]
    muted = 0
    baked = _baked_track(rig)
    for strip in baked.strips if baked is not None else ():
        if not strip.mute and strip.frame_start <= last and strip.frame_end >= first:
            strip.mute = True
            muted += 1
    return muted


def muted_text(count: int) -> str:
    """What a report says about the bakes :func:`lay_out` muted."""
    several = count != 1
    return (
        f"Muted {count} baked strip{'s' if several else ''} on '{BAKED_TRACK}' that "
        f"no longer show{'' if several else 's'} the job; unmute to play the old bake"
    )


def _cover(rig, slot) -> None:
    """Before and After stretched to what the job's action holds now: hand keys
    moved into it can reach past them."""
    track = _job_track(rig)
    spans = [
        (s.name, float(s.action_frame_start), float(s.action_frame_end))
        for s in track.strips if s.name not in (BEFORE, AFTER)
    ]
    if spans:
        _tile(rig, spans, slot)


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

    Whatever the preference: it says how the next Generate lays the job out, and
    until then the job is where it is. Hand keys are moved into the job first,
    as Generate would, so the copy holds them: call this once the bake is sure
    to go ahead.
    """
    anim = rig.animation_data
    if anim is None or anim.use_tweak_mode or not laid_out(rig):
        yield Bake(False)
        return
    job = rig.kinema_job_action
    slot = _slot(rig)
    hand, hand_slot = anim.action, getattr(anim, "action_slot", None)
    if hand is not None and hand != job:
        _merge(hand, hand_slot, job, slot)
        _cover(rig, slot)
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
    job_track = _job_track(rig)
    was_muted = job_track.mute
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
        job_track.mute = was_muted
        if not state.keep:
            bpy.data.actions.remove(baked)
    if state.keep:
        _place_bake(rig, baked, slot, float(first), float(last))


def _place_bake(rig, baked, slot, first: float, last: float) -> None:
    """Show ``baked`` over ``first``-``last`` on the bakes' track. An earlier bake
    under those frames gives them up and keeps the rest, on either side."""
    anim = rig.animation_data
    track = _baked_track(rig) or _new_track(anim, BAKED_TRACK)
    pieces = []
    for strip in track.strips:
        a, b = float(strip.frame_start), float(strip.frame_end)
        offset = float(strip.action_frame_start) - a
        kept = [(a, b)] if b <= first or a >= last else [
            (x, y) for x, y in ((a, min(b, first)), (max(a, last), b)) if y > x
        ]
        for x, y in kept:
            pieces.append((x, y, strip.action, strip.action_slot, offset, strip.mute))
    pieces.append((first, last, baked, slot, 0.0, False))
    shown = {piece[2] for piece in pieces}
    gone = {strip.action for strip in track.strips} - shown
    for strip in list(track.strips):
        track.strips.remove(strip)
    # In frame order, as _tile lays the job out, and for the same reason.
    for a, b, action, action_slot, offset, mute in sorted(pieces, key=lambda p: p[0]):
        strip = track.strips.new(f"Baked {a:g}-{b:g}", int(a), action)
        if action_slot is not None and hasattr(strip, "action_slot"):
            strip.action_slot = action_slot
        strip.use_sync_length = False
        strip.action_frame_start = a + offset
        strip.action_frame_end = b + offset
        strip.frame_start_ui = a
        strip.scale = 1.0
        strip.blend_type = "REPLACE"
        strip.extrapolation = "NOTHING"
        strip.mute = mute
    for action in gone:
        # A bake nothing shows any more.
        if action is not None and PROP_BAKED in action and action.users == 0:
            bpy.data.actions.remove(action)
