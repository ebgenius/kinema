"""Read joint limits from a MoveIt or ros2_control ``joint_limits.yaml``.

A URDF's ``<limit velocity>`` is often a placeholder, URDF has no field for
acceleration or jerk at all, and the numbers a robot is actually run at live in
its MoveIt configuration instead. Both MoveIt and ros2_control write them the
same way::

    joint_limits:
      joint_1:
        has_velocity_limits: true
        max_velocity: 2.175
        has_acceleration_limits: true
        max_acceleration: 3.75
        has_jerk_limits: false
        has_effort_limits: true
        max_effort: 87.0

MoveIt 2 and ros2_control files often nest that section under a node name and
``ros__parameters``, so it is searched for rather than expected at the top.

Following MoveIt, the file overrides the description joint by joint and kind by
kind: ``has_<kind>_limits: true`` sets that limit, ``false`` turns it off, and a
kind the file does not mention keeps whatever the description gave the joint.

Deliberately free of ``bpy``, like the rest of ``io/``.
"""

from __future__ import annotations

import math
from pathlib import Path

SECTION = "joint_limits"
#: The kinds of limit the file can carry, spelled as its keys spell them:
#: ``has_<kind>_limits`` and ``max_<kind>``.
KINDS = ("velocity", "acceleration", "jerk", "effort")


class JointLimitsError(ValueError):
    """A file that holds no joint limits Kinema can read."""


def joint_limits(text: str, *, source: str = "the file") -> dict[str, dict[str, float | None]]:
    """Joint name -> {kind: maximum, or None where the file turns that limit off}.

    A kind the file says nothing usable about -- no ``has_<kind>_limits``, or a
    limit switched on without a positive ``max_<kind>`` -- is left out, so the
    description's own value stands for it. A joint with no usable kind at all
    is left out too.
    """
    try:
        import yaml
    except ImportError:
        # Bundled as a wheel, since Blender's Python has none. Missing means a
        # broken install, which is worth a sentence rather than a traceback.
        raise JointLimitsError("PyYAML is unavailable; check Kinema's dependencies") from None

    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise JointLimitsError(f"Could not read {source} as YAML: {exc}") from exc

    sections = list(_find_sections(document))
    if not sections:
        raise JointLimitsError(f"No '{SECTION}' section in {source}")

    limits: dict[str, dict[str, float | None]] = {}
    for section in sections:
        for joint, entry in section.items():
            if not isinstance(entry, dict):
                continue
            found: dict[str, float | None] = {}
            for kind in KINDS:
                flag = f"has_{kind}_limits"
                if flag not in entry:
                    continue
                if not _truthy(entry[flag]):
                    found[kind] = None
                    continue
                value = _positive(entry.get(f"max_{kind}"))
                if value is not None:
                    found[kind] = value
            if found:
                limits.setdefault(str(joint), {}).update(found)
    return limits


def read_joint_limits(path: str | Path) -> dict[str, dict[str, float | None]]:
    """:func:`joint_limits` for a file on disk."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise JointLimitsError(f"Could not open {path.name}: {exc}") from exc
    return joint_limits(text, source=path.name)


def _find_sections(node):
    """Every mapping stored under a ``joint_limits`` key, outermost first."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == SECTION and isinstance(value, dict):
                yield value
            else:
                yield from _find_sections(value)
    elif isinstance(node, list):
        for item in node:
            yield from _find_sections(item)


def _truthy(value) -> bool:
    # YAML already turns true/yes/on into a bool. A quoted "true" arrives as a
    # string, and bool("false") would be True.
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "on", "1"}
    return bool(value)


def _positive(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0.0 and math.isfinite(number) else None
