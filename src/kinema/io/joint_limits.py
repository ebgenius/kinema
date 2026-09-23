"""Read joint velocity limits from a MoveIt or ros2_control ``joint_limits.yaml``.

A URDF's ``<limit velocity>`` is often a placeholder, and the numbers a robot
is actually run at live in its MoveIt configuration instead. Both MoveIt and
ros2_control write them the same way::

    joint_limits:
      joint_1:
        has_velocity_limits: true
        max_velocity: 2.175

MoveIt 2 and ros2_control files often nest that section under a node name and
``ros__parameters``, so it is searched for rather than expected at the top.

Following MoveIt, the file overrides the description joint by joint:
``has_velocity_limits: true`` sets the limit, ``false`` turns it off, and a
joint the file does not mention keeps whatever the description gave it.

Deliberately free of ``bpy``, like the rest of ``io/``.
"""

from __future__ import annotations

import math
from pathlib import Path

SECTION = "joint_limits"


class JointLimitsError(ValueError):
    """A file that holds no joint limits Kinema can read."""


def velocity_limits(text: str, *, source: str = "the file") -> dict[str, float | None]:
    """Joint name -> max velocity, or None where the file turns the limit off.

    Joints the file says nothing usable about -- no ``has_velocity_limits``,
    or a limit switched on without a positive ``max_velocity`` -- are left out,
    so the description's own value stands for them.
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

    limits: dict[str, float | None] = {}
    for section in sections:
        for joint, entry in section.items():
            if not isinstance(entry, dict) or "has_velocity_limits" not in entry:
                continue
            if not _truthy(entry["has_velocity_limits"]):
                limits[str(joint)] = None
                continue
            value = _positive(entry.get("max_velocity"))
            if value is not None:
                limits[str(joint)] = value
    return limits


def read_velocity_limits(path: str | Path) -> dict[str, float | None]:
    """:func:`velocity_limits` for a file on disk."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise JointLimitsError(f"Could not open {path.name}: {exc}") from exc
    return velocity_limits(text, source=path.name)


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
