"""kinema.api's surface, recorded (#102): every name, signature and field.

Another add-on calls these, and can't be changed with them, so a change here
fails until ``api_surface.txt`` is written again -- and ``VERSION`` with it:
the minor number for something added, the major number for something changed or
gone. Writing it again is refused until the version says so. To write it, once
it does:

    KINEMA_WRITE_API_SURFACE=1 uv run pytest tests/unit/test_api_surface.py

A version that hasn't shipped yet can still grow without a bump of its own:
delete ``api_surface.txt`` first, and it is written as it is. Every change to
the record shows in the diff, for review to see.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import os
from pathlib import Path

import pytest

from ..conftest import load_addon_module

api = load_addon_module("api")
RECORDED = Path(__file__).with_name("api_surface.txt")


def surface() -> str:
    lines = [f"VERSION = {api.VERSION!r}"]
    for name in api.__all__:
        if name == "VERSION":
            continue
        thing = getattr(api, name)
        if dataclasses.is_dataclass(thing):
            # The constructor's signature: field names, types, order and defaults.
            frozen = " frozen" if thing.__dataclass_params__.frozen else ""
            lines.append(f"dataclass{frozen} {name}{inspect.signature(thing)}")
        elif inspect.isclass(thing):
            bases = ", ".join(base.__name__ for base in thing.__bases__)
            lines.append(f"class {name}({bases})")
        else:
            lines.append(f"def {name}{inspect.signature(thing)}")
    return "\n".join(lines) + "\n"


def version_owed(recorded: str, now: str) -> str | None:
    """Why ``now`` can't replace ``recorded`` at the version it gives, or None."""
    old, new = recorded.splitlines(), now.splitlines()
    old_version = ast.literal_eval(old[0].split("=", 1)[1].strip())
    new_version = ast.literal_eval(new[0].split("=", 1)[1].strip())
    gone, added = set(old[1:]) - set(new[1:]), set(new[1:]) - set(old[1:])
    if new_version < old_version:
        return f"VERSION can't go back from {old_version}"
    if gone and new_version[0] <= old_version[0]:
        return f"something changed or went: the major number must grow past {old_version}"
    if added and new_version <= old_version:
        return f"something was added: VERSION must grow past {old_version}"
    return None


def test_the_surface_is_as_recorded():
    now = surface()
    if os.environ.get("KINEMA_WRITE_API_SURFACE"):
        if RECORDED.exists():
            owed = version_owed(RECORDED.read_text(encoding="utf-8"), now)
            assert owed is None, f"api_surface.txt not written: {owed}"
        RECORDED.write_text(now, encoding="utf-8", newline="\n")
    recorded = RECORDED.read_text(encoding="utf-8")
    assert now == recorded, (
        "kinema.api changed. Bump VERSION -- minor for an addition, major for a "
        "change or a removal -- and write api_surface.txt again (see this file's "
        "docstring)."
    )


#: The record the cases below are written over: version 1.0, one function.
BEFORE = "VERSION = (1, 0)\ndef a(x)\n"


@pytest.mark.parametrize(
    "recorded, now, owed",
    [
        (BEFORE, "VERSION = (1, 0)\ndef a(x)\n", None),
        (BEFORE, "VERSION = (1, 0)\ndef a(x)\ndef b()\n", "something was added"),
        (BEFORE, "VERSION = (1, 1)\ndef a(x)\ndef b()\n", None),
        (BEFORE, "VERSION = (1, 1)\ndef a(x, y)\n", "the major number must grow"),
        (BEFORE, "VERSION = (2, 0)\ndef a(x, y)\n", None),
        (BEFORE, "VERSION = (1, 1)\n", "the major number must grow"),
        ("VERSION = (1, 2)\ndef a(x)\n", BEFORE, "can't go back"),
    ],
    ids=[
        "unchanged", "added-unbumped", "added-minor", "changed-minor", "changed-major", "gone",
        "lowered",
    ],
)
def test_writing_it_again_needs_the_version_to_say_what_changed(recorded, now, owed):
    found = version_owed(recorded, now)
    assert found is None if owed is None else owed in (found or "")


def test_every_public_name_is_listed():
    """A public name left out of ``__all__`` would escape the record."""
    own = {
        name
        for name, thing in vars(api).items()
        if not name.startswith("_")
        and not inspect.ismodule(thing)
        and getattr(thing, "__module__", api.__name__) == api.__name__
    }
    own.discard("annotations")
    assert own == set(api.__all__)


def test_the_version_is_two_numbers():
    major, minor = api.VERSION
    assert isinstance(major, int) and isinstance(minor, int) and major >= 1
