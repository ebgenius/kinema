"""kinema.api's surface, recorded (#102): every name, signature and field.

Another add-on calls these, and can't be changed with them, so a change here
fails until ``api_surface.txt`` is written again -- and ``VERSION`` with it:
the minor number for something added, the major number for something changed or
gone. To write it again, once the version says what changed:

    KINEMA_WRITE_API_SURFACE=1 uv run pytest tests/unit/test_api_surface.py
"""

from __future__ import annotations

import dataclasses
import inspect
import os
from pathlib import Path

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
            fields = ", ".join(f"{f.name}: {f.type}" for f in dataclasses.fields(thing))
            frozen = thing.__dataclass_params__.frozen
            lines.append(f"dataclass{' frozen' if frozen else ''} {name}({fields})")
        elif inspect.isclass(thing):
            bases = ", ".join(base.__name__ for base in thing.__bases__)
            lines.append(f"class {name}({bases})")
        else:
            lines.append(f"def {name}{inspect.signature(thing)}")
    return "\n".join(lines) + "\n"


def test_the_surface_is_as_recorded():
    now = surface()
    if os.environ.get("KINEMA_WRITE_API_SURFACE"):
        RECORDED.write_text(now, encoding="utf-8", newline="\n")
    recorded = RECORDED.read_text(encoding="utf-8")
    assert now == recorded, (
        "kinema.api changed. Bump VERSION -- minor for an addition, major for a "
        "change or a removal -- and write api_surface.txt again (see this file's "
        "docstring)."
    )


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
