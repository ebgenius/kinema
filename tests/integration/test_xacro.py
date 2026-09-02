"""Importing a xacro from disk.

38 of the catalogue's 186 robots ship only a xacro, and since Kinema stopped
downloading descriptions this is the path a user takes to reach any of them.
It had no coverage at all, and was broken on Windows the whole time.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ..conftest import load_addon_module

loader = load_addon_module("io.loader")


@pytest.fixture(scope="module")
def xacro_path(fixture_dir):
    return fixture_dir / "kinema_fixture" / "urdf" / "arm.urdf.xacro"


def test_xacro_renders_and_parses(xacro_path):
    """Macros, properties and $(find) all have to survive the render.

    The fixture's three joints only exist as one xacro macro invoked three
    times, so a model with three joints proves the render really ran rather
    than the file being parsed as plain XML.
    """
    result = loader.load_file(xacro_path)
    assert result.error is None, result.error
    assert result.model is not None
    assert len(result.model.joints) == 3
    assert {j.name for j in result.model.joints} == {"joint_1", "joint_2", "joint_3"}


def test_the_render_is_not_left_behind(xacro_path):
    """The rendered URDF is a temp file and must be cleaned up.

    xacrodoc's own temp_urdf_file_path holds the file open while yielding its
    path, which Windows refuses to reopen -- so loader renders it by hand. That
    makes cleanup ours to get right too.
    """
    temp_dir = Path(tempfile.gettempdir())
    before = set(temp_dir.glob("kinema-*.urdf"))
    loader.load_file(xacro_path)
    assert set(temp_dir.glob("kinema-*.urdf")) == before


def test_source_is_recorded_as_the_xacro_not_the_render(xacro_path):
    """The rig has to remember the file the user picked: the render is gone by
    the time the solver wants to reload it."""
    result = loader.load_file(xacro_path)
    assert result.source == ("file", str(xacro_path))
