"""A description's path relative to the .blend: rig/description_path.py (#92)."""

from __future__ import annotations

import os

import pytest

from ..conftest import load_addon_module

paths = load_addon_module("rig.description_path")

PROJECT = os.path.abspath(os.path.join(os.sep, "work", "project"))
BLEND = os.path.join(PROJECT, "scene.blend")


@pytest.mark.parametrize(
    "where, kept",
    [
        (("robots", "arm6.urdf"), "//" + os.path.join("robots", "arm6.urdf")),
        (("arm6.urdf",), "//arm6.urdf"),
    ],
    ids=["under", "beside"],
)
def test_a_description_in_the_blends_folder_is_kept_relative(where, kept):
    assert paths.relative_to(os.path.join(PROJECT, *where), BLEND) == kept


@pytest.mark.parametrize(
    "where",
    [
        (os.pardir, "robots", "arm6.urdf"),
        (os.pardir, "arm6.urdf"),
        # A folder whose name starts as the project's does is not under it.
        (os.pardir, "project-robots", "arm6.urdf"),
    ],
    ids=["sibling", "parent", "prefix"],
)
def test_a_description_elsewhere_is_not(where):
    assert paths.relative_to(os.path.normpath(os.path.join(PROJECT, *where)), BLEND) is None


def test_nothing_is_relative_to_an_unsaved_file():
    assert paths.relative_to(os.path.join(PROJECT, "arm6.urdf"), "") is None


@pytest.mark.skipif(os.name != "nt", reason="drives are Windows'")
def test_a_description_on_another_drive_is_not():
    assert paths.relative_to(r"Q:\robots\arm6.urdf", r"C:\work\project\scene.blend") is None
