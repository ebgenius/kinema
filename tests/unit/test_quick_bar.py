"""The quick bar's layout (#103): two to a row, the first alone on an odd count."""

from __future__ import annotations

import pytest

from ..conftest import load_addon_module

quick_bar = load_addon_module("ui.quick_bar")


@pytest.mark.parametrize(
    "keys, laid_out",
    [
        (("LIVE_IK",), [["LIVE_IK"]]),
        (("LIVE_IK", "REST_POSE"), [["LIVE_IK", "REST_POSE"]]),
        (
            ("LIVE_IK", "REST_POSE", "GRAPH_EDITOR"),
            [["LIVE_IK"], ["REST_POSE", "GRAPH_EDITOR"]],
        ),
        (
            quick_bar.DEFAULT,
            [
                ["LIVE_IK"],
                ["CONFIGURATIONS", "AUTO_CONFIG"],
                ["REST_POSE", "SNAP_IK"],
                ["RECORD_WAYPOINT", "GENERATE"],
                ["MOTION_CHECK", "GRAPH_EDITOR"],
            ],
        ),
        ((), []),
    ],
    ids=["one", "two", "three", "default", "none"],
)
def test_two_to_a_row_the_first_alone_on_an_odd_count(keys, laid_out):
    assert quick_bar.rows(keys) == laid_out


def test_a_button_it_doesnt_know_is_skipped_before_laying_out():
    """From a newer Kinema's preferences, say: left out, not counted."""
    assert quick_bar.rows(("LIVE_IK", "FROM_THE_FUTURE", "REST_POSE")) == [
        ["LIVE_IK", "REST_POSE"]
    ]


def test_the_defaults_are_buttons_it_knows():
    assert set(quick_bar.DEFAULT) <= set(quick_bar.BUTTONS)
    assert len(set(quick_bar.DEFAULT)) == len(quick_bar.DEFAULT)
