"""Splitting the suite across processes: tests/sharding.py."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from .. import sharding

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestShares:
    def test_the_longest_go_first_to_the_least_loaded_share(self):
        """100 and 90 s apart; the rest fill in round them."""
        durations = {"a": 100.0, "b": 90.0, "c": 30.0, "d": 20.0, "e": 10.0}
        owners = sharding.shares(list(durations), durations, 2)
        loads = [0.0, 0.0]
        for nodeid, owner in zip(durations, owners, strict=True):
            loads[owner] += durations[nodeid]
        assert owners[0] != owners[1]
        assert max(loads) - min(loads) <= 10.0

    def test_every_process_works_out_the_same_split(self):
        """Worked out in separate interpreters, with different string hashing, as the
        Blender processes do: many ties, so any order that came from hashing shows."""
        script = (
            "import json, sys\n"
            "from tests import sharding\n"
            "ids = [f'tests/x.py::t{i}' for i in range(60)]\n"
            "times = {n: float(i % 3) for i, n in enumerate(ids) if i % 5}\n"
            "print(json.dumps(sharding.shares(ids, times, 4)))\n"
        )
        splits = []
        for seed in ("1", "2", "3"):
            env = {**os.environ, "PYTHONHASHSEED": seed}
            out = subprocess.run(
                [sys.executable, "-c", script], cwd=REPO_ROOT, env=env,
                capture_output=True, text=True, check=True,
            ).stdout
            splits.append(json.loads(out))
        assert splits[0] == splits[1] == splits[2]

    def test_every_test_goes_to_exactly_one_share(self):
        nodeids = [f"t{i}" for i in range(30)]
        owners = sharding.shares(nodeids, {}, 3)
        assert len(owners) == 30 and set(owners) == {0, 1, 2}

    def test_a_test_not_timed_yet_counts_as_the_median(self):
        """Two timed at 1 and 3 s, and two not: each counts as 2, so both go together
        opposite the 3, and the 1 joins the 3. At 1 s each, two of the three 1s
        would go opposite the 3."""
        owners = sharding.shares(
            ["slow", "fast", "new1", "new2"], {"slow": 3.0, "fast": 1.0}, 2
        )
        assert owners == [0, 0, 1, 1]

    def test_one_share_is_all_of_it(self):
        assert sharding.shares(["a", "b"], {}, 1) == [0, 0]

    def test_a_shard_is_read_as_index_and_count(self):
        assert sharding.parse_shard("2/5") == (2, 5)
        with pytest.raises(ValueError):
            sharding.parse_shard("5/5")

    def test_recorded_times_read_back_or_none(self, tmp_path):
        path = tmp_path / "durations.json"
        assert sharding.read_durations(path) == {}
        path.write_text('{"a": 1.5}', encoding="utf-8")
        assert sharding.read_durations(path) == {"a": 1.5}

    @pytest.mark.parametrize("text", ["[]", "null", "3", '{"a": "fast"}', '{"a": null}', "{"])
    def test_recorded_times_of_the_wrong_shape_count_as_none(self, tmp_path, text):
        """Every process reads them while collecting: one raising would fail them all."""
        path = tmp_path / "durations.json"
        path.write_text(text, encoding="utf-8")
        assert sharding.read_durations(path) == {}
