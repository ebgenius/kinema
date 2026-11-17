"""Splitting the suite across processes: tests/sharding.py."""

from __future__ import annotations

import pytest

from .. import sharding


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
        nodeids = [f"t{i}" for i in range(50)]
        durations = {n: float(i % 7) for i, n in enumerate(nodeids)}
        assert sharding.shares(nodeids, durations, 4) == sharding.shares(nodeids, durations, 4)

    def test_every_test_goes_to_exactly_one_share(self):
        nodeids = [f"t{i}" for i in range(30)]
        owners = sharding.shares(nodeids, {}, 3)
        assert len(owners) == 30 and set(owners) == {0, 1, 2}

    def test_a_test_not_timed_yet_counts_as_the_median(self):
        """Two timed at 1 and 3 s, and one not: it counts as 2, so it goes with the 1."""
        owners = sharding.shares(["slow", "fast", "new"], {"slow": 3.0, "fast": 1.0}, 2)
        assert owners[2] == owners[1]

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
