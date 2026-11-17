"""tools/dev.py test: starting and stopping the Blender processes, and how many."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def dev():
    spec = importlib.util.spec_from_file_location("kinema_dev_tool", REPO_ROOT / "tools" / "dev.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeBlender:
    """A started process that runs until it is waited for, or stopped."""

    def __init__(self, cmd, env, stdout, stderr, interrupt=False):
        self.log, self.interrupt = stdout, interrupt
        self.code, self.stopped = None, False

    def poll(self):
        return self.code

    def wait(self, timeout=None):
        if self.interrupt and not self.stopped:
            raise KeyboardInterrupt
        if self.code is None:
            self.code = 0
        return self.code

    def terminate(self):
        self.stopped, self.code = True, 1

    kill = terminate


def _starter(monkeypatch, dev, fail_at=None, interrupt_at=None):
    started = []
    opened = _opened(monkeypatch)

    def popen(cmd, env, stdout, stderr):
        if len(started) == fail_at:
            raise OSError("no Blender here")
        process = FakeBlender(cmd, env, stdout, stderr, interrupt=len(started) == interrupt_at)
        started.append(process)
        return process

    monkeypatch.setattr(dev.subprocess, "Popen", popen)
    return started, opened


def _opened(monkeypatch):
    """Every file opened through a Path from here on, to see each log closed."""
    files, real = [], Path.open

    def record(self, *args, **kwargs):
        handle = real(self, *args, **kwargs)
        files.append(handle)
        return handle

    monkeypatch.setattr(Path, "open", record)
    return files


class TestRunningShares:
    def test_all_run_to_the_end_and_their_logs_are_closed(self, dev, monkeypatch, tmp_path):
        started, opened = _starter(monkeypatch, dev)
        assert dev._run_shares(["blender"], 3, tmp_path, lambda i: {}) == [0, 0, 0]
        assert not any(p.stopped for p in started)
        assert len(opened) == 3 and all(f.closed for f in opened)

    def test_one_failing_to_start_stops_those_already_running(self, dev, monkeypatch, tmp_path):
        started, opened = _starter(monkeypatch, dev, fail_at=2)
        with pytest.raises(OSError):
            dev._run_shares(["blender"], 4, tmp_path, lambda i: {})
        assert len(started) == 2
        assert all(p.stopped for p in started), "left running"
        assert len(opened) == 3, "the one that failed to start had its log opened"
        assert all(f.closed for f in opened)

    def test_an_interrupted_wait_stops_them_all(self, dev, monkeypatch, tmp_path):
        started, opened = _starter(monkeypatch, dev, interrupt_at=0)
        with pytest.raises(KeyboardInterrupt):
            dev._run_shares(["blender"], 3, tmp_path, lambda i: {})
        assert all(p.stopped for p in started), "left running"
        assert all(f.closed for f in opened)

    def test_one_that_will_not_stop_is_killed(self, dev, monkeypatch, tmp_path):
        started, _ = _starter(monkeypatch, dev, fail_at=1)

        def deaf(timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("blender", timeout)
            return 1

        killed = []
        monkeypatch.setattr(FakeBlender, "terminate", lambda self: None)
        monkeypatch.setattr(FakeBlender, "kill", lambda self: killed.append(self))
        monkeypatch.setattr(FakeBlender, "wait", lambda self, timeout=None: deaf(timeout))
        with pytest.raises(OSError):
            dev._run_shares(["blender"], 2, tmp_path, lambda i: {})
        assert killed == started


def _part(times, whole=True):
    return {"whole": whole, "collected": ["a", "b", "c"], "times": times}


class TestMergingDurations:
    RECORDED = {"a": 1.0, "c": 4.0, "gone": 9.0}

    def test_the_whole_suite_drops_a_test_that_is_gone(self, dev):
        merged = dev._merged_durations(self.RECORDED, [_part({"a": 2.0}), _part({"b": 3.0})])
        assert merged == {"a": 2.0, "b": 3.0, "c": 4.0}

    def test_part_of_it_or_a_failed_import_keeps_the_rest(self, dev):
        """Either way a process says it didn't collect the whole suite."""
        parts = [_part({"a": 2.0}), _part({"b": 3.0}, whole=False)]
        merged = dev._merged_durations(self.RECORDED, parts)
        assert merged == {"a": 2.0, "b": 3.0, "c": 4.0, "gone": 9.0}

    def test_a_report_that_cannot_be_read_keeps_the_rest(self, dev):
        merged = dev._merged_durations(self.RECORDED, [_part({"a": 2.0}), None])
        assert merged["gone"] == 9.0

    def test_no_process_reporting_drops_nothing(self, dev):
        assert dev._merged_durations({"a": 1.0}, []) == {"a": 1.0}

    @pytest.mark.parametrize("recorded", [None, [], "x", {"a": "slow", "b": None}])
    def test_a_bad_file_is_written_over_with_the_times_measured(self, dev, recorded):
        assert dev._merged_durations(recorded, [_part({"a": 2.0})]) == {"a": 2.0}

    def test_a_file_cut_short_reads_as_none(self, dev, tmp_path):
        path = tmp_path / "durations.json"
        path.write_text('{"a": 1.', encoding="utf-8")
        assert dev._read_json(path) is None
        assert dev._read_json(tmp_path / "missing.json") is None


class TestReports:
    def test_a_process_that_wrote_none_is_none(self, dev, tmp_path):
        (tmp_path / "durations-0.json").write_text('{"whole": true}', encoding="utf-8")
        assert dev._reports(tmp_path, 2) == [{"whole": True}, None]


class TestShareEnv:
    def test_every_share_splits_by_the_same_copy(self, dev, tmp_path):
        snapshot = tmp_path / "durations.json"
        envs = [dev._share_env({"PATH": "x"}, i, 3, tmp_path, snapshot, False) for i in range(3)]
        assert [env["KINEMA_TEST_SHARD"] for env in envs] == ["0/3", "1/3", "2/3"]
        assert {env["KINEMA_TEST_DURATIONS"] for env in envs} == {str(snapshot)}
        assert all(env["PATH"] == "x" and "KINEMA_RECORD_DURATIONS" not in env for env in envs)

    def test_recording_each_reports_to_a_file_of_its_own(self, dev, tmp_path):
        env = dev._share_env({}, 1, 3, tmp_path, tmp_path / "d.json", True)
        assert env["KINEMA_RECORD_DURATIONS"] == str(tmp_path / "durations-1.json")


class TestSnapshot:
    def test_it_is_a_copy_the_file_changing_after_does_not_touch(self, dev, tmp_path, monkeypatch):
        source = tmp_path / "durations.json"
        source.write_text('{"a": 1.0}', encoding="utf-8")
        logs = tmp_path / "logs"
        logs.mkdir()
        monkeypatch.setattr(dev, "DURATIONS", source)
        snapshot = dev._snapshot_durations(logs)
        source.write_text('{"a": 9.0}', encoding="utf-8")
        assert snapshot.parent == logs
        assert dev._read_json(snapshot) == {"a": 1.0}

    def test_no_times_is_an_empty_copy(self, dev, tmp_path, monkeypatch):
        monkeypatch.setattr(dev, "DURATIONS", tmp_path / "missing.json")
        assert dev._read_json(dev._snapshot_durations(tmp_path)) == {}


class TestWritingDurations:
    def test_it_lands_whole_and_leaves_nothing_beside_it(self, dev, tmp_path):
        path = tmp_path / "durations.json"
        path.write_text('{"old": 1.0}', encoding="utf-8")
        dev._write_durations(path, {"a": 2.0})
        assert dev._read_json(path) == {"a": 2.0}
        assert [p.name for p in tmp_path.iterdir()] == ["durations.json"]

    def test_two_runs_write_files_of_their_own(self, dev, tmp_path, monkeypatch):
        """Each moves its own file over the times: not one another's, half written."""
        moved, real = [], dev.os.replace

        def replace(source, destination):
            moved.append(Path(source))
            real(source, destination)

        monkeypatch.setattr(dev.os, "replace", replace)
        path = tmp_path / "durations.json"
        dev._write_durations(path, {"a": 1.0})
        dev._write_durations(path, {"a": 2.0})
        assert moved[0] != moved[1]
        assert all(source.parent == tmp_path for source in moved), "beside it, to move"

    def test_it_keeps_the_mode_of_the_file_it_replaces(self, dev, tmp_path, monkeypatch):
        """mkstemp makes a file only its owner can read: moved over the times, it
        would take that mode with it."""
        given, real = [], dev.os.chmod

        def chmod(target, mode):
            given.append(mode)
            real(target, mode)

        monkeypatch.setattr(dev.os, "chmod", chmod)
        path = tmp_path / "durations.json"
        dev._write_durations(path, {"a": 1.0})
        path.write_text("{}", encoding="utf-8")
        before = path.stat().st_mode & 0o777
        dev._write_durations(path, {"a": 2.0})
        assert given == [0o644, before], "a new file is 644; one written over keeps its mode"

    def test_a_failed_write_leaves_the_times_and_no_file(self, dev, tmp_path, monkeypatch):
        path = tmp_path / "durations.json"
        path.write_text('{"old": 1.0}', encoding="utf-8")

        def refuse(source, destination):
            raise OSError("in use")

        monkeypatch.setattr(dev.os, "replace", refuse)
        with pytest.raises(OSError):
            dev._write_durations(path, {"a": 2.0})
        assert dev._read_json(path) == {"old": 1.0}
        assert [p.name for p in tmp_path.iterdir()] == ["durations.json"]


class TestFreeMemory:
    def test_a_v2_limit_less_what_it_uses(self, dev, tmp_path):
        (tmp_path / "memory.max").write_text(f"{8 * 2**30}\n", encoding="ascii")
        (tmp_path / "memory.current").write_text(f"{3 * 2**30}\n", encoding="ascii")
        assert dev._container_free_gb(tmp_path) == 5.0

    def test_no_v2_limit_is_none(self, dev, tmp_path):
        (tmp_path / "memory.max").write_text("max\n", encoding="ascii")
        assert dev._container_free_gb(tmp_path) is None

    def test_a_v1_limit_less_what_it_uses_and_its_none_is_none(self, dev, tmp_path):
        (tmp_path / "memory").mkdir()
        limit = tmp_path / "memory" / "memory.limit_in_bytes"
        limit.write_text(f"{12 * 2**30}\n", encoding="ascii")
        used = tmp_path / "memory" / "memory.usage_in_bytes"
        used.write_text(f"{2 * 2**30}\n", encoding="ascii")
        assert dev._container_free_gb(tmp_path) == 10.0
        limit.write_text("9223372036854771712\n", encoding="ascii")
        assert dev._container_free_gb(tmp_path) is None

    def test_no_control_group_is_none(self, dev, tmp_path):
        assert dev._container_free_gb(tmp_path) is None

    def test_what_linux_says_is_available(self, dev, tmp_path):
        meminfo = tmp_path / "meminfo"
        meminfo.write_text(
            "MemTotal:       65536000 kB\nMemFree:  1000 kB\nMemAvailable:   20971520 kB\n",
            encoding="ascii",
        )
        assert dev._available_gb(meminfo) == 20.0
        assert dev._available_gb(tmp_path / "missing") is None
