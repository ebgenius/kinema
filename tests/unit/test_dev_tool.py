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


class TestMergingDurations:
    PARTS = [
        {"collected": ["a", "b", "c"], "times": {"a": 2.0}},
        {"collected": ["a", "b", "c"], "times": {"b": 3.0}},
    ]

    def test_the_whole_suite_drops_a_test_that_is_gone(self, dev):
        merged = dev._merged_durations({"a": 1.0, "c": 4.0, "gone": 9.0}, self.PARTS, whole=True)
        assert merged == {"a": 2.0, "b": 3.0, "c": 4.0}

    def test_part_of_it_keeps_the_rest(self, dev):
        merged = dev._merged_durations({"a": 1.0, "gone": 9.0}, self.PARTS, whole=False)
        assert merged == {"a": 2.0, "b": 3.0, "gone": 9.0}

    def test_no_process_reporting_drops_nothing(self, dev):
        assert dev._merged_durations({"a": 1.0}, [], whole=True) == {"a": 1.0}


class TestContainerMemory:
    def test_a_v2_limit_is_read(self, dev, tmp_path):
        (tmp_path / "memory.max").write_text(f"{8 * 2**30}\n", encoding="ascii")
        assert dev._container_limit_gb(tmp_path) == 8.0

    def test_no_v2_limit_is_none(self, dev, tmp_path):
        (tmp_path / "memory.max").write_text("max\n", encoding="ascii")
        assert dev._container_limit_gb(tmp_path) is None

    def test_a_v1_limit_is_read_and_its_none_is_none(self, dev, tmp_path):
        (tmp_path / "memory").mkdir()
        path = tmp_path / "memory" / "memory.limit_in_bytes"
        path.write_text(f"{12 * 2**30}\n", encoding="ascii")
        assert dev._container_limit_gb(tmp_path) == 12.0
        path.write_text("9223372036854771712\n", encoding="ascii")
        assert dev._container_limit_gb(tmp_path) is None

    def test_no_control_group_is_none(self, dev, tmp_path):
        assert dev._container_limit_gb(tmp_path) is None
