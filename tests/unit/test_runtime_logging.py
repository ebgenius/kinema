"""Silencing the vendored solver's logging without silencing everyone else's.

jaxls emits four INFO lines from every ``LeastSquaresProblem.analyze()``, which
is intolerable from a viewport handler firing on mouse move. Muting it used to
be ``logger.remove()`` -- but loguru's logger is a process-wide singleton shared
with every other add-on in the Blender session, so that removed *their* sinks
too. Any other add-on using loguru went silent the moment Kinema loaded.
"""

from __future__ import annotations

import pytest

from ..conftest import load_addon_module

runtime = load_addon_module("runtime")
logger = pytest.importorskip("loguru").logger


@pytest.fixture
def sink():
    """A sink registered *before* Kinema touches logging, like another add-on's."""
    records: list[str] = []
    sink_id = logger.add(records.append, level="DEBUG")
    for name in runtime._LOGURU_MODULES:
        logger.enable(name)
    yield records
    logger.remove(sink_id)
    for name in runtime._LOGURU_MODULES:
        logger.enable(name)


def test_a_foreign_sink_survives(sink):
    """The actual bug: another add-on's logging must keep working."""
    runtime.silence_vendor_logging()
    logger.info("a message from some other add-on")
    assert any("some other add-on" in record for record in sink)


def log_as(module_name: str, message: str) -> None:
    """Emit a record loguru will attribute to ``module_name``.

    ``logger.disable`` filters on the calling frame's ``__name__``, read at call
    time -- ``patch()`` rewrites the record afterwards and is not enough to fake
    it. So build a function whose globals really carry that name.
    """
    namespace = {"__name__": module_name, "logger": logger}
    exec("def emit(text):\n    logger.info(text)", namespace)  # noqa: S102
    namespace["emit"](message)


def test_vendor_modules_are_silenced(sink):
    for name in runtime._LOGURU_MODULES:
        log_as(name, "solver noise")
    assert len(sink) == len(runtime._LOGURU_MODULES), "sanity: unfiltered records arrive"

    sink.clear()
    runtime.silence_vendor_logging()
    for name in runtime._LOGURU_MODULES:
        log_as(name, "solver noise")
    assert sink == []


def test_debug_re_enables_them(sink):
    runtime.silence_vendor_logging()
    runtime.silence_vendor_logging(debug=True)
    log_as(runtime._LOGURU_MODULES[0], "solver noise")
    assert any("solver noise" in record for record in sink)


def test_missing_loguru_is_survivable(monkeypatch):
    """Kinema falls back to a NumPy solver when the stack is unavailable, so
    this must not raise on the way there."""
    import builtins

    real_import = builtins.__import__

    def no_loguru(name, *args, **kwargs):
        if name == "loguru":
            raise ImportError("no loguru")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_loguru)
    runtime.silence_vendor_logging()
