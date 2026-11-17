"""Splitting the test suite across processes, by each test's last measured time.

``tools/dev.py test -j N`` runs N Blender processes at once, each with
``KINEMA_TEST_SHARD=i/N``; ``conftest.py`` keeps that process's share of what
it collected. Every process collects the same tests in the same order, so each
works out the same split on its own, and none needs telling which tests are
its.

The split deals the tests out longest first, each to the share with the least
time in it so far, from ``tests/durations.json``: the Optimize Motion tests
take most of the suite's time, a few of them nearly two minutes, and splitting
by count would leave one process with most of them. A test not measured yet
counts as the median of those that were.

Each share keeps its tests in the order they were collected, so a test's
neighbours change but not which comes first.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

#: The suite's folder.
TESTS = Path(__file__).resolve().parent
#: Each test's last measured time in seconds, by node id: ``dev.py test --record-durations``.
DURATIONS = TESTS / "durations.json"


def read_durations(path: Path = DURATIONS) -> dict[str, float]:
    """The recorded times, or none where there are none yet."""
    try:
        return {str(k): float(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    except (OSError, ValueError, TypeError, AttributeError):
        # Unreadable, or not a mapping of numbers: as if never recorded.
        return {}


def shares(nodeids: list[str], durations: dict[str, float], count: int) -> list[int]:
    """Which of ``count`` shares each of ``nodeids`` goes to, in their order."""
    if count < 1:
        raise ValueError(f"there have to be one or more shares, not {count}")
    known = [durations[n] for n in nodeids if n in durations]
    default = statistics.median(known) if known else 1.0
    cost = {n: durations.get(n, default) for n in nodeids}
    loads = [0.0] * count
    owner: dict[str, int] = {}
    for nodeid in sorted(nodeids, key=lambda n: (-cost[n], n)):
        least = min(range(count), key=lambda share: (loads[share], share))
        owner[nodeid] = least
        loads[least] += cost[nodeid]
    return [owner[n] for n in nodeids]


def covers_suite(args: list[str], base: Path, tests: Path = TESTS) -> bool:
    """Whether pytest was asked for all of the suite: one of its paths, ``args``
    relative to ``base``, is the tests folder or a folder holding it."""
    for arg in args:
        path = Path(base, arg.split("::")[0]).resolve()
        if path == tests or path in tests.parents:
            return True
    return False


def narrowed(getoption) -> bool:
    """Whether collecting left out tests in the paths asked for, before any hook
    could see them: ``--ignore``, ``--ignore-glob``, or ``--last-failed``, which
    skips the files with no failure. ``getoption`` is ``config.getoption``."""
    return any(getoption(name, None) for name in ("ignore", "ignore_glob", "lf"))


def parse_shard(text: str) -> tuple[int, int]:
    """``"i/n"`` as ``(i, n)``, with ``0 <= i < n``."""
    index, _, count = text.partition("/")
    i, n = int(index), int(count)
    if not 0 <= i < n:
        raise ValueError(f"shard {text!r}: the first number has to be from 0 to {n - 1}")
    return i, n
