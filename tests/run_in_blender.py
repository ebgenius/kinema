"""Run the test suite inside Blender, so ``bpy`` is the real thing.

Invoked by ``tools/dev.py test``. Anything importing ``bpy`` has to run in this
process; the pure-parser tests under ``tests/unit`` run faster outside Blender
via a plain ``uv run pytest``.

Blender always exits 0, so the pytest status is re-raised through
``sys.exit()`` to keep CI honest.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    try:
        import pytest
    except ImportError:
        print("[kinema-test] pytest not importable inside Blender.")
        print("[kinema-test] Run 'uv sync --group dev' so dev_bootstrap can add it.")
        sys.exit(1)

    # Everything after a bare "--" belongs to pytest.
    extra = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = extra or [str(REPO_ROOT / "tests")]
    args = ["-q", "--no-header", *args]

    print(f"[kinema-test] pytest {' '.join(args)}")
    sys.exit(pytest.main(args))


main()
