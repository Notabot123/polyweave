"""Minimal test runner (no pytest dependency).

Discovers ``tests/test_*.py`` modules, runs every top-level ``test_*`` function,
and reports a pass/fail summary. If pytest is installed, you can equally run::

    pytest

Run from the project root::

    python run_tests.py
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def discover_modules() -> list[str]:
    tests_dir = ROOT / "tests"
    return sorted(
        f"tests.{p.stem}"
        for p in tests_dir.glob("test_*.py")
    )


def main() -> int:
    passed = failed = 0
    failures: list[str] = []
    for mod_name in discover_modules():
        module = importlib.import_module(mod_name)
        fns = [
            getattr(module, n)
            for n in dir(module)
            if n.startswith("test_") and callable(getattr(module, n))
        ]
        for fn in fns:
            label = f"{mod_name}.{fn.__name__}"
            try:
                fn()
                passed += 1
                print(f"PASS  {label}")
            except Exception:  # noqa: BLE001 - report every failure
                failed += 1
                failures.append(label)
                print(f"FAIL  {label}")
                traceback.print_exc()
    print("\n" + "=" * 60)
    print(f"{passed} passed, {failed} failed")
    if failures:
        print("failed:")
        for f in failures:
            print(f"  - {f}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
