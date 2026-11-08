"""Run every test module except three explicitly dataset-dependent suites."""

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_MODULES = {
    "test_cycle2_reserve": "ImageBank loads the original training datasets to check reserve exclusions",
    "test_data": "ImageBank loads the original training datasets to check preprocessing and splits",
    "test_reserve": "ImageBank loads the original training datasets to check reserve sampling",
}


def main():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    modules = sorted(path.stem for path in (PROJECT_ROOT / "tests").glob("test_*.py") if path.is_file())
    if not modules:
        raise RuntimeError("No test modules were found")
    for name, reason in sorted(EXCLUDED_MODULES.items()):
        print(f"EXCLUDED tests.{name}: {reason}", flush=True)
    selected = [name for name in modules if name not in EXCLUDED_MODULES]
    print(f"Running {len(selected)} of {len(modules)} test modules; only the three named dataset suites are excluded.", flush=True)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite(loader.loadTestsFromName("tests." + name) for name in selected)
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    print(f"Offline result: tests={result.testsRun}, failures={len(result.failures)}, errors={len(result.errors)}, skipped={len(result.skipped)}", flush=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
