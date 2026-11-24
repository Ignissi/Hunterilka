"""Run documented application checks: python -m tests.coursework [report.json]."""

import argparse
import json
import platform
from pathlib import Path

from tests.coursework_input import file_checks, threshold_checks
from tests.coursework_paths import path_checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, nargs="?")
    args = parser.parse_args()
    paths, coverage = path_checks()
    suites = {"file": file_checks(), "thresholds": threshold_checks(), "paths": paths}
    report = {"python": platform.python_version(), "os": platform.platform(),
              "machine": platform.machine(), **suites, "coverage": coverage}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ok = True
    for group, cases in suites.items():
        failed = [c["id"] for c in cases if not c["passed"]]
        print(f"{group}: {len(cases) - len(failed)}/{len(cases)}", ", ".join(failed))
        ok = ok and not failed
    for decision, outcomes in coverage.items():
        if not all(outcomes.values()):
            print(f"Missing outcome for {decision}: {outcomes}")
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
