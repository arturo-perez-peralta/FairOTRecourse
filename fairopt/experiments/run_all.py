#!/usr/bin/env python3
"""Run all fairness experiments (F1-F3) sequentially.

F1: individual fairness, F2: group fairness, F3: mixed (group + individual,
enhanced solver).  F1/F2 run 10-fold CV; F3 uses its own warm-started runner.

Docker default entrypoint: ``python -m fairopt.experiments.run_all``.

NOTE: F3 sweeps 14 lambda_ind x 34 lambda_g x 10 folds across 7 datasets and can
take hours-to-days.  Prefer running it separately with ``--resume``:
``python -m fairopt.experiments.f3_mixed_fairness --resume --device cpu``.
"""
import sys
import time
import traceback
from pathlib import Path

LOG = Path(__file__).resolve().parents[2] / "results" / "fairness_run_all.log"


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")
        f.flush()


def run_experiment(name: str, import_path: str, run_fn: str, **kwargs) -> bool:
    log(f"=== STARTING {name} ===")
    start = time.time()
    try:
        __import__(import_path, fromlist=[run_fn])
        mod = sys.modules[import_path]
        runner = getattr(mod, run_fn)
        runner(**kwargs)
        elapsed = time.time() - start
        log(f"=== FINISHED {name} in {elapsed:.1f}s ===")
        return True
    except Exception as e:
        elapsed = time.time() - start
        log(f"=== FAILED {name} after {elapsed:.1f}s: {e} ===")
        traceback.print_exc()
        return False


def main() -> None:
    log("=" * 60)
    log("RUNNING ALL FAIRNESS EXPERIMENTS SEQUENTIALLY")
    log("=" * 60)

    experiments = [
        ("F1 Individual Fairness (CV)", "fairopt.experiments.f1_individual_fairness", "run", {"use_cv": True}),
        ("F2 Group Fairness (CV)", "fairopt.experiments.f2_group_fairness", "run", {"use_cv": True}),
        ("F3 Mixed Fairness (enhanced)", "fairopt.experiments.f3_mixed_fairness", "main", {}),
    ]

    successes = 0
    failures = 0

    for name, import_path, run_fn, kwargs in experiments:
        ok = run_experiment(name, import_path, run_fn, **kwargs)
        if ok:
            successes += 1
        else:
            failures += 1

    log("=" * 60)
    log(f"ALL DONE: {successes} succeeded, {failures} failed")
    log("=" * 60)


if __name__ == "__main__":
    main()
