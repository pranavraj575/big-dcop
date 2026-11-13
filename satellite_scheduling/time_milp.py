"""Time every OR-Tools MILP (solve_local_schedule) call made during the experiments.

Wraps scheduler.solve_local_schedule with a timer, runs the experiment pipeline
(main.main) over every scenario / framework / algorithm, and reports aggregate
timing stats for the MILP solves.
"""
import argparse
import glob
import os
import statistics
import time

import scheduler
import iterative_pricing
import constraint_generation
import main as main_mod

TIMINGS = []

_real = scheduler.solve_local_schedule


def _timed(*args, **kwargs):
    t0 = time.perf_counter()
    try:
        return _real(*args, **kwargs)
    finally:
        TIMINGS.append(time.perf_counter() - t0)


# Patch every reference (the function is imported by-name into these modules).
scheduler.solve_local_schedule = _timed
iterative_pricing.solve_local_schedule = _timed
constraint_generation.solve_local_schedule = _timed


def run(scenarios_dir, algorithms_json, frameworks, max_iterations, step_size_c):
    scenario_paths = sorted(glob.glob(os.path.join(scenarios_dir, "scenario_*.json")))
    for framework in frameworks:
        for scenario in scenario_paths:
            main_mod.main(
                scenario=scenario,
                output_json=os.path.join("/tmp/milp_timing_out", framework,
                                         os.path.basename(scenario)),
                algorithms_json=algorithms_json,
                max_iterations=max_iterations,
                framework=framework,
                step_size_c=step_size_c,
            )


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default=os.path.join(here, "scenarios_larger"))
    ap.add_argument("--algorithms", default=os.path.join(here, "algo_configs/cosp_algorithm_configs.json"))
    ap.add_argument("--max-iterations", type=int, default=4)
    ap.add_argument("--step-size-c", type=float, default=1.8)
    ap.add_argument("--frameworks", nargs="+", default=["iterative_pricing", "constraint_generation"])
    args = ap.parse_args()

    t_start = time.perf_counter()
    run(args.scenarios, args.algorithms, args.frameworks, args.max_iterations, args.step_size_c)
    wall = time.perf_counter() - t_start

    n = len(TIMINGS)
    total = sum(TIMINGS)
    print("\n" + "=" * 60)
    print("OR-Tools MILP (solve_local_schedule / SCIP) timing")
    print("=" * 60)
    print(f"  scenarios dir : {args.scenarios}")
    print(f"  frameworks    : {', '.join(args.frameworks)}")
    print(f"  MILP calls    : {n}")
    print(f"  total MILP time : {total:.3f} s  ({total / wall:.1%} of {wall:.1f}s wall)")
    if n:
        print(f"  AVERAGE       : {total / n * 1000:.3f} ms per solve")
        print(f"  median        : {statistics.median(TIMINGS) * 1000:.3f} ms")
        print(f"  min / max     : {min(TIMINGS) * 1000:.3f} / {max(TIMINGS) * 1000:.3f} ms")
        if n > 1:
            print(f"  stdev         : {statistics.stdev(TIMINGS) * 1000:.3f} ms")
    print("=" * 60)
