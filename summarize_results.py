#!/usr/bin/env python3
"""
summarize_results.py

Reads all output/<framework>/scenario_*.json files and prints a summary table
showing per-algorithm averages (and optionally per-scenario detail).

Usage:
    python3 summarize_results.py [--output-dir OUTPUT_DIR] [--detail]

Options:
    --output-dir DIR   Root of the output tree (default: output)
    --detail           Also print per-scenario rows, not just averages
    --framework FW     Restrict to one framework (constraint_generation or
                       iterative_pricing); default: show both
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(val, fmt=".1%"):
    return format(val, fmt) if val is not None else "—"


def _col_widths(rows, headers):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    return widths


def _print_table(headers, rows, title=None):
    if title:
        print(f"\n{'─' * 2} {title} {'─' * max(0, 76 - len(title))}")
    widths = _col_widths(rows, headers)
    sep = "  ".join("─" * w for w in widths)
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))
    print()


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_results(output_dir: Path):
    """
    Returns a nested dict:
        data[framework][scenario_stem][algo_name] = aux_info dict
    """
    data = defaultdict(lambda: defaultdict(dict))
    for fw_dir in sorted(output_dir.iterdir()):
        if not fw_dir.is_dir():
            continue
        fw = fw_dir.name
        for json_path in sorted(fw_dir.glob("scenario_*.json")):
            try:
                with json_path.open() as f:
                    doc = json.load(f)
            except Exception as e:
                print(f"  [WARN] Could not parse {json_path}: {e}")
                continue
            scenario = json_path.stem
            for algo, algo_data in doc.get("output", {}).items():
                info = algo_data.get("aux_info", {})
                data[fw][scenario][algo] = info
    return data


# ---------------------------------------------------------------------------
# Summarise
# ---------------------------------------------------------------------------

def summarize(data, frameworks_filter=None, detail=False):
    frameworks = sorted(data.keys())
    if frameworks_filter:
        frameworks = [f for f in frameworks if f == frameworks_filter]

    for fw in frameworks:
        scenarios = data[fw]
        # Collect all algorithm names (preserve order of first occurrence)
        algos = []
        seen = set()
        for scenario_algos in scenarios.values():
            for a in scenario_algos:
                if a not in seen:
                    algos.append(a)
                    seen.add(a)

        # ── per-scenario detail ──────────────────────────────────────────
        if detail:
            headers = ["scenario"] + algos
            rows = []
            for scenario in sorted(scenarios.keys(),
                                   key=lambda s: int(s.replace("scenario_", ""))):
                row = [scenario]
                for algo in algos:
                    info = scenarios[scenario].get(algo, {})
                    sched = info.get("best_total_scheduled")
                    row.append(_fmt(sched) if sched is not None else "—")
                rows.append(row)
            _print_table(headers, rows, title=f"{fw}  —  scheduled % per scenario")

        # ── averages ────────────────────────────────────────────────────
        headers = ["algorithm", "avg scheduled %", "avg messages", "avg runtime (s)", "n scenarios"]
        rows = []
        for algo in algos:
            sched_vals, msg_vals, rt_vals = [], [], []
            for scenario_algos in scenarios.values():
                info = scenario_algos.get(algo, {})
                if "best_total_scheduled" in info:
                    sched_vals.append(info["best_total_scheduled"])
                if "total_messages" in info:
                    msg_vals.append(info["total_messages"])
                if "runtime_s" in info:
                    rt_vals.append(info["runtime_s"])
            n = len(sched_vals)
            avg_s = sum(sched_vals) / n if n else None
            avg_m = sum(msg_vals) / len(msg_vals) if msg_vals else None
            avg_r = sum(rt_vals) / len(rt_vals) if rt_vals else None
            rows.append([
                algo,
                _fmt(avg_s) if avg_s is not None else "—",
                f"{avg_m:,.0f}" if avg_m is not None else "—",
                f"{avg_r:.2f}s" if avg_r is not None else "—",
                str(n),
            ])

        # Sort by average scheduled % descending
        rows.sort(key=lambda r: float(r[1].rstrip("%")) if r[1] != "—" else 0, reverse=True)
        _print_table(headers, rows, title=f"{fw}  —  averages across {len(scenarios)} scenarios")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default="output", type=Path,
                   help="Root of the output tree (default: output)")
    p.add_argument("--detail", action="store_true",
                   help="Also print per-scenario rows")
    p.add_argument("--framework", default=None,
                   help="Restrict to one framework")
    args = p.parse_args()

    if not args.output_dir.exists():
        print(f"ERROR: output directory '{args.output_dir}' does not exist.")
        raise SystemExit(1)

    data = load_results(args.output_dir)
    if not data:
        print("No results found.")
        raise SystemExit(1)

    summarize(data, frameworks_filter=args.framework, detail=args.detail)


if __name__ == "__main__":
    main()
