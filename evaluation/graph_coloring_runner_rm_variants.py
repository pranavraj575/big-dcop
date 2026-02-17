import subprocess
import os
import glob
import argparse
import pandas as pd
import json

from eval_helpers import reformat_file_for_maxsum, extract_json_from_output
from algo_configs import get_algo_info, get_display_name


def run_pydcop(problem_file, algo_config, args):
    """
    Runs pydcop solve.
    Returns a df(of mid run metrics), dict(of final metrics).
    """
    base_name, extra_alg_params = get_algo_info(algo_config)

    # build Command
    cmd = [
        "pydcop", "--timeout", str(args.timeout), "solve",
        problem_file,
        "--algo", base_name,
    ]

    # append the specific args from the dict
    cmd.extend(extra_alg_params)

    if args.collect_on is not None:
        cmd.extend(['--collect_on', args.collect_on])
        if args.period is not None:
            cmd.extend(['--period', str(args.period)])
        cmd.extend(['--run_metrics', args.temp_csv])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False  # We handle return codes manually
        )

        # Parse Output
        final_data = extract_json_from_output(result.stdout)
        if args.collect_on is not None:
            mid_df = pd.read_csv(args.temp_csv)
        else:
            mid_df = None
        if final_data and "assignment" in final_data:
            return mid_df, {
                "status": "1",  # 1 = success
                "cost": float(final_data.get("cost", -1)),
                "time": float(final_data.get("time", -1)),
                "msg_count": int(final_data.get("msg_count", 0)),
                "cycles": int(final_data.get("cycles", 0)),
                "assignment": str(final_data.get("assignment", {}))  # store as string to fit in DF
            }
        else:
            # failed or timed out
            err_msg = result.stderr.strip() if result.stderr else result.stdout.strip()[-200:]
            return mid_df, {"status": "FAILED", "error": err_msg}

    except Exception as e:
        return None, {"status": "ERROR", "error": str(e)}


def main():
    DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Run pyDCOP benchmark")
    parser.add_argument("--algorithms", type=str, default=os.path.join(DIR, "evaluation", "configs", "rm_configs.json"),
                        help="json file with algorithm configs to use")
    parser.add_argument("--input_dir", type=str, default=os.path.join(DIR, "output", "graph_coloring_instances_hard"),
                        help="Directory containing .yaml problem files")
    parser.add_argument("--output_csv", type=str, default=os.path.join(DIR, "output", "results_rm.csv"),
                        help="Path to save the results DataFrame")
    parser.add_argument("--temp_csv", type=str, default=os.path.join(DIR, "output", "temp_rm.csv"),
                        help="Path to save mid-run results into (cleared after running each experiemnt)")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials per algorithm per problem")
    parser.add_argument("--timeout", type=float, default=30.,
                        help="Timeout in seconds per run")
    parser.add_argument(
        "-c",
        "--collect_on",
        choices=["value_change", "cycle_change", "period"],
        default="value_change",
        help='collect mid-run data upon this event',
    )
    parser.add_argument(
        "--period",
        type=float,
        default=None,
        help="Period for collecting metrics. only available "
             "when using --collect_on period. Defaults to 1 "
             "second if not specified",
    )
    args = parser.parse_args()

    if not os.path.exists(args.algorithms):
        print(f"Error: Directory '{args.algorithms}' does not exist.")
        return
    with open(args.algorithms) as f:
        algorithms = json.load(f)
        f.close()

    if not os.path.exists(args.input_dir):
        print(f"Error: Directory '{args.input_dir}' does not exist.")
        return

    problem_files = sorted(glob.glob(os.path.join(args.input_dir, "*.yaml")))
    if not problem_files:
        print(f"No .yaml files found in {args.input_dir}")
        return

    # loop over problems
    added_header = False
    for p_idx, problem_path in enumerate(problem_files):
        problem_name = os.path.basename(problem_path)
        print(f"Processing [{p_idx + 1}/{len(problem_files)}] {problem_name}...")

        # apply fix for maxsum if needed
        reformat_file_for_maxsum(problem_path)

        for algo_idx, algo_config in enumerate(algorithms):
            display_name = get_display_name(algo_config)
            print(f"    Algorithm: {display_name}")
            for trial in range(args.trials):
                trial_records = []
                # Run the solver
                mid_df, final_metrics = run_pydcop(
                    problem_file=problem_path,
                    algo_config=algo_config,
                    args=args,
                )

                # if we are taking mid-run statistics,
                #   the last row will include final metrics
                if mid_df is not None:
                    for row in mid_df.itertuples():
                        mid_record = {
                            "problem": problem_name,
                            "algorithm": display_name,
                            "trial": trial + 1,
                            "status": row.status,
                            "cost": row.cost,
                            "time": row.time,
                            "msg_count": row.msg_count,
                            "cycles": row.cycle,
                            "timeout_param": args.timeout,
                            "error_msg": final_metrics.get("error", "")
                        }
                        trial_records.append(mid_record)
                else:
                    # Create a record row=
                    record = {
                        "problem": problem_name,
                        "algorithm": display_name,
                        "trial": trial + 1,
                        "status": final_metrics.get("status"),
                        "cost": final_metrics.get("cost"),
                        "time": final_metrics.get("time"),
                        "msg_count": final_metrics.get("msg_count"),
                        "cycles": final_metrics.get("cycles"),
                        "timeout_param": args.timeout,
                        "error_msg": final_metrics.get("error", "")
                    }
                    trial_records.append(record)
                print(
                    f"        Trial {trial + 1}: {final_metrics.get('status')},"
                    f" Cost: {final_metrics.get('cost', 'N/A')},"
                    f" Time: {final_metrics.get('time', 'N/A')}")
                error_msg = final_metrics.get("error", "")
                if error_msg:
                    print(f"        ERROR: {error_msg}")

                # save memory by not tracking records across all trials, just dump them into the csv after each trial
                df = pd.DataFrame(trial_records)
                # Ensure output directory exists
                os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
                df.to_csv(args.output_csv,
                          index=False,
                          mode='a' if added_header else 'w',
                          header=not added_header,
                          )
                added_header = True
                del df

    print(f"\nResults saved to {args.output_csv}")
    if os.path.exists(args.temp_csv):
        os.remove(args.temp_csv)


if __name__ == "__main__":
    main()
