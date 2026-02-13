import subprocess
import os
import glob
import argparse
import pandas as pd
import json
import itertools

from eval_helpers import reformat_file_for_maxsum, extract_json_from_output
from algo_configs import get_algo_info, get_display_name


def run_pydcop(problem_file, algo_config, timeout):
    """
    Runs pydcop solve.
    Returns a dictionary of metrics.
    """
    base_name, extra_alg_params = get_algo_info(algo_config)

    # build Command
    cmd = [
        "pydcop", "--timeout", str(timeout), "solve",
        "--algo", base_name,
        problem_file
    ]

    # append the specific args from the dict
    cmd.extend(extra_alg_params)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False  # We handle return codes manually
        )

        # Parse Output
        data = extract_json_from_output(result.stdout)

        if data and "assignment" in data:
            return {
                "status": "1",  # 1 = success
                "cost": float(data.get("cost", -1)),
                "time": float(data.get("time", -1)),
                "msg_count": int(data.get("msg_count", 0)),
                "cycles": int(data.get("cycles", 0)),
                "assignment": str(data.get("assignment", {}))  # store as string to fit in DF
            }
        else:
            # failed or timed out
            err_msg = result.stderr.strip() if result.stderr else result.stdout.strip()[-200:]
            return {"status": "FAILED", "error": err_msg}

    except Exception as e:
        return {"status": "ERROR", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Run pyDCOP benchmark")
    parser.add_argument("--algorithms", type=str, default="evaluation/algorithm_configs.json",
                        help="json file with algorithm configs to use")
    parser.add_argument("--input_dir", type=str, default="output/graph_coloring_instances",
                        help="Directory containing .yaml problem files")
    parser.add_argument("--output_csv", type=str, default="output/results.csv",
                        help="Path to save the results DataFrame")
    parser.add_argument("--trials", type=int, default=1, help="Number of trials per algorithm per problem")
    parser.add_argument("--timeout", type=float, default=[10.], nargs='+',
                        help="Timeout in seconds per run, use multiple values if you want to run with different paramters")
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

    all_records = []
    # loop over problems
    for p_idx, problem_path in enumerate(problem_files):
        problem_name = os.path.basename(problem_path)
        print(f"Processing [{p_idx + 1}/{len(problem_files)}] {problem_name}...")

        # apply fix for maxsum if needed
        reformat_file_for_maxsum(problem_path)

        for algo_config in algorithms:
            display_name = get_display_name(algo_config)
            print(f"    Algorithm: {display_name}")
            for timeout, trial in itertools.product(args.timeout, range(args.trials)):
                # Run the solver
                metrics = run_pydcop(
                    problem_file=problem_path,
                    algo_config=algo_config,
                    timeout=timeout,
                )

                # Create a record row
                record = {
                    "problem": problem_name,
                    "algorithm": display_name,
                    "trial": trial + 1,
                    "status": metrics.get("status"),
                    "cost": metrics.get("cost"),
                    "time": metrics.get("time"),
                    "msg_count": metrics.get("msg_count"),
                    "cycles": metrics.get("cycles"),
                    "timeout_param": timeout,
                    "error_msg": metrics.get("error", "")
                }

                all_records.append(record)
                print(
                    f"        Trial {trial + 1}: {metrics.get('status')},"
                    f" Cost: {metrics.get('cost', 'N/A')},"
                    f" Time: {metrics.get('time', 'N/A')})")
                error_msg = metrics.get("error", "")
                if error_msg:
                    print(f"        ERROR: {error_msg}")

    if all_records:
        df = pd.DataFrame(all_records)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

        df.to_csv(args.output_csv, index=False)
        print(f"\nResults saved to {args.output_csv}")


if __name__ == "__main__":
    main()
