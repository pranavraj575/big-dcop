import subprocess
import json
import os
import numpy as np
import yaml
import argparse


def reformat_file_for_maxsum(problem_file):
    with open(problem_file, "r") as f:
        data = yaml.safe_load(f)

    # Convert agents list to dict with capacity for maxsum
    if isinstance(data['agents'], list):
        print("Patching agents...")
        new_agents = {}
        for agent_name in data['agents']:
            new_agents[agent_name] = {'capacity': 1000}
        data['agents'] = new_agents

        # Save
        with open(problem_file, "w") as f:
            yaml.dump(data, f, sort_keys=False)


problems = [
    "evaluation/graph_coloring_3agts.yaml",
    "evaluation/graph_coloring_50.yaml"
]

# Common algorithms supported by pyDCOP
algorithms = [
    "ftrl",
    "ftrl(context_based:1)",
    "ftrl(context_based:1,update_prob:0.95)",
    "regret_matching",
    "regret_matching(context_based:1)",
    "regret_matching(context_based:1,update_prob:0.95)",
    "regret_matching(rm_plus:1)",
    "regret_matching(rm_plus:1,predictive:1)",
    "regret_matching(predictive:1)",
    "regret_matching(rm_plus:1,predictive:1,update_prob:0.95)",
    "regret_matching(predictive:1,update_prob:0.95)",
    "dpop",
    "dsa",
    "mgm",
    "maxsum"
]


def run_pydcop(args, problem_file, algo):
    """
    Runs pydcop solve for a specific problem and algorithm
    Returns JSON result or error info
    """
    print(f"--> Running {algo} on {problem_file}...")
    if '(' in algo:
        alg_name = algo[:algo.index('(')]
        alg_parameters = algo[algo.index('(') + 1:-1].split(',')
    else:
        alg_name = algo
        alg_parameters = []

    cmd = [
        "pydcop",
        "--timeout", str(args.timeout),
        "solve",
        "--algo", alg_name,
        problem_file
    ]

    # for maxsum, need to make adhoc
    if alg_name == "maxsum":
        cmd.insert(4, "--dist")
        cmd.insert(5, "adhoc")

    # iterations for local search algs
    if alg_name in ["dsa", "mgm", "regret_matching", 'ftrl']:
        # Add algo_params to stop after fixed cycles if timeout doesn't kill it first
        cmd.extend(["--algo_params", "stop_cycle:50"])
    for param in alg_parameters:
        cmd.extend(["--algo_params", param])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        # read logs
        output_lines = result.stdout.strip().split('\n')
        full_output = result.stdout

        # heuristic to find JSON start/end if mixed with logs
        try:
            # Attempt to parse the whole output 
            data = json.loads(full_output)
            return data
        except json.JSONDecodeError:
            print(f"    [Warning] raw JSON parse failed, output might contain logs.")
            return {"status": "PARSE_ERROR", "raw_output": full_output[:200] + "..."}

    except subprocess.CalledProcessError as e:
        print(f"    [Error] Command failed with exit code {e.returncode}")
        print(f"    Error message: {e.stderr}")
        return {"status": "FAILED", "error": e.stderr}


def main(args):
    all_results = {}
    keys_to_store = ("status", "cost", "time", "msg_count", "cycle")
    scalar_keys = ("cost", "time", "msg_count", "cycle")

    for problem in args.problems:
        if not os.path.exists(problem):
            print(f"Skipping {problem} (file not found)")
            continue

        all_results[problem] = {}

        for algo in args.algorithms:
            # run algorithm
            all_summaries = []
            for _ in range(args.trials):
                data = run_pydcop(args, problem, algo)
                # store results
                if "assignment" in data:
                    summary = {
                        key: data.get(key) for key in keys_to_store
                    }
                    print(f"    Success! Cost: {summary['cost']}, Time: {summary['time']:.4f}s")
                else:
                    summary = {"error": "No assignment found", "raw": data}
                    print("    Failed to get valid assignment.")
                all_summaries.append(summary)
            overall_summary = dict()
            for key in scalar_keys:
                arr = [float(sm[key]) for sm in all_summaries if key in sm]
                if arr:
                    overall_summary[key] = {'mean': np.mean(arr), 'std': np.std(arr), 'n': len(arr)}
                p = len(arr)/args.trials
                overall_summary['proportion_completed'] = {'mean': p, 'std': np.sqrt(p*(1 - p)), 'n': args.trials}
            all_results[problem][algo] = {'summary': overall_summary, 'trials': all_summaries}
            print('    Stats across all trials:', overall_summary)

    # save all details
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\Run complete. Results saved to {args.output_file}")


if __name__ == "__main__":
    reformat_file_for_maxsum("evaluation/graph_coloring_50.yaml")
    # Configuration for runs
    p = argparse.ArgumentParser()
    p.add_argument("--trials",
                   required=False,
                   type=int,
                   default=30,
                   help="number of trials to run",
                   )
    p.add_argument("--timeout",
                   required=False,
                   type=int,
                   default=10,
                   help="seconds before timeout",
                   )
    p.add_argument('--algorithms',
                   required=False,
                   nargs="*",
                   default=algorithms,
                   type=str,
                   help='algorithms to include',
                   )
    p.add_argument('--problems',
                   required=False,
                   nargs="*",
                   default=problems,
                   type=str,
                   help='problem (.yaml files) to include',
                   )
    p.add_argument('--output_file',
                   required=False,
                   default="output/results.json",
                   type=str,
                   help='output file',
                   )

    main(p.parse_args())
