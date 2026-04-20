import json
import os
from shortuuid import uuid

from constraint_generation import solve_constraint_generation
from utils import parse_json_to_dcop_and_overlaps
from iterative_pricing import solve_iterative_pricing
from evaluation.algo_configs import get_display_name
import argparse


"""
ITAI: something really weird happens where when running this script pydcop runs an entire dcop for a min. Not sure what this is. It happens before any imports or anything...
"""


def main(
    scenario,
    output_json,
    algorithms_json,
    pydcop_mode,
    timeout,
    max_iterations,
    framework,
):
    assert not os.path.exists(output_json), f"{output_json} already exists"
    run_info = {
        "scenario": scenario,
        "algorithms_json": algorithms_json,
    }
    working_dir = os.path.join("output", f"{framework}_runs")
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    # temp_json = os.path.join( output_dir,"dcop_global.json")
    with open(algorithms_json, "r") as f:
        algorithms = json.load(f)
    alg_to_info = dict()
    for algorithm_config in algorithms:
        algo_name = get_display_name(algorithm_config)

        temp_output_json = os.path.join(working_dir, f"temp_alg_data_{uuid()}.json")
        print(f"running algorithm {algo_name}, saving to {temp_output_json}")

        # Parse JSON
        (
            pydcop_dict,
            agent_tasks,
            agent_downlinks,
            agent_capacities,
            var_to_details,
            requests,
        ) = parse_json_to_dcop_and_overlaps(scenario)
        if framework == "iterative_pricing":
            best_total_scheduled, best_iter, _ = solve_iterative_pricing(
                pydcop_dict,
                agent_tasks,
                agent_downlinks,
                agent_capacities,
                var_to_details,
                requests,
                algorithm_config,
                timeout=timeout,
                pydcop_mode=pydcop_mode,
                max_iterations=max_iterations,
                output_json=temp_output_json,
                working_dir=working_dir,
                clear_temp_files=True,
            )
        elif framework == "constraint_generation":
            best_total_scheduled, best_iter, _ = solve_constraint_generation(
                pydcop_dict,
                agent_tasks,
                agent_downlinks,
                agent_capacities,
                var_to_details,
                requests,
                algorithm_config,
                timeout=timeout,
                pydcop_mode=pydcop_mode,
                max_iterations=max_iterations,
                output_json=temp_output_json,
                working_dir=working_dir,
                clear_temp_files=True,
            )
        else:
            raise NotImplementedError
        print("result", best_total_scheduled)
        alg_to_info[algo_name] = {
            "file": temp_output_json,
            "aux_info": {"best_total_scheduled": best_total_scheduled, "best_iter": best_iter},
        }
    # save results
    output_dic = {"run_info": run_info, "algorithm_configs": algorithms, "output": dict()}

    for algo_name, info in alg_to_info.items():
        with open(info["file"], "r") as f:
            data = json.load(f)

        output_dic["output"][algo_name] = {
            "data": data,
            "aux_info": info["aux_info"],
        }

    with open(output_json, "w") as f:
        json.dump(output_dic, f, indent=2)
    # clear temp files
    for _, info in alg_to_info.items():
        os.remove(info["file"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="satellite_scheduling/test.json", type=str, help="scenario json to evaluate")
    p.add_argument("--output_json", default="output/test_main.json", type=str, help="output file to save results to")
    p.add_argument(
        "--algorithms_json",
        default="satellite_scheduling/rm_algorithm_configs.json",
        type=str,
        help="json with list of algorithm configs to test",
    )
    p.add_argument(
        "--framework",
        default="iterative_pricing",
        type=str,
        help="use iterative pricing or constraint generation",
        choices=["iterative_pricing", "constraint_generation"],
    )
    p.add_argument(
        "--pydcop_mode",
        default="thread",
        type=str,
        help="mode to run pydcop in (https://pydcop.readthedocs.io/en/latest/usage/cli/solve.html)",
        choices=["thread", "process"],
    )
    p.add_argument(
        "--timeout",
        default=-1,
        type=float,
        help="force timeout after this number of seconds (-1 for infinity)",
    )
    p.add_argument(
        "--max_iterations",
        default=4,
        type=int,
        help="number of iterations to run subgradient loop",
    )
    args = p.parse_args()
    main(
        scenario=args.scenario,
        output_json=args.output_json,
        algorithms_json=args.algorithms_json,
        pydcop_mode=args.pydcop_mode,
        timeout=args.timeout,
        max_iterations=args.max_iterations,
        framework=args.framework,
    )
