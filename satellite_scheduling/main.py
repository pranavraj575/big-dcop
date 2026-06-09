import json
import os

from constraint_generation import solve_constraint_generation
from utils import parse_json_to_dcop_and_overlaps
from iterative_pricing import solve_iterative_pricing
import argparse


def _get_display_name(algorithm_config: dict) -> str:
    return algorithm_config.get("display_name") or algorithm_config.get("plt_name") or algorithm_config["name"]


def main(
    scenario,
    output_json,
    algorithms_json,
    max_iterations,
    framework,
):
    assert not os.path.exists(output_json), f"{output_json} already exists"
    run_info = {
        "scenario": scenario,
        "algorithms_json": algorithms_json,
    }
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    with open(algorithms_json, "r") as f:
        algorithms = json.load(f)

    output_dic = {"run_info": run_info, "algorithm_configs": algorithms, "output": {}}

    for algorithm_config in algorithms:
        algo_name = _get_display_name(algorithm_config)
        print(f"\nRunning algorithm: {algo_name}")

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
                max_iterations=max_iterations,
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
                max_iterations=max_iterations,
            )
        else:
            raise NotImplementedError(f"Unknown framework: {framework}")

        print(f"Result for {algo_name}: {best_total_scheduled:.1%} (best at iteration {best_iter})")
        output_dic["output"][algo_name] = {
            "aux_info": {"best_total_scheduled": best_total_scheduled, "best_iter": best_iter},
        }

    with open(output_json, "w") as f:
        json.dump(output_dic, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="satellite_scheduling/test.json", type=str, help="scenario json to evaluate")
    p.add_argument("--output_json", default="output/test_main.json", type=str, help="output file to save results to")
    p.add_argument(
        "--algorithms_json",
        default="satellite_scheduling/cosp_algorithm_configs.json",
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
        "--max_iterations",
        default=4,
        type=int,
        help="number of outer iterations",
    )
    args = p.parse_args()
    main(
        scenario=args.scenario,
        output_json=args.output_json,
        algorithms_json=args.algorithms_json,
        max_iterations=args.max_iterations,
        framework=args.framework,
    )
