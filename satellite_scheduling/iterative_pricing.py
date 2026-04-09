import math
from collections import defaultdict
from scheduler import solve_local_schedule
import copy
import os
from shortuuid import uuid
import utils
import json


def update_dcop_utilities(base_pydcop_dict, lambda_penalties, var_to_details):
    """
    Applies penalties to the DCOP utility structure.

    Args:
        base_pydcop_dict: The pristine, original DCOP dictionary (iteration 0).
        lambda_penalties: Dictionary of {(agent_id, req_id): cumulative_penalty}
        var_to_details: Mapping of DCOP variable names to agent/request details.

    Returns:
        A new PyDCOP dictionary with the penalized utilities for this iteration.
    """

    current_pydcop_dict = copy.deepcopy(base_pydcop_dict)

    #  (agent_id, req_id) -> var_name
    details_to_var = {(details["agent_id"], details["req_id"]): var_name for var_name, details in var_to_details.items()}

    # Unary Constraints for Penalties
    for (agent_id, req_id), penalty in lambda_penalties.items():
        if penalty <= 0.0:
            continue

        var_name = details_to_var.get((agent_id, req_id))
        if not var_name:
            continue

        # Create a unique constraint name for this specific penalty
        penalty_c_name = f"penalty_{var_name}"

        current_pydcop_dict["constraints"][penalty_c_name] = {
            "type": "intention",
            "variables": [var_name],
            # If v_x == 1, it adds (-penalty) to the global score.
            # If v_x == 0, it adds 0.
            "function": f"{-penalty} * {var_name}",
        }

    return current_pydcop_dict


def solve_iterative_pricing(
    pydcop_dict,
    agent_tasks,
    agent_downlinks,
    agent_capacities,
    var_to_details,
    requests,
    algorithm_config,
    pydcop_mode,
    timeout=-1,
    output_json=None,
    ignore_keys=("agt_metrics",),
    clear_temp_files=True,
    temp_json=None,
    working_dir="output/iterative_pricing_runs",
    max_iterations=5,
    step_size_c=10.0,  # Constant for step size
):
    """
    Args:
        temp_json:
        working_dir: dir to save runs to
        timeout: -1 if no timeout, otherwise # of seconds to set timeout to
    """

    if output_json is not None:
        assert not os.path.exists(output_json), f"output file {output_json} already exists"

    # Track  penalties: lambda_penalties[(agent_id, req_id)] = penalty_value
    lambda_penalties = defaultdict(float)

    # Track the incumbent (best) solution found across all iterations
    best_total_scheduled = 0
    best_iteration = 0

    os.makedirs(working_dir, exist_ok=True)
    run_files = []
    if temp_json is None:
        temp_json = os.path.join(working_dir, f"temp_json_{uuid()}.json")
    for iteration in range(max_iterations):
        print(f"\nIteration {iteration}")
        pydcop_results_json = os.path.join(working_dir, f"it_{iteration}_temp_{uuid()}.json")

        # This function subtracts the lambda penalty from the base utility
        current_pydcop_dict = update_dcop_utilities(pydcop_dict, lambda_penalties, var_to_details)
        # Run global DCOP
        print("Running global dispatch")
        utils.run_global_dispatcher(
            current_pydcop_dict, algorithm_config, temp_json, pydcop_results_json, pydcop_mode=pydcop_mode, timeout=timeout
        )
        print("Done global dispatch")
        run_files.append(pydcop_results_json)
        assignments = utils.load_assignments(pydcop_results_json)

        # Group assigned requests by agent
        agent_assigned_reqs = defaultdict(set)
        for var_name, value in assignments.items():
            if int(value) == 1 and var_name in var_to_details:
                details = var_to_details[var_name]
                agent_assigned_reqs[details["agent_id"]].add(details["req_id"])

        global_scheduled_reqs = set()
        total_dropped_this_iteration = 0

        #  dynamic learning rate (c / sqrt(t))
        alpha = step_size_c / math.sqrt(iteration + 1)

        for agent_id, assigned_reqs in agent_assigned_reqs.items():
            if not assigned_reqs:
                continue

            # Run the local scheduler
            scheduled_reqs, final_schedule = solve_local_schedule(
                agent_id,
                assigned_reqs,
                agent_tasks[agent_id],
                agent_downlinks[agent_id],
                agent_capacities[agent_id],
            )

            global_scheduled_reqs.update(scheduled_reqs)

            # Calculate subgradient
            dropped_reqs = assigned_reqs - scheduled_reqs
            total_dropped_this_iteration += len(dropped_reqs)

            # Update penalties for dropped requests
            for req_id in dropped_reqs:
                lambda_penalties[(agent_id, req_id)] += alpha * 1.0

            print(
                f"Agent {agent_id}: Assigned {len(assigned_reqs)} -> Scheduled {len(scheduled_reqs)} (Dropped {len(dropped_reqs)}, Penalties updated)"
            )

        #  check state state
        true_total_scheduled = len(global_scheduled_reqs)
        print(f"Network Total: {true_total_scheduled} of {len(requests)} physically scheduled.")

        if true_total_scheduled > best_total_scheduled:
            best_total_scheduled = true_total_scheduled
            best_iteration = iteration

        # Convergence Check
        if total_dropped_this_iteration == 0:
            # The DCOP assignment perfectly matches physical reality
            print("Nothing dropped...")
            break

    # manually write to a json, to avoid having to store a bunch of stuff in memory
    if output_json is not None:
        f = open(output_json, "a")
        f.write("[\n")
        for i, fn in enumerate(run_files):
            with open(fn, "r") as ff:
                temp = json.load(ff)
            if type(temp) is not dict:
                # this shouldn't run, but just in case it does
                temp = {"run_info": temp}
            temp = {k: v for k, v in temp.items() if k not in ignore_keys}
            json.dump(temp, f, indent=2)
            if i < len(run_files) - 1:
                f.write(",\n")
        f.write("\n]")
        f.close()
    if clear_temp_files:
        os.remove(temp_json)
        for fn in run_files:
            os.remove(fn)
        run_files = []
    return best_total_scheduled * 1.0 / len(requests), best_iteration, run_files
