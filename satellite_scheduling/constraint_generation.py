import json
import os
import time
from collections import defaultdict
from scheduler import solve_local_schedule, get_constraints, add_constraints
import utils


def solve_constraint_generation(
    pydcop_dict,
    agent_tasks,
    agent_downlinks,
    agent_capacities,
    var_to_details,
    requests,
    algorithm_config,
    pydcop_mode=None,       # unused — kept for API compatibility
    ignore_keys=("agt_metrics",),
    timeout=-1,             # unused — kept for API compatibility
    output_json=None,
    clear_temp_files=None,  # unused — kept for API compatibility
    temp_json=None,         # unused — kept for API compatibility
    working_dir=None,       # unused — kept for API compatibility
    max_iterations=5,
):
    if output_json is not None:
        assert not os.path.exists(output_json), f"output file {output_json} already exists"

    num_constraints_added = 0
    best_total_scheduled = 0
    best_iteration = 0
    iteration = 0
    run_data = []        # in-memory iteration results (no temp files)
    utility_per_iter = []
    total_messages = 0
    t_start = time.time()

    while iteration < max_iterations:

        result = utils.run_global_dispatcher_cosp(pydcop_dict, algorithm_config)

        run_data.append({k: v for k, v in result.items() if k not in ignore_keys})
        total_messages += result.get("run_info", {}).get("total_messages", 0)

        assignments = result["assignment"]

        # Group assigned variables by agent
        agent_assigned_reqs = defaultdict(set)
        for var_name, value in assignments.items():
            if int(value) == 1 and var_name in var_to_details:
                d = var_to_details[var_name]
                agent_assigned_reqs[d["agent_id"]].add(d["req_id"])

        global_scheduled_reqs = set()
        constraints = []

        for agent_id, assigned_reqs in agent_assigned_reqs.items():
            if not assigned_reqs:
                continue


            scheduled_reqs, final_schedule = solve_local_schedule(
                agent_id,
                assigned_reqs,
                agent_tasks[agent_id],
                agent_downlinks[agent_id],
                agent_capacities[agent_id],
            )
            global_scheduled_reqs.update(scheduled_reqs)

            constraints.extend(get_constraints(
                assigned_reqs=assigned_reqs,
                scheduled_reqs=scheduled_reqs,
                agent_id=agent_id,
                pydcop_dict=pydcop_dict,
            ))

        true_total = len(global_scheduled_reqs)
        utility_per_iter.append(true_total / len(requests))
        if true_total > best_total_scheduled:
            best_total_scheduled = true_total
            best_iteration = iteration

        pydcop_dict, new_num = add_constraints(
            constraints=constraints,
            counter=num_constraints_added,
            pydcop_dict=pydcop_dict,
        )

        iteration += 1
        if num_constraints_added == new_num:
            break
        num_constraints_added = new_num

    if output_json is not None:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(run_data, f, indent=2)

    runtime_s = time.time() - t_start
    return best_total_scheduled / len(requests), best_iteration, {
        "total_messages": total_messages,
        "runtime_s": runtime_s,
        "utility_per_iter": utility_per_iter,
    }
