import logging
import math
import json
import os
import copy
import time
from collections import defaultdict
from scheduler import solve_local_schedule
import utils

logger = logging.getLogger(__name__)


def update_dcop_utilities(base_pydcop_dict, lambda_penalties, var_to_details):
    """
    Deep-copies base_pydcop_dict and appends unary penalty constraints for any
    (agent_id, req_id) pair that has a positive accumulated penalty.

    Returns a new pydcop_dict — base_pydcop_dict is not modified.
    """
    current = copy.deepcopy(base_pydcop_dict)
    details_to_var = {(d["agent_id"], d["req_id"]): v for v, d in var_to_details.items()}

    for (agent_id, req_id), penalty in lambda_penalties.items():
        if penalty <= 0.0:
            continue
        var_name = details_to_var.get((agent_id, req_id))
        if not var_name:
            continue
        # Capture penalty value in closure: returns -penalty if var=1, else 0.
        p = penalty
        current["constraints"][f"penalty_{var_name}"] = {
            "variables": [var_name],
            "fn": lambda vi, a, _p=p: -_p * a[vi[0]],
        }

    return current


def solve_iterative_pricing(
    pydcop_dict,
    agent_tasks,
    agent_downlinks,
    agent_capacities,
    var_to_details,
    requests,
    algorithm_config,
    pydcop_mode=None,  # unused — kept for API compatibility
    timeout=-1,  # unused — kept for API compatibility
    output_json=None,
    ignore_keys=("agt_metrics",),
    clear_temp_files=None,  # unused — kept for API compatibility
    temp_json=None,  # unused — kept for API compatibility
    working_dir=None,  # unused — kept for API compatibility
    max_iterations=5,
    step_size_c=15.0,
):
    if output_json is not None:
        assert not os.path.exists(output_json), f"output file {output_json} already exists"

    lambda_penalties = defaultdict(float)
    best_total_scheduled = 0
    best_iteration = 0
    run_data = []  # in-memory iteration results (no temp files)
    utility_per_iter = []
    runtime_per_iter = []
    messages_per_iter = []
    total_messages = 0
    t_start = time.time()

    for iteration in range(max_iterations):
        iter_t_start = time.time()
        current_pydcop_dict = update_dcop_utilities(pydcop_dict, lambda_penalties, var_to_details)
        result = utils.run_global_dispatcher_cosp(current_pydcop_dict, algorithm_config)

        run_data.append({k: v for k, v in result.items() if k not in ignore_keys})
        total_messages += result.get("run_info", {}).get("total_messages", 0)
        messages_per_iter.append(result.get("run_info", {}).get("total_messages", 0))

        assignments = result["assignment"]

        # Group assigned variables by agent
        agent_assigned_reqs = defaultdict(set)
        for var_name, value in assignments.items():
            if int(value) == 1 and var_name in var_to_details:
                d = var_to_details[var_name]
                agent_assigned_reqs[d["agent_id"]].add(d["req_id"])

        global_scheduled_reqs = set()
        total_dropped = 0
        alpha = step_size_c / math.sqrt(iteration + 1)

        for agent_id, assigned_reqs in agent_assigned_reqs.items():
            if not assigned_reqs:
                continue

            scheduled_reqs, _ = solve_local_schedule(
                agent_id,
                assigned_reqs,
                agent_tasks[agent_id],
                agent_downlinks[agent_id],
                agent_capacities[agent_id],
            )
            global_scheduled_reqs.update(scheduled_reqs)

            dropped = assigned_reqs - scheduled_reqs
            total_dropped += len(dropped)
            for req_id in dropped:
                lambda_penalties[(agent_id, req_id)] += alpha

        true_total = len(global_scheduled_reqs)
        utility_per_iter.append(true_total / len(requests))
        logger.info(f"Iteration {iteration}: {true_total}/{len(requests)} scheduled, {total_dropped} dropped")

        if true_total > best_total_scheduled:
            best_total_scheduled = true_total
            best_iteration = iteration
        runtime_per_iter.append(time.time() - iter_t_start)

        if total_dropped == 0:
            break

    if output_json is not None:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(run_data, f, indent=2)

    runtime_s = time.time() - t_start
    return (
        best_total_scheduled / len(requests),
        best_iteration,
        {
            "total_messages": total_messages,
            "runtime_s": runtime_s,
            "utility_per_iter": utility_per_iter,
            "runtime_per_iter": runtime_per_iter,
            "messages_per_iter": messages_per_iter,
        },
    )
