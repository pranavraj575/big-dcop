import json
import os
from collections import defaultdict
from scheduler import solve_local_schedule, get_constraints, add_constraints
import utils

from shortuuid import uuid


def solve_constraint_generation(
    pydcop_dict,
    agent_tasks,
    agent_downlinks,
    agent_capacities,
    var_to_details,
    requests,
    algorithm_config,
    pydcop_mode,
    ignore_keys=("agt_metrics",),
    timeout=-1,
    output_json=None,
    clear_temp_files=True,
    temp_json=None,
    working_dir="output/constraint_generation_runs",
    max_iterations=5,
):

    if output_json is not None:
        assert not os.path.exists(output_json), f"output file {output_json} already exists"

    num_constraints_added = 0
    best_total_scheduled = 0
    best_iteration = 0
    iteration = 0

    os.makedirs(working_dir, exist_ok=True)
    run_files = []
    if temp_json is None:
        temp_json = os.path.join(working_dir, f"temp_json_{uuid()}.json")
    while iteration < max_iterations:
        print(f"\nIteration {iteration}")
        pydcop_results_json = os.path.join(working_dir, f"it_{iteration}_temp_{uuid()}.json")

        # Run DCOP
        print("Running global dispatch")
        utils.run_global_dispatcher(
            pydcop_dict, algorithm_config, temp_json, pydcop_results_json, pydcop_mode=pydcop_mode, timeout=timeout
        )
        print("Done global dispatch")
        run_files.append(pydcop_results_json)
        # Load assignments
        assignments = utils.load_assignments(pydcop_results_json)

        # Group assigned requests by agent
        agent_assigned_reqs = defaultdict(set)
        for var_name, value in assignments.items():
            if int(value) == 1 and var_name in var_to_details:
                details = var_to_details[var_name]
                agent_assigned_reqs[details["agent_id"]].add(details["req_id"])
        # Execute Local Solvers
        global_scheduled_reqs = set()  # Global set to prevent double counting
        constraints = []
        for agent_id, assigned_reqs in agent_assigned_reqs.items():
            if not assigned_reqs:
                print(f"Agent {agent_id}: 0 requests assigned by DCOP.")
                continue

            print(f"Agent {agent_id} received {len(assigned_reqs)} requests. Computing exact schedule...")

            # Now returns the set of request IDs instead of just an integer
            scheduled_reqs, final_schedule = solve_local_schedule(
                agent_id,
                assigned_reqs,
                agent_tasks[agent_id],
                agent_downlinks[agent_id],
                agent_capacities[agent_id],
            )

            # Merge this agent's scheduled requests into the global set
            global_scheduled_reqs.update(scheduled_reqs)
            print(f"  -> Successfully scheduled: {len(scheduled_reqs)} requests (Tasks chosen: {final_schedule})")
            additional_constraints = get_constraints(
                assigned_reqs=assigned_reqs,
                scheduled_reqs=scheduled_reqs,
                agent_id=agent_id,
                pydcop_dict=pydcop_dict,
            )
            constraints.extend(additional_constraints)

        # The true length of the global set is our final, accurate count
        true_total_scheduled = len(global_scheduled_reqs)
        if true_total_scheduled > best_total_scheduled:
            best_total_scheduled = true_total_scheduled
            best_iteration = iteration

        print(f"Total requests successfully scheduled across network: {true_total_scheduled} of {len(requests)}")

        pydcop_dict, new_num_constraints_added = add_constraints(
            constraints=constraints,
            counter=num_constraints_added,
            pydcop_dict=pydcop_dict,
        )

        iteration += 1
        if num_constraints_added == new_num_constraints_added:
            # no new constraints added means our schedule is feasible
            # break here
            break
        num_constraints_added = new_num_constraints_added

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
    # return ratio satisifed
    return best_total_scheduled * 1.0 / len(requests), best_iteration, run_files
