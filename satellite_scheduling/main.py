import json
import yaml
import subprocess
import time
from collections import defaultdict
from scheduler import solve_local_schedule, get_constraints, add_constraints
from utils import parse_json_to_dcop_and_overlaps, run_global_dispatcher


def main():
    raw_json = "satellite_scheduling/test_large.json"
    temp_yaml = "output/dcop_global.yaml"
    pydcop_results = "output/pydcop_results.json"

    # Parse JSON
    pydcop_dict, agent_tasks, agent_downlinks, agent_capacities, var_to_details, requests = parse_json_to_dcop_and_overlaps(
        raw_json)

    additional_constraint_counter = 0
    # Run DCOP
    run_global_dispatcher(pydcop_dict, temp_yaml, pydcop_results)

    # Load assignments
    print("Running Local Solvers")
    with open(pydcop_results, 'r') as f:
        results = json.load(f)

    assignments = results.get('assignment', {})

    # Group assigned requests by agent
    agent_assigned_reqs = defaultdict(set)
    for var_name, value in assignments.items():
        if value == 1 and var_name in var_to_details:
            details = var_to_details[var_name]
            agent_assigned_reqs[details['agent_id']].add(details['req_id'])
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
            agent_capacities[agent_id]
        )

        # Merge this agent's scheduled requests into the global set
        global_scheduled_reqs.update(scheduled_reqs)
        print(f"  -> Successfully scheduled: {len(scheduled_reqs)} requests (Tasks chosen: {final_schedule})")
        additional_constraints = get_constraints(assigned_reqs=assigned_reqs,
                                                 scheduled_reqs=scheduled_reqs,
                                                 agent_id=agent_id,
                                                 pydcop_dict=pydcop_dict,
                                                 )
        constraints.extend(additional_constraints)

    # The true length of the global set is our final, accurate count
    true_total_scheduled = len(global_scheduled_reqs)

    print(f"Total requests successfully scheduled across network: {true_total_scheduled} of {len(requests)}")
    pydcop_dict, additional_constraint_counter = add_constraints(
        constraints=constraints,
        counter=additional_constraint_counter,
        pydcop_dict=pydcop_dict,
    )


if __name__ == "__main__":
    main()
