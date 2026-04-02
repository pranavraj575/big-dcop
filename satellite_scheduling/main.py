import json
from collections import defaultdict
from scheduler import solve_local_schedule, get_constraints, add_constraints
from constraint_generation import solve_constraint_generation
from utils import parse_json_to_dcop_and_overlaps
from iterative_pricing import solve_iterative_pricing

MAX_ITERATIONS = 4

"""
ITAI: something really weird happens where when running this script pydcop runs an entire dcop for a min. Not sure what this is. It happens before any imports or anything...
"""

def main(constraint_generation):

    raw_json = "satellite_scheduling/test_large.json"
    temp_json = "output/dcop_global.json"
    pydcop_results = "output/pydcop_results.json"
    with open("satellite_scheduling/algorithm_configs.json", "r") as f:
            algorithms = json.load(f)


    algorithm_config = algorithms[0]
    # Parse JSON
    (
        pydcop_dict,
        agent_tasks,
        agent_downlinks,
        agent_capacities,
        var_to_details,
        requests,
    ) = parse_json_to_dcop_and_overlaps(raw_json)
 
    best_total_scheduled = solve_iterative_pricing( 
        pydcop_dict,
        agent_tasks,
        agent_downlinks,
        agent_capacities,
        var_to_details,
        requests,
        algorithm_config,
        max_iterations = MAX_ITERATIONS
    )
    print(best_total_scheduled)


if __name__ == "__main__":
    main(constraint_generation=True)
