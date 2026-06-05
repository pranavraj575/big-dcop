import json
import subprocess
from collections import defaultdict
import time

HARD_PENALTY = -1000000


def parse_json_to_dcop_and_overlaps(json_filepath):
    """
    Reads JSON, builds PyDCOP with Request-level variables,
    and extracts tasks and memory bounds for the local solver.
    """
    # start_time = time.time()

    with open(json_filepath, "r") as f:
        data = json.load(f)
    requests = data.get("requests", [])

    pydcop = {
        "name": "Satellite_Request_Allocation",
        "objective": "max",
        "domains": {"binary_domain": {"values": [0, 1]}},
        "variables": {},
        "agents": {},
        "constraints": {},
        "distribution": {},
    }

    agent_tasks = defaultdict(list)
    agent_downlinks = defaultdict(list)
    agent_capacities = {}

    req_to_agents = defaultdict(list)
    agent_to_reqs = defaultdict(set)

    agents_data = data.get("agents", [])

    # Map all capabilities and data limits
    for agent_data in agents_data:
        agent_id = str(agent_data.get("agent_id", agent_data.get("identifier")))
        capacity = float(agent_data.get("data_volume_MB", 1000000.0))
        agent_capacities[agent_id] = capacity

        pydcop["agents"][agent_id] = {}
        pydcop["distribution"][agent_id] = []

        # Store Downlinks for the local memory tracker
        for dl in agent_data.get("downlinks", []):
            agent_downlinks[agent_id].append(
                {
                    "start": float(dl["start"]),
                    "end": float(dl["end"]),
                    "data": float(dl["data_volume_MB"]),
                }
            )

        # Store actual Tasks (Fulfillments)
        for f in agent_data.get("fulfillments", []):
            req_id = str(f["request_id"])
            task_id = str(f["fulfillment_id"])

            agent_to_reqs[agent_id].add(req_id)
            if agent_id not in req_to_agents[req_id]:
                req_to_agents[req_id].append(agent_id)

            agent_tasks[agent_id].append(
                {
                    "task_id": task_id,
                    "req_id": req_id,
                    "start": float(f["start_time"]),
                    "end": float(f["end_time"]),
                    "data": float(f["data_volume_MB"]),
                }
            )

    var_to_details = {}
    req_to_vars = defaultdict(list)

    # Create one PyDCOP variable per (Agent, Request) combination
    for agent_id, reqs in agent_to_reqs.items():
        var_counter = 0
        for req_id in reqs:
            var_name = f"v_{agent_id}_{var_counter}"
            var_counter += 1

            pydcop["variables"][var_name] = {"domain": "binary_domain"}
            pydcop["distribution"][agent_id].append(var_name)
            req_to_vars[req_id].append(var_name)

            # Save mapping so we know what this variable actually means later
            var_to_details[var_name] = {"req_id": req_id, "agent_id": agent_id}

    # DCOP Constraints
    # Ensure exactly 1 agent takes the request. Penalty if overlap.
    for req_id, var_list in req_to_vars.items():
        c_name = f"reward_req_{req_id}"
        sum_expr = " + ".join(var_list)
        pydcop["constraints"][c_name] = {
            "type": "intention",
            "variables": var_list,
            # "function": f"1 if ({sum_expr}) == 1 else ({HARD_PENALTY} if ({sum_expr}) > 1 else 0)",
            # TODO: try different constraints here, want f(1)=1, nf(n)<1, and (n+1)f(n+1)<nf(n)
            #  currently, nf(n)=1/n, which seems too strong
            "function": f"0 if {sum_expr} == 0 else 1 / ({sum_expr} ** 2)",
        }

    return (
        pydcop,
        agent_tasks,
        agent_downlinks,
        agent_capacities,
        var_to_details,
        requests,
    )


def load_assignments(pydcop_results):

    with open(pydcop_results, "r") as f:
        results = json.load(f)

    assignments = results.get("assignment", {})
    return assignments


def run_global_dispatcher(pydcop_dict, algorithm_config, json_filepath, output_json, pydcop_mode, timeout=-1):
    """
    Writes the PyDCOP dict to a temporary YAML and executes the solver.
    Args:
        pydcop_mode: uses the --mode parameter for pydcop
            mode='thread' uses only a single core (which is lightweight but inefficient)
            mode='process' uses multiple cores, which is heavier, but good parallelism across cores
        timeout: # seconds to set timeout to (-1 for no timeout)
    """

    with open(json_filepath, "w") as f:
        json.dump(pydcop_dict, f)
    cmd = ["pydcop"]
    if timeout >= 0:
        cmd.extend(["--timeout", str(timeout)])
    cmd = cmd + ["--output", output_json, "solve", "--mode", pydcop_mode, "--algo", algorithm_config["name"]]

    # add algorithm parameters
    if "algo_params" in algorithm_config:
        for param in algorithm_config["algo_params"]:
            cmd.extend(["--algo_param", param])

    # add the distribution and filepaths at the end
    cmd.extend(
        [
            "--distribution",
            json_filepath,
            json_filepath,
        ]
    )
    try:
        ts = time.time()
        subprocess.run(cmd, check=True)  # capture_output=True, text=True)
        te = time.time()
        print(f"Run in {te - ts} seconds")

        print("PyDCOP finished successfully.")
    except subprocess.CalledProcessError:
        print("Error running PyDCOP. Check terminal output.")
        exit(1)


def translate_pydcop_to_cosp_config(algorithm_config: dict) -> dict:
    """
    Translate pydcop algorithm config to COSPSolver config format.
    
    Pydcop format:
    {
        "name": "mgm",
        "algo_params": ["param1:value1", "param2:value2"]
    }
    
    COSPSolver format:
    {
        "algorithm": "mgm",
        "param1": "value1",
        "param2": "value2"
    }
    """
    cosp_config = {
        "algorithm": algorithm_config.get("name", "mgm").lower()
    }
    
    if "algo_params" in algorithm_config:
        for param in algorithm_config["algo_params"]:
            if ":" in param:
                key, value = param.split(":", 1)
                try:
                    cosp_config[key] = int(value)
                except ValueError:
                    try:
                        cosp_config[key] = float(value)
                    except ValueError:
                        cosp_config[key] = value
    
    return cosp_config


def normalize_constraints(pydcop_dict: dict) -> dict:
    """
    Normalize constraint format to ensure they all support the "function" field.
    Preserves existing format but converts simple lists to dict format with variables field.
    
    Returns:
        dict: New pydcop_dict with normalized constraints (doesn't modify original)
    """
    normalized = pydcop_dict.copy()
    
    if "constraints" not in normalized:
        return normalized
    
    normalized_constraints = {}
    for constraint_name, constraint_spec in pydcop_dict["constraints"].items():
        if isinstance(constraint_spec, list):
            normalized_constraints[constraint_name] = {
                "variables": constraint_spec
            }
        else:
            normalized_constraints[constraint_name] = constraint_spec.copy()
    
    normalized["constraints"] = normalized_constraints
    return normalized


def convert_cosp_to_assignments(cosp_result: dict) -> dict:
    """
    Convert COSPSolver result format to pydcop result format.
    
    COSPSolver format:
    {
        "algorithm": "MGM",
        "solution": {"agent_id": [list_of_vars]},
        "iterations": 5,
        "converged": true
    }
    
    Pydcop format:
    {
        "assignment": {"var_name": 0/1, ...}
    }
    """
    assignment = {}
    
    if "solution" in cosp_result:
        for agent_id, var_list in cosp_result["solution"].items():
            for var_name in var_list:
                assignment[var_name] = 1
    
    return {
        "assignment": assignment,
        "run_info": {
            "algorithm": cosp_result.get("algorithm", "unknown"),
            "iterations": cosp_result.get("iterations", 0),
            "converged": cosp_result.get("converged", False)
        }
    }


def run_global_dispatcher_cosp(pydcop_dict, algorithm_config, json_filepath, 
                               output_json, pydcop_mode, timeout=-1):
    """
    COSPSolver wrapper that replaces or supplements pydcop run_global_dispatcher.
    Uses in-process solver instead of external CLI subprocess.
    
    Args:
        pydcop_dict: DCOP problem dictionary with constraints in pydcop format
        algorithm_config: Algorithm config with "name" and "algo_params" fields
        json_filepath: Path to write temp JSON (for compatibility with existing code)
        output_json: Path where output JSON should be written
        pydcop_mode: Ignored (for API compatibility with pydcop version)
        timeout: Ignored (for API compatibility with pydcop version)
    """
    from cosp_solver import build_cosp
    
    normalized_pydcop = normalize_constraints(pydcop_dict)
    cosp_config = translate_pydcop_to_cosp_config(algorithm_config)
    
    try:
        ts = time.time()
        
        solver = build_cosp(normalized_pydcop, cosp_config)
        cosp_result = solver.solve()
        
        te = time.time()
        print(f"COSPSolver completed in {te - ts:.2f} seconds")
        
        pydcop_result = convert_cosp_to_assignments(cosp_result)
        
        with open(output_json, "w") as f:
            json.dump(pydcop_result, f, indent=2)
        
        print("COSPSolver finished successfully.")
    except Exception as e:
        print(f"Error running COSPSolver: {e}")
        raise

