from abc import ABC, abstractmethod
from agent import Agent
from typing import List
from collections import defaultdict
import time


def build_cosp(pydcop_dict: dict, algorithm_config: dict):
    pass


def cosp_run_global_dispatcher(pydcop_dict: dict, algorithm_config):
    cosp_solver: COSPSolver = build_cosp(pydcop_dict, algorithm_config)
    t0 = time.time()
    solution = cosp_solver.solve()
    return solution, time.time() - t0


def cosp_parse_json_to_dcop_and_overlaps(json_filepath):
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


class COSPSolver(ABC):
    def __init__(self, algorithm_config: dict, pydcop_dict: dict):
        self.algorithm_config = algorithm_config
        self.pydcop_dict = pydcop_dict.copy()
        # TODO: create array of agents
        self.agents: List[Agent] = []
        self.variable_assignments: List[set] = []

    @abstractmethod
    def update(self, pydcop_dict: dict):
        for agent in self.agents:
            agent.update(variable_assignments=self.variable_assignments, pydcop_dict=pydcop_dict)
        for agent_id, agent in enumerate(self.agents):
            self.variable_assignments[agent_id] = agent.tasks()

    def solve(self):
        pass


if __name__ == "__main__":
    import json

    fp = "scenarios/scenario_0.json"
    with open(fp, "r") as f:
        data = json.load(f)
    print(data)
    pydcop_dict = cosp_parse_json_to_dcop_and_overlaps(fp)
    print(pydcop_dict)
    # c = COSPSolver(pydcop_dict=pydcop_dict, algorithm_config=dict())
