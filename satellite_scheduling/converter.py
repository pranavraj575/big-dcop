import json
import yaml
from collections import defaultdict
import time

"""
DEPRECATED
"""


def convert_scenario_bin_to_pydcop(json_filepath, yaml_filepath):
    print(f"Loading data from {json_filepath}...")
    start_time = time.time()

    with open(json_filepath, "r") as f:
        data = json.load(f)

    pydcop = {
        "name": "Satellite_Campaign_Task_Limit",
        "objective": "max",
        "domains": {"binary_domain": {"values": [0, 1]}},
        "variables": {},
        "agents": {},
        "constraints": {},
        "distribution": {},  # <--- Explicit Distribution Mapping
    }

    req_to_vars = defaultdict(list)
    agent_to_vars = defaultdict(list)
    var_details = {}

    HARD_PENALTY = -1000000
    MAX_TASKS_PER_WINDOW = 5

    agents_data = data.get("agents", [])
    print(f"Processing {len(agents_data)} agents...")

    for agent_data in agents_data:
        agent_id = str(agent_data.get("agent_id", agent_data.get("identifier")))
        pydcop["agents"][agent_id] = {}
        pydcop["distribution"][agent_id] = []  # Initialize this agent's task list

        # Parse Downlinks
        downlinks = []
        for dl in agent_data.get("downlinks", []):
            downlinks.append({"start": float(dl["start"]), "end": float(dl["end"])})
        downlinks = sorted(downlinks, key=lambda d: d["start"])

        # Parse Variables (Fulfillments)
        fulfillments = agent_data.get("fulfillments", [])
        fulfillments = sorted(fulfillments, key=lambda f: float(f["start_time"]))

        for rf in fulfillments:
            f_id = str(rf["fulfillment_id"])
            var_name = f"f_{f_id}"
            req_id = str(rf["request_id"])

            pydcop["variables"][var_name] = {"domain": "binary_domain"}
            req_to_vars[req_id].append(var_name)
            agent_to_vars[agent_id].append(var_name)

            # Map ONLY the variable computation to the physical agent
            pydcop["distribution"][agent_id].append(var_name)

            var_details[var_name] = {
                "start": float(rf["start_time"]),
                "end": float(rf["end_time"]),
            }

        # ---------------------------------------------------------
        # Hard Constraints (Overlaps & Task Limits)
        # ---------------------------------------------------------
        agent_vars = agent_to_vars[agent_id]

        # Build Downlink Windows
        windows = []
        last_end = -float("inf")
        for dl in downlinks:
            windows.append({"start": last_end, "end": dl["start"], "vars": []})
            last_end = dl["end"]
        windows.append({"start": last_end, "end": float("inf"), "vars": []})

        for i, v1 in enumerate(agent_vars):
            f1 = var_details[v1]

            # A. Downlink Blockage
            for dl in downlinks:
                if f1["start"] < dl["end"] and dl["start"] < f1["end"]:
                    c_name = f"dl_block_{v1}"
                    pydcop["constraints"][c_name] = {
                        "type": "intention",
                        "variables": [v1],
                        "function": f"{HARD_PENALTY} if {v1} == 1 else 0",
                    }

            # Assign variable to a specific window
            for w in windows:
                if w["start"] <= f1["start"] < w["end"]:
                    w["vars"].append(v1)
                    break

            # B. Task Overlap
            for j in range(i + 1, len(agent_vars)):
                v2 = agent_vars[j]
                f2 = var_details[v2]
                if f1["start"] < f2["end"] and f2["start"] < f1["end"]:
                    c_name = f"overlap_{v1}_{v2}"
                    pydcop["constraints"][c_name] = {
                        "type": "intention",
                        "variables": [v1, v2],
                        "function": f"{HARD_PENALTY} if ({v1} == 1 and {v2} == 1) else 0",
                    }

        # C. Max Tasks Per Window
        for w_idx, w in enumerate(windows):
            window_vars = w["vars"]
            if len(window_vars) > MAX_TASKS_PER_WINDOW:
                c_name = f"task_limit_{agent_id}_w{w_idx}"
                sum_expr = " + ".join(window_vars)
                pydcop["constraints"][c_name] = {
                    "type": "intention",
                    "variables": window_vars,
                    "function": f"0 if ({sum_expr}) <= {MAX_TASKS_PER_WINDOW} else {HARD_PENALTY}",
                }

    # ---------------------------------------------------------
    # Global Objective (Soft Constraints)
    # ---------------------------------------------------------
    print(f"Building utility functions for {len(req_to_vars)} requests...")
    for req_id, var_list in req_to_vars.items():
        if not var_list:
            continue
        c_name = f"reward_req_{req_id}"
        sum_expr = " + ".join(var_list)
        pydcop["constraints"][c_name] = {
            "type": "intention",
            "variables": var_list,
            "function": f"1 if ({sum_expr}) > 0 else 0",
        }

    # ---------------------------------------------------------
    # YAML Serialization
    # ---------------------------------------------------------
    print("Writing PyDCOP configuration to disk...")
    with open(yaml_filepath, "w") as f:
        yaml.dump(pydcop, f, default_flow_style=False, sort_keys=False)

    total_time = time.time() - start_time
    print(f"Done in {total_time:.2f} seconds. Output: {yaml_filepath}")


if __name__ == "__main__":
    convert_scenario_bin_to_pydcop("test.json", "test.yaml")
