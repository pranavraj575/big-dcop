from collections import defaultdict
from ortools.linear_solver import pywraplp
from utils import HARD_PENALTY


def solve_local_schedule(agent_id, assigned_reqs, all_tasks, downlinks, capacity):
    """
    OR-Tools Local Solver:
    Computes a conflict-free schedule including memory constraints
    """
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        return set(), []

    valid_tasks = [t for t in all_tasks if t["req_id"] in assigned_reqs]
    if not valid_tasks:
        return set(), []

    task_vars = {}
    req_to_task_vars = defaultdict(list)

    for t in valid_tasks:
        tid = t["task_id"]
        tv = solver.IntVar(0, 1, f"t_{tid}")
        task_vars[tid] = tv
        req_to_task_vars[t["req_id"]].append(tv)

    # Choose at most 1 task per assigned request
    for req_id, tvs in req_to_task_vars.items():
        solver.Add(sum(tvs) <= 1)

    # Standard Time Overlaps
    for i in range(len(valid_tasks)):
        for j in range(i + 1, len(valid_tasks)):
            t1 = valid_tasks[i]
            t2 = valid_tasks[j]
            if t1["start"] < t2["end"] and t2["start"] < t1["end"]:
                solver.Add(task_vars[t1["task_id"]] + task_vars[t2["task_id"]] <= 1)

    # Continuous Memory Tracking
    events = []
    for t in valid_tasks:
        events.append(
            {"time": t["end"], "type": "task", "data": t["data"], "id": t["task_id"]}
        )
    for dl in downlinks:
        events.append({"time": dl["end"], "type": "dl", "data": dl["data"], "id": None})

    events.sort(key=lambda x: x["time"])

    mem_vars = []
    for i in range(len(events)):
        mv = solver.NumVar(0, capacity, f"mem_{i}")
        mem_vars.append(mv)

    for i, ev in enumerate(events):
        prev_mem = mem_vars[i - 1] if i > 0 else 0
        if ev["type"] == "task":
            solver.Add(mem_vars[i] == prev_mem + task_vars[ev["id"]] * ev["data"])
        else:
            solver.Add(mem_vars[i] >= prev_mem - ev["data"])
            solver.Add(mem_vars[i] <= prev_mem)

            # Objective: Maximize the number of successfully scheduled requests
    objective = solver.Objective()
    for tv in task_vars.values():
        objective.SetCoefficient(tv, 1)
    objective.SetMaximization()

    solver.Solve()

    scheduled_tasks = []
    scheduled_reqs = set()  # Track unique Request IDs

    for t in valid_tasks:
        tid = t["task_id"]
        if task_vars[tid].solution_value() == 1:
            scheduled_tasks.append(tid)
            scheduled_reqs.add(t["req_id"])  # Add the req_id to our set

    # Returns the set of unique requests and the list of tasks
    return scheduled_reqs, scheduled_tasks


def get_constraints(assigned_reqs, scheduled_reqs, agent_id, pydcop_dict):
    """
    gets constraints to add to pydcop instance based on assigned tasks and scheduled tasks
    currently does the following:
        if assigned tasks != scheduled tasks,
            add a constraint for every singleton that can be added to scheduled tasks
            i.e. scheduled tasks U {additional_task} is not allowed for any additional task
    returns list of (list of variables)
    """

    def agent_of(var_name):
        assert var_name.count("_") == 2
        return var_name.split("_")[1]

    constraints = []
    if len(scheduled_reqs) < len(assigned_reqs):
        # constraints.append(assigned_reqs)
        diff = set(assigned_reqs).difference(set(scheduled_reqs))
        for item in diff:
            constraints.append(assigned_reqs.union({item}))
    out = []
    for c in constraints:
        variables = []
        for req_id in c:
            relevant_vars = [
                v
                for v in pydcop_dict["constraints"][f"reward_req_{req_id}"]["variables"]
                if agent_of(v) == agent_id
            ]
            assert len(relevant_vars) == 1
            variables += relevant_vars
        assert len(set(variables)) == len(c)
        out.append(variables)
    return out


def add_constraints(constraints, counter, pydcop_dict):
    """
    adds additional constraints to pydcop dict
    names are "additional_constraint_i" where i starts at counter
    returns (dict, new counter)
        (i.e. additional_constraint_j will be unused where j is the returnec counter)
    """
    for var_list in constraints:
        sum_expr = " + ".join(var_list)
        pydcop_dict["constraints"][f"additional_constraint_{counter}"] = {
            "type": "intention",
            "variables": var_list,
            "function": f"{HARD_PENALTY} if ({sum_expr}) == {len(var_list)} else 0",
        }
        counter += 1
    return pydcop_dict, counter
