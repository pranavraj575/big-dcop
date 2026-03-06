import json
import yaml
import subprocess
import time
from collections import defaultdict
from ortools.linear_solver import pywraplp

def solve_local_schedule(agent_id, assigned_reqs, all_tasks, downlinks, capacity):
    """
    OR-Tools Local Solver:
    Computes a conflict-free schedule including memory constraints
    """
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        return set(), []

    valid_tasks = [t for t in all_tasks if t['req_id'] in assigned_reqs]
    if not valid_tasks:
        return set(), []

    task_vars = {}
    req_to_task_vars = defaultdict(list)
    
    for t in valid_tasks:
        tid = t['task_id']
        tv = solver.IntVar(0, 1, f"t_{tid}")
        task_vars[tid] = tv
        req_to_task_vars[t['req_id']].append(tv)

    # Choose at most 1 task per assigned request
    for req_id, tvs in req_to_task_vars.items():
        solver.Add(sum(tvs) <= 1)

    # Standard Time Overlaps
    for i in range(len(valid_tasks)):
        for j in range(i + 1, len(valid_tasks)):
            t1 = valid_tasks[i]
            t2 = valid_tasks[j]
            if t1['start'] < t2['end'] and t2['start'] < t1['end']:
                solver.Add(task_vars[t1['task_id']] + task_vars[t2['task_id']] <= 1)

    # Continuous Memory Tracking
    events = []
    for t in valid_tasks:
        events.append({'time': t['end'], 'type': 'task', 'data': t['data'], 'id': t['task_id']})
    for dl in downlinks:
        events.append({'time': dl['end'], 'type': 'dl', 'data': dl['data'], 'id': None})
        
    events.sort(key=lambda x: x['time'])

    mem_vars = []
    for i in range(len(events)):
        mv = solver.NumVar(0, capacity, f"mem_{i}")
        mem_vars.append(mv)

    for i, ev in enumerate(events):
        prev_mem = mem_vars[i-1] if i > 0 else 0
        if ev['type'] == 'task':
            solver.Add(mem_vars[i] == prev_mem + task_vars[ev['id']] * ev['data'])
        else:
            solver.Add(mem_vars[i] >= prev_mem - ev['data'])
            solver.Add(mem_vars[i] <= prev_mem) 

    # Objective: Maximize the number of successfully scheduled requests
    objective = solver.Objective()
    for tv in task_vars.values():
        objective.SetCoefficient(tv, 1)
    objective.SetMaximization()

    solver.Solve()

    scheduled_tasks = []
    scheduled_reqs = set() # Track unique Request IDs
    
    for t in valid_tasks:
        tid = t['task_id']
        if task_vars[tid].solution_value() == 1:
            scheduled_tasks.append(tid)
            scheduled_reqs.add(t['req_id']) # Add the req_id to our set

    # Returns the set of unique requests and the list of tasks
    return scheduled_reqs, scheduled_tasks