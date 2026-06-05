from abc import ABC, abstractmethod
from agent import Agent
from constraint_parser import ConstraintFunctionEvaluator
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import time
import random
import json
import logging

logger = logging.getLogger(__name__)


def build_cosp(pydcop_dict: dict, algorithm_config: dict):
    """Factory function to instantiate the appropriate COSPSolver subclass."""
    algorithm_name = algorithm_config.get("algorithm", "mgm").lower()
    
    if algorithm_name == "mgm":
        return MGMSolver(algorithm_config, pydcop_dict)
    elif algorithm_name == "dsa":
        return DSASolver(algorithm_config, pydcop_dict)
    elif algorithm_name == "maxsum":
        return MaxSumSolver(algorithm_config, pydcop_dict)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")


def cosp_run_global_dispatcher(pydcop_dict: dict, algorithm_config):
    """Run DCOP solver and return solution and timing."""
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
        # "objective": "max",
        # "domains": {"binary_domain": {"values": [0, 1]}},
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
    pydcop.pop("distribution")
    pydcop["variables"] = list(pydcop["variables"].keys())
    pydcop["agents"] = list(pydcop["agents"].keys())
    pydcop["constraints"] = {k: v["variables"] for k, v in pydcop["constraints"].items()}
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
        """
        Parameters
        ----------
        algorithm_config: specifies what algorithm to use for each agent
        pydcop_dict: dict
            {
                'name' -> name of instance
                'variables' -> list of variables, each variable is named v_{agent id}_{variable_id}
                'agents' -> list of agent ids
                'constraints -> dict of {
                    request id -> list of variables corresponding to this request (at most one per agent)
                }
            }
        """
        self.algorithm_config = algorithm_config
        self.pydcop_dict = pydcop_dict.copy()
        self.agents: List[Agent] = []
        self.variable_assignments: List[Set] = []
        self.utility_evaluator = ConstraintFunctionEvaluator()

    @abstractmethod
    def update(self, pydcop_dict: dict):
        for agent in self.agents:
            agent.update(variable_assignments=self.variable_assignments, pydcop_dict=pydcop_dict)
        for agent_id, agent in enumerate(self.agents):
            self.variable_assignments[agent_id] = agent.tasks()

    def _setup_agents(self):
        """Inject the utility evaluator into all agents."""
        for agent in self.agents:
            agent.set_utility_evaluator(self.utility_evaluator)

    @abstractmethod
    def solve(self):
        pass


class MGMAgent(Agent):
    """MGM (Maximum Gain Message) Agent implementation."""
    
    def __init__(self, agent_id: str, config: dict):
        super().__init__(config)
        self.agent_id = agent_id
        self.current_value = 0
        self.variable_gains: Dict[str, float] = {}
        self.neighbor_values: Dict[str, int] = {}
        self.assigned_variables: Set[str] = set()

    def update(self, variable_assignments: List[Set], pydcop_dict: dict):
        """
        Update agent's variable assignments using MGM logic.
        MGM tries to maximize individual gain at each step.
        """
        agent_idx = int(self.agent_id.split('_')[-1]) if '_' in str(self.agent_id) else 0
        
        if agent_idx < len(variable_assignments):
            self.assigned_variables = variable_assignments[agent_idx].copy()
        
        if pydcop_dict and "constraints" in pydcop_dict:
            for constraint_name, constraint_spec in pydcop_dict["constraints"].items():
                var_list = self._get_variable_list(constraint_spec)
                function_str = self._get_function_str(constraint_spec)
                
                agent_vars = [v for v in var_list if self.agent_id in v]
                for var in agent_vars:
                    gain_1 = self._calculate_gain(var, 1, var_list, function_str, pydcop_dict)
                    gain_0 = self._calculate_gain(var, 0, var_list, function_str, pydcop_dict)
                    
                    if gain_1 > gain_0:
                        self.assigned_variables.add(var)
                    else:
                        self.assigned_variables.discard(var)

    def _calculate_gain(self, variable: str, value: int, var_list: List[str], function_str: str, pydcop_dict: dict) -> float:
        """Calculate the gain for assigning a value to a variable."""
        return self._constraint_utility(var_list, variable, value, function_str)

    def _constraint_utility(self, var_list: List[str], variable: str, value: int, function_str: str = None) -> float:
        """Calculate utility from a constraint, using custom function if provided."""
        if function_str and self.utility_evaluator:
            var_values = {v: (1 if v in self.assigned_variables else 0) for v in var_list}
            var_values[variable] = value
            return self.evaluate_constraint_utility(function_str, var_values)
        
        current_sum = sum(1 for v in var_list if v in self.assigned_variables)
        
        if value == 1:
            new_sum = current_sum + 1
        else:
            new_sum = max(0, current_sum - 1)
        
        if new_sum == 0:
            return 0
        else:
            return 1 / (new_sum ** 2)

    def _get_variable_list(self, constraint_spec) -> List[str]:
        """Extract variable list from constraint spec (handles both formats)."""
        if isinstance(constraint_spec, list):
            return constraint_spec
        elif isinstance(constraint_spec, dict) and "variables" in constraint_spec:
            return constraint_spec["variables"]
        return []

    def _get_function_str(self, constraint_spec) -> str:
        """Extract function string from constraint spec if present."""
        if isinstance(constraint_spec, dict) and "function" in constraint_spec:
            return constraint_spec["function"]
        return None

    def tasks(self) -> Set:
        return self.assigned_variables.copy()


class DSAAgent(Agent):
    """DSA (Distributed Stochastic Algorithm) Agent implementation."""
    
    def __init__(self, agent_id: str, config: dict):
        super().__init__(config)
        self.agent_id = agent_id
        self.variant = config.get("variant", "B")
        self.probability = config.get("probability", 0.7)
        self.assigned_variables: Set[str] = set()

    def update(self, variable_assignments: List[Set], pydcop_dict: dict):
        """
        Update agent's variable assignments using DSA logic.
        DSA uses stochastic decisions based on local constraints.
        """
        agent_idx = int(self.agent_id.split('_')[-1]) if '_' in str(self.agent_id) else 0
        
        if agent_idx < len(variable_assignments):
            self.assigned_variables = variable_assignments[agent_idx].copy()
        
        if pydcop_dict and "constraints" in pydcop_dict:
            for constraint_name, constraint_spec in pydcop_dict["constraints"].items():
                var_list = self._get_variable_list(constraint_spec)
                function_str = self._get_function_str(constraint_spec)
                
                agent_vars = [v for v in var_list if self.agent_id in v]
                for var in agent_vars:
                    best_value = self._get_best_value(var, var_list, function_str, pydcop_dict)
                    
                    if random.random() < self.probability:
                        if best_value == 1:
                            self.assigned_variables.add(var)
                        else:
                            self.assigned_variables.discard(var)

    def _get_best_value(self, variable: str, var_list: List[str], function_str: str, pydcop_dict: dict) -> int:
        """Get the best value for a variable based on constraints."""
        gain_1 = 0
        gain_0 = 0
        
        if function_str and self.utility_evaluator:
            var_values_1 = {v: (1 if v in self.assigned_variables else 0) for v in var_list}
            var_values_1[variable] = 1
            var_values_0 = {v: (1 if v in self.assigned_variables else 0) for v in var_list}
            var_values_0[variable] = 0
            
            gain_1 = self.evaluate_constraint_utility(function_str, var_values_1)
            gain_0 = self.evaluate_constraint_utility(function_str, var_values_0)
        else:
            current_sum = sum(1 for v in var_list if v in self.assigned_variables)
            
            new_sum_1 = current_sum + 1
            new_sum_0 = max(0, current_sum - 1)
            
            if new_sum_1 == 0:
                gain_1 += 0
            else:
                gain_1 += 1 / (new_sum_1 ** 2)
            
            if new_sum_0 == 0:
                gain_0 += 0
            else:
                gain_0 += 1 / (new_sum_0 ** 2)
        
        return 1 if gain_1 >= gain_0 else 0

    def _get_variable_list(self, constraint_spec) -> List[str]:
        """Extract variable list from constraint spec (handles both formats)."""
        if isinstance(constraint_spec, list):
            return constraint_spec
        elif isinstance(constraint_spec, dict) and "variables" in constraint_spec:
            return constraint_spec["variables"]
        return []

    def _get_function_str(self, constraint_spec) -> str:
        """Extract function string from constraint spec if present."""
        if isinstance(constraint_spec, dict) and "function" in constraint_spec:
            return constraint_spec["function"]
        return None

    def tasks(self) -> Set:
        return self.assigned_variables.copy()


class MaxSumAgent(Agent):
    """MaxSum (Belief Propagation) Agent implementation."""
    
    def __init__(self, agent_id: str, config: dict):
        super().__init__(config)
        self.agent_id = agent_id
        self.damping = config.get("damping", 0.0)
        self.assigned_variables: Set[str] = set()
        self.message_history: Dict[str, float] = {}
        self.incoming_messages: Dict[str, float] = {}

    def update(self, variable_assignments: List[Set], pydcop_dict: dict):
        """
        Update agent's variable assignments using MaxSum logic.
        MaxSum uses belief propagation with damped messages.
        """
        agent_idx = int(self.agent_id.split('_')[-1]) if '_' in str(self.agent_id) else 0
        
        if agent_idx < len(variable_assignments):
            self.assigned_variables = variable_assignments[agent_idx].copy()
        
        if pydcop_dict and "constraints" in pydcop_dict:
            self._update_beliefs(pydcop_dict)

    def _update_beliefs(self, pydcop_dict: dict):
        """Update beliefs based on incoming messages from constraints."""
        for constraint_name, constraint_spec in pydcop_dict["constraints"].items():
            var_list = self._get_variable_list(constraint_spec)
            function_str = self._get_function_str(constraint_spec)
            
            agent_vars = [v for v in var_list if self.agent_id in v]
            
            for var in agent_vars:
                if function_str and self.utility_evaluator:
                    var_values_1 = {v: (1 if v in self.assigned_variables else 0) for v in var_list}
                    var_values_1[var] = 1
                    utility_1 = self.evaluate_constraint_utility(function_str, var_values_1)
                    
                    var_values_0 = {v: (1 if v in self.assigned_variables else 0) for v in var_list}
                    var_values_0[var] = 0
                    utility_0 = self.evaluate_constraint_utility(function_str, var_values_0)
                    
                    incoming_sum = utility_1 - utility_0
                else:
                    incoming_sum = sum(self.incoming_messages.get(v, 0) for v in var_list if v != var)
                
                current_message = self.message_history.get(var, 0)
                new_message = (1 - self.damping) * incoming_sum + self.damping * current_message
                
                self.message_history[var] = new_message
                
                if new_message > 0:
                    self.assigned_variables.add(var)
                else:
                    self.assigned_variables.discard(var)

    def _get_variable_list(self, constraint_spec) -> List[str]:
        """Extract variable list from constraint spec (handles both formats)."""
        if isinstance(constraint_spec, list):
            return constraint_spec
        elif isinstance(constraint_spec, dict) and "variables" in constraint_spec:
            return constraint_spec["variables"]
        return []

    def _get_function_str(self, constraint_spec) -> str:
        """Extract function string from constraint spec if present."""
        if isinstance(constraint_spec, dict) and "function" in constraint_spec:
            return constraint_spec["function"]
        return None

    def tasks(self) -> Set:
        return self.assigned_variables.copy()


class MGMSolver(COSPSolver):
    """MGM (Maximum Gain Message) DCOP Solver."""
    
    def __init__(self, algorithm_config: dict, pydcop_dict: dict):
        super().__init__(algorithm_config, pydcop_dict)
        self.max_iterations = algorithm_config.get("max_iterations", 100)
        self.convergence_threshold = algorithm_config.get("convergence_threshold", 0)
        self.break_mode = algorithm_config.get("break_mode", "first")
        self.stop_cycle = algorithm_config.get("stop_cycle", 0)
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize MGM agents."""
        for agent_id in self.pydcop_dict.get("agents", []):
            agent = MGMAgent(agent_id, self.algorithm_config)
            self.agents.append(agent)
            self.variable_assignments.append(set())

    def update(self, pydcop_dict: dict):
        """Coordinate MGM update across all agents."""
        super().update(pydcop_dict)

    def solve(self) -> Dict:
        """Run MGM algorithm iterations until convergence."""
        self._setup_agents()
        iteration = 0
        previous_assignments = [set() for _ in range(len(self.agents))]
        
        while iteration < self.max_iterations:
            if self.stop_cycle > 0 and iteration >= self.stop_cycle:
                break
            
            self.update(self.pydcop_dict)
            
            if self._has_converged(previous_assignments):
                logger.info(f"MGM converged at iteration {iteration}")
                break
            
            previous_assignments = [a.copy() for a in self.variable_assignments]
            iteration += 1
        
        return {
            "algorithm": "MGM",
            "iterations": iteration,
            "solution": self._extract_solution(),
            "converged": iteration < self.max_iterations
        }

    def _has_converged(self, previous_assignments: List[Set]) -> bool:
        """Check if algorithm has converged."""
        for i, current in enumerate(self.variable_assignments):
            if current != previous_assignments[i]:
                return False
        return True

    def _extract_solution(self) -> Dict:
        """Extract final solution as variable assignments."""
        solution = {}
        for agent_id, assignments in zip(self.pydcop_dict.get("agents", []), self.variable_assignments):
            solution[agent_id] = list(assignments)
        return solution


class DSASolver(COSPSolver):
    """DSA (Distributed Stochastic Algorithm) DCOP Solver."""
    
    def __init__(self, algorithm_config: dict, pydcop_dict: dict):
        super().__init__(algorithm_config, pydcop_dict)
        self.max_iterations = algorithm_config.get("max_iterations", 100)
        self.variant = algorithm_config.get("variant", "B")
        self.probability = algorithm_config.get("probability", 0.7)
        self.stop_cycle = algorithm_config.get("stop_cycle", 0)
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize DSA agents."""
        config = {
            "variant": self.variant,
            "probability": self.probability
        }
        for agent_id in self.pydcop_dict.get("agents", []):
            agent = DSAAgent(agent_id, config)
            self.agents.append(agent)
            self.variable_assignments.append(set())

    def update(self, pydcop_dict: dict):
        """Coordinate DSA update across all agents."""
        super().update(pydcop_dict)

    def solve(self) -> Dict:
        """Run DSA algorithm iterations until convergence or max_iterations."""
        self._setup_agents()
        iteration = 0
        
        while iteration < self.max_iterations:
            if self.stop_cycle > 0 and iteration >= self.stop_cycle:
                break
            
            self.update(self.pydcop_dict)
            iteration += 1
        
        return {
            "algorithm": "DSA",
            "variant": self.variant,
            "iterations": iteration,
            "solution": self._extract_solution(),
            "converged": False
        }

    def _extract_solution(self) -> Dict:
        """Extract final solution as variable assignments."""
        solution = {}
        for agent_id, assignments in zip(self.pydcop_dict.get("agents", []), self.variable_assignments):
            solution[agent_id] = list(assignments)
        return solution


class MaxSumSolver(COSPSolver):
    """MaxSum (Belief Propagation) DCOP Solver."""
    
    def __init__(self, algorithm_config: dict, pydcop_dict: dict):
        super().__init__(algorithm_config, pydcop_dict)
        self.max_iterations = algorithm_config.get("max_iterations", 100)
        self.damping = algorithm_config.get("damping", 0.0)
        self.stability = algorithm_config.get("stability", 1e-4)
        self.damping_nodes = algorithm_config.get("damping_nodes", "vars")
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize MaxSum agents."""
        config = {
            "damping": self.damping,
            "damping_nodes": self.damping_nodes
        }
        for agent_id in self.pydcop_dict.get("agents", []):
            agent = MaxSumAgent(agent_id, config)
            self.agents.append(agent)
            self.variable_assignments.append(set())

    def update(self, pydcop_dict: dict):
        """Coordinate MaxSum update across all agents."""
        super().update(pydcop_dict)

    def solve(self) -> Dict:
        """Run MaxSum algorithm iterations until convergence."""
        self._setup_agents()
        iteration = 0
        previous_messages = {}
        
        while iteration < self.max_iterations:
            self.update(self.pydcop_dict)
            
            current_messages = self._collect_messages()
            
            if self._has_converged(previous_messages, current_messages):
                logger.info(f"MaxSum converged at iteration {iteration}")
                break
            
            previous_messages = current_messages
            iteration += 1
        
        return {
            "algorithm": "MaxSum",
            "iterations": iteration,
            "solution": self._extract_solution(),
            "converged": iteration < self.max_iterations
        }

    def _collect_messages(self) -> Dict:
        """Collect current messages from all agents."""
        messages = {}
        for agent in self.agents:
            if isinstance(agent, MaxSumAgent):
                messages[agent.agent_id] = agent.message_history.copy()
        return messages

    def _has_converged(self, previous_messages: Dict, current_messages: Dict) -> bool:
        """Check if MaxSum has converged based on message stability."""
        if not previous_messages:
            return False
        
        for agent_id, current_msgs in current_messages.items():
            if agent_id not in previous_messages:
                return False
            
            previous_msgs = previous_messages[agent_id]
            for var, curr_msg in current_msgs.items():
                if var not in previous_msgs:
                    return False
                
                prev_msg = previous_msgs[var]
                if abs(curr_msg - prev_msg) > self.stability:
                    return False
        
        return True

    def _extract_solution(self) -> Dict:
        """Extract final solution as variable assignments."""
        solution = {}
        for agent_id, assignments in zip(self.pydcop_dict.get("agents", []), self.variable_assignments):
            solution[agent_id] = list(assignments)
        return solution


if __name__ == "__main__":
    import json

    fp = "scenarios/scenario_0.json"
    with open(fp, "r") as f:
        data = json.load(f)
    # print(data)
    (
        pydcop,
        agent_tasks,
        agent_downlinks,
        agent_capacities,
        var_to_details,
        requests,
    ) = cosp_parse_json_to_dcop_and_overlaps(fp)
    print(pydcop)
    print(pydcop.keys())
    # c = COSPSolver(pydcop_dict=pydcop_dict, algorithm_config=dict())
