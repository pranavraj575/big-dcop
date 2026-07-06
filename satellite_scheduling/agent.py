from typing import List
from typing import Dict, Set
import random


class Agent:
    def __init__(self, config: dict):
        self.config = config
        self.utility_evaluator = None

    def update(self, variable_assignments: List[set], pydcop_dict: dict):
        """

        Parameters
        ----------
        variable_assignments: previous time step

        Returns
        -------
        set of binary variables that are true
        """
        return set()

    def tasks(self):
        """
        returns set of all currently assigned tasks
        Returns
        -------

        """
        return set()

    def set_utility_evaluator(self, evaluator):
        """Set the utility function evaluator for custom constraint functions."""
        self.utility_evaluator = evaluator

    def evaluate_constraint_utility(self, function_str, var_values):
        """
        Evaluate a constraint utility function using the evaluator.

        Parameters
        ----------
        function_str : str
            Expression string like "-penalty * var_name" to evaluate
        var_values : dict
            Dictionary of variable names to values for evaluation

        Returns
        -------
        float
            Evaluated utility value, or 0.0 if no evaluator or error occurs
        """
        if self.utility_evaluator is None:
            return 0.0
        return self.utility_evaluator.evaluate(function_str, var_values)


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
        agent_idx = int(self.agent_id.split("_")[-1]) if "_" in str(self.agent_id) else 0

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
            return 1 / (new_sum**2)

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
        agent_idx = int(self.agent_id.split("_")[-1]) if "_" in str(self.agent_id) else 0

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
                gain_1 += 1 / (new_sum_1**2)

            if new_sum_0 == 0:
                gain_0 += 0
            else:
                gain_0 += 1 / (new_sum_0**2)

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
        agent_idx = int(self.agent_id.split("_")[-1]) if "_" in str(self.agent_id) else 0

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
