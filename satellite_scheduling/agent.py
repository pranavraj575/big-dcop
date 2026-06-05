from typing import List


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
