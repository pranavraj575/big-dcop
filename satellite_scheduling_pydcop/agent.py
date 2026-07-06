from typing import List


class Agent:
    def __init__(self, config: dict):
        self.config = config

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
