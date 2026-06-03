from abc import ABC, abstractmethod
from agent import Agent
from typing import List


class COSPSolver(ABC):
    def __init__(self, config: dict, pydcop_dict: dict):
        self.config = config
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

    def solve(
        self,
    ):
        pass


if __name__ == "__main__":
    import json

    with open("scenarios/scenario_0.json", "r") as f:
        data = json.load(f)
    print(data)
    c = COSPSolver()
