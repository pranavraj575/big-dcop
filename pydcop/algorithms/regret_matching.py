import logging
import numpy as np

from pydcop.algorithms import AlgoParameterDef, ComputationDef
from pydcop.infrastructure.computations import Message, VariableComputation, register
from pydcop.computations_graph.constraints_hypergraph import VariableComputationNode
from pydcop.dcop.relations import (
    find_costs,
    filter_assignment_dict,
    optimal_cost_value,
)

HEADER_SIZE = 0
UNIT_SIZE = 1

# Type of computations graph that must be used with dsa
GRAPH_TYPE = "constraints_hypergraph"

algo_params = [
    AlgoParameterDef("rm_plus", "int", [0, 1], 0),
    AlgoParameterDef("predictive", "int", [0, 1], 0),
    AlgoParameterDef("context_based", "int", [0, 1], 0),
    AlgoParameterDef("stop_cycle", "int", None, 0),
    AlgoParameterDef("update_prob", "float", None, 1.0),
]


def computation_memory(computation: VariableComputationNode) -> float:
    return UNIT_SIZE*len(computation.variable.domain)*2


def communication_load(src: VariableComputationNode, target: str) -> float:
    return UNIT_SIZE + HEADER_SIZE


class RMMessage(Message):
    def __init__(self, value):
        super().__init__("rm_value", None)
        self._value = value

    @property
    def value(self):
        return self._value

    @property
    def size(self):
        return 1

    def __str__(self):
        return "RMMessage({})".format(self.value)

    def __repr__(self):
        return "RMMessage({})".format(self.value)

    def __eq__(self, other):
        if type(other) != RMMessage:
            return False
        if self.value == other.value:
            return True
        return False


class RMComputation(VariableComputation):

    def __init__(self, comp_def: ComputationDef):
        super().__init__(comp_def.node.variable, comp_def)

        self.regrets = None
        self.last_strategy = None
        self.ordered_neighbors = None
        # technically, the order of neighbors in self.neighbors might change time
        # fix an order on initialization so we do not permute these accidentally

        assert comp_def.algo.algo == "regret_matching"

        assert (comp_def.algo.mode == "min") or (comp_def.algo.mode == "max")

        self.mode = comp_def.algo.mode

        self.use_rm_plus = bool(comp_def.algo.param_value("rm_plus"))
        self.use_predictive = bool(comp_def.algo.param_value("predictive"))
        self.context_based = bool(comp_def.algo.param_value("context_based"))
        self.stop_cycle = comp_def.algo.param_value("stop_cycle")
        self.update_prob = float(comp_def.algo.param_value("update_prob"))
        self.constraints = comp_def.node.constraints

        # Maps for the values of our neighbors for the current and next cycle:
        self.current_cycle = {}
        self.next_cycle = {}

    def get_initial_regrets(self):
        return {val: 0 for val in self.variable.domain}

    def get_uniform_policy(self):
        return {val: 1/len(self.variable.domain) for val in self.variable.domain}

    def on_start(self):
        if not self.neighbors:
            # If a variable has no neighbors, we must select its final value immediately
            # as it will never receive any message.
            value, cost = optimal_cost_value(self._variable, self.mode)
            self.value_selection(value, cost)
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"Select initial value {self.current_value} "
                    f"based on cost function for var {self._variable.name}"
                )
            self.finished()
            self.stop()
        else:
            if self.context_based:
                self.regrets = dict()
            else:
                self.regrets = self.get_initial_regrets()
            self.ordered_neighbors = tuple(self.neighbors)
            self.last_strategy = self.get_uniform_policy()
            self.random_value_selection()
            self.logger.debug(
                "RM starts: randomly select value %s", self.current_value
            )
            self.post_to_all_neighbors(RMMessage(self.current_value))

            # As everything is asynchronous, we might have received our
            # neighbors values even before starting this algorithm.
            self.evaluate_cycle()

    @register("rm_value")
    def _on_value_msg(self, variable_name, recv_msg, t):
        if not self._running:
            return
        if variable_name not in self.current_cycle:
            self.current_cycle[variable_name] = recv_msg.value
            self.logger.debug(
                "Receiving value %s from %s", recv_msg.value, variable_name
            )
            self.evaluate_cycle()

        else:
            self.logger.debug(
                "Receiving value %s from %s for the next cycle.",
                recv_msg.value,
                variable_name,
            )
            self.next_cycle[variable_name] = recv_msg.value

    def evaluate_cycle(self):
        if len(self.current_cycle) == len(self.neighbors):
            self.logger.debug(
                "Full neighbors assignment for cycle %s : %s ",
                self.cycle_count,
                self.current_cycle,
            )

            self.current_cycle[self.variable.name] = self.current_value
            assignment = self.current_cycle.copy()
            costs = find_costs(variable=self.variable, assignment=assignment, constraints=self.constraints)
            if self.mode == 'min':
                utilities = {k: -v for k, v in costs.items()}
            else:
                utilities = costs
            last_u = sum(utilities[k]*self.last_strategy[k] for k in utilities)

            # calc instantaneous regret
            instant_regrets = {
                k: u_action - last_u
                for k, u_action in utilities.items()
            }

            # update regrets, (if context based, update based on neighbors context)
            if self.context_based:
                context = tuple(self.current_cycle[n] for n in self.neighbors)
                if context not in self.regrets:
                    self.regrets[context] = self.get_initial_regrets()
                cum_regrets = self.regrets[context]
            else:
                cum_regrets = self.regrets
            if self.use_rm_plus:  # RM+
                for k in cum_regrets:
                    cum_regrets[k] = max(0, cum_regrets[k] + instant_regrets[k])
            else:
                for k in cum_regrets:
                    cum_regrets[k] += instant_regrets[k]

            # get probability distribution, (if predictive RM, use prediction of the most recent utilities obtained)
            if self.use_predictive:  # predictive RM
                regret_basis = {
                    k: cum_regrets[k] + instant_regrets[k]
                    for k in cum_regrets
                }
            else:  # default vanilla RM
                regret_basis = cum_regrets
            positive_part = sum(max(0, rT) for _, rT in regret_basis.items())
            if positive_part <= 0:
                self.last_strategy = self.get_uniform_policy()
            else:
                self.last_strategy = {k: max(0, rT)/positive_part for k, rT in cum_regrets.items()}

            self.assign_sampled_value(self.last_strategy, costs)

            self.new_cycle()
            self.current_cycle, self.next_cycle = self.next_cycle, {}

            # Check if this was the last cycle
            if self.stop_cycle and self.cycle_count >= self.stop_cycle:
                self.finished()
                self.stop()
                return

            self.post_to_all_neighbors(RMMessage(self.current_value))

    def assign_sampled_value(self, strategy, costs):
        if np.random.random() < self.update_prob:
            dom = list(self.variable.domain)
            value = np.random.choice(dom, p=[strategy[val] for val in dom])
        else:
            value = self.current_value
        self.value_selection(value, costs[value])
        return costs[value]

    def exists_violated_constraint(self) -> bool:
        """
        Tells if there is a violated soft constraint regarding the current
        assignment
        :return: a boolean
        """
        for c in self.constraints:
            asgt = self.current_cycle.copy()
            asgt[self.name] = self.current_value
            const = c(**filter_assignment_dict(asgt, c.dimensions))
            if const != self.best_constraints_costs[c.name]:
                return True
        return False
