import logging
import numpy as np

from pydcop.algorithms import AlgoParameterDef, ComputationDef
from pydcop.infrastructure.computations import Message, VariableComputation, register
from pydcop.computations_graph.constraints_hypergraph import VariableComputationNode
from pydcop.dcop.relations import (
    find_costs,
    optimal_cost_value,
)

HEADER_SIZE = 0
UNIT_SIZE = 1

# Type of computations graph that must be used with dsa
GRAPH_TYPE = "constraints_hypergraph"

algo_params = [
    AlgoParameterDef("regularization", "str", ["mwu", "none"], "mwu"),
    AlgoParameterDef('eta_reg', 'float', None, 1.0),
    AlgoParameterDef("predictive", "int", [0, 1], 0),
    AlgoParameterDef("context_based", "int", [0, 1], 0),
    AlgoParameterDef("stop_cycle", "int", None, 0),
    AlgoParameterDef("update_prob", "float", None, 1.0),
]


def computation_memory(computation: VariableComputationNode) -> float:
    return UNIT_SIZE*len(computation.variable.domain)*2


def communication_load(src: VariableComputationNode, target: str) -> float:
    return UNIT_SIZE + HEADER_SIZE


class FTRLMessage(Message):
    def __init__(self, value):
        super().__init__("ftrl_value", None)
        self._value = value

    @property
    def value(self):
        return self._value

    @property
    def size(self):
        return 1

    def __str__(self):
        return "FTRLMessage({})".format(self.value)

    def __repr__(self):
        return "FTRLMessage({})".format(self.value)

    def __eq__(self, other):
        if type(other) != FTRLMessage:
            return False
        if self.value == other.value:
            return True
        return False


class FTRLComputation(VariableComputation):

    def __init__(self, comp_def: ComputationDef):
        super().__init__(comp_def.node.variable, comp_def)

        self.utilities = None
        self.last_strategy = None
        self.ordered_domain = None

        assert comp_def.algo.algo == "ftrl"

        assert (comp_def.algo.mode == "min") or (comp_def.algo.mode == "max")

        self.mode = comp_def.algo.mode

        self.regularization = comp_def.algo.param_value("regularization")
        self.eta_reg = comp_def.algo.param_value("eta_reg")
        self.use_predictive = bool(comp_def.algo.param_value("predictive"))
        self.context_based = bool(comp_def.algo.param_value("context_based"))
        self.stop_cycle = comp_def.algo.param_value("stop_cycle")
        self.update_prob = float(comp_def.algo.param_value("update_prob"))
        self.constraints = comp_def.node.constraints

        # Maps for the values of our neighbors for the current and next cycle:
        self.current_cycle = {}
        self.next_cycle = {}

    def get_initial_utilities(self):
        return np.zeros(len(self.ordered_domain))

    def get_uniform_policy(self):
        return np.ones(len(self.ordered_domain))/len(self.ordered_domain)

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
            self.ordered_domain = tuple(self.variable.domain)
            if self.context_based:
                self.utilities = dict()
            else:
                self.utilities = self.get_initial_utilities()
            self.last_strategy = self.get_uniform_policy()
            self.random_value_selection()
            self.logger.debug(
                "FTRL starts: randomly select value %s", self.current_value
            )
            self.post_to_all_neighbors(FTRLMessage(self.current_value))

            # As everything is asynchronous, we might have received our
            # neighbors values even before starting this algorithm.
            self.evaluate_cycle()

    @register("ftrl_value")
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
            costs = np.array([costs[k] for k in self.ordered_domain])
            if self.mode == 'min':
                utilities = -costs
            else:
                utilities = costs
            # update utilities
            if self.context_based:
                context = tuple(self.current_cycle[n] for n in self.neighbors)
                if context not in self.utilities:
                    self.utilities[context] = self.get_initial_utilities()
                self.utilities[context] += utilities
                cum_utilities = self.utilities[context]
            else:
                self.utilities += utilities

                cum_utilities = self.utilities

            # get probability distribution
            if self.use_predictive:
                prediction = utilities
                basis = cum_utilities + prediction
            else:
                basis = cum_utilities
            if self.regularization == "mwu":
                # subtract max to reduce precision error between high values
                # very negative values may be pushed towards zero, but these were very low probability anyway
                dist = np.exp(self.eta_reg*(basis - np.max(basis)))
                dist = dist/np.sum(dist)
            elif self.regularization == 'none':
                support = np.equal(basis, np.max(basis))
                dist = support/np.sum(support)
            else:
                raise ValueError(self.regularization)

            self.assign_sampled_value(dist, costs)

            self.new_cycle()
            self.current_cycle, self.next_cycle = self.next_cycle, {}

            # Check if this was the last cycle
            if self.stop_cycle and self.cycle_count >= self.stop_cycle:
                self.finished()
                self.stop()
                return

            self.post_to_all_neighbors(FTRLMessage(self.current_value))

    def assign_sampled_value(self, strategy, costs):
        if np.random.random() < self.update_prob:
            idx = np.random.choice(np.arange(len(self.ordered_domain)), p=strategy)
        else:
            idx = self.ordered_domain.index(self.current_value)
        self.value_selection(self.ordered_domain[idx], costs[idx])
        return costs[idx]
