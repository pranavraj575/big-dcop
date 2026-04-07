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
    AlgoParameterDef("rm_plus", "int", [0, 1], 0),
    AlgoParameterDef("predictive", "int", [0, 1], 0),
    AlgoParameterDef("ir_prm", "int", [0, 1], 0),
    AlgoParameterDef("context_based", "int", [0, 1], 0),
    AlgoParameterDef("stop_cycle", "int", None, 0),
    AlgoParameterDef("update_prob", "float", None, 1.0),
    AlgoParameterDef("discounted_rm", "int", [0, 1], 0),
    AlgoParameterDef("alpha", "float", None, 1.5),  # Discount for negative regrets
    AlgoParameterDef("beta", "float", None, 0),  # Discount for negative regrets
    AlgoParameterDef("damping", "float", None, 0.0),
]


def computation_memory(computation: VariableComputationNode) -> float:
    return UNIT_SIZE * len(computation.variable.domain) * 2


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
        if type(other) is not RMMessage:
            return False
        if self.value == other.value:
            return True
        return False


class RMComputation(VariableComputation):
    def __init__(self, comp_def: ComputationDef):
        super().__init__(comp_def.node.variable, comp_def)

        self.regrets = None
        self.ir_prm_prediction = None
        self.last_strategy = None
        self.ordered_domain = None
        self.domain_size = None
        self.ordered_neighbors = None
        # technically, the order of neighbors in self.neighbors might change time
        # fix an order on initialization so we do not permute these accidentally

        assert comp_def.algo.algo == "regret_matching"

        assert (comp_def.algo.mode == "min") or (comp_def.algo.mode == "max")

        self.mode = comp_def.algo.mode

        self.use_rm_plus = bool(comp_def.algo.param_value("rm_plus"))
        self.use_predictive = bool(comp_def.algo.param_value("predictive"))
        self.use_ir_prm = bool(comp_def.algo.param_value("ir_prm"))
        self.context_based = bool(comp_def.algo.param_value("context_based"))
        self.stop_cycle = comp_def.algo.param_value("stop_cycle")
        self.update_prob = float(comp_def.algo.param_value("update_prob"))
        self.alpha = float(comp_def.algo.param_value("alpha"))
        self.beta = float(comp_def.algo.param_value("beta"))
        self.use_discounted = bool(comp_def.algo.param_value("discounted_rm"))
        self.damping = float(comp_def.algo.param_value("damping"))
        self.constraints = comp_def.node.constraints

        # Maps for the values of our neighbors for the current and next cycle:
        self.current_cycle = {}
        self.next_cycle = {}

    def get_zero_vector(self):
        return np.zeros(self.domain_size)

    def get_uniform_policy(self):
        return np.ones(self.domain_size) / self.domain_size

    def on_start(self):
        if not self.neighbors:
            # If a variable has no neighbors, we must select its final value immediately
            # as it will never receive any message.
            value, cost = optimal_cost_value(self._variable, self.mode)
            self.value_selection(value, cost)
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(f"Select initial value {self.current_value} based on cost function for var {self._variable.name}")
            self.finished()
            self.stop()
        else:
            if self.context_based:
                self.regrets = dict()
            else:
                self.regrets = self.get_zero_vector()
            if self.use_ir_prm:
                self.ir_prm_prediction = self.get_zero_vector()
            self.ordered_neighbors = tuple(self.neighbors)
            self.ordered_domain = tuple(self.variable.domain)
            self.domain_size = len(self.ordered_domain)
            self.last_strategy = self.get_uniform_policy()
            self.random_value_selection()
            self.logger.debug("RM starts: randomly select value %s", self.current_value)
            self.post_to_all_neighbors(RMMessage(self.current_value))

            # As everything is asynchronous, we might have received our
            # neighbors values even before starting this algorithm.
            self.evaluate_cycle()

    @register("rm_value")
    def _on_value_msg(self, variable_name, recv_msg, t):
        if not self._running:
            return

        # In synchronous mode, we strictly separate current and next
        # If we get a message for a cycle we already processed, it's for the next one.
        if variable_name not in self.current_cycle:
            self.current_cycle[variable_name] = recv_msg.value
        else:
            self.next_cycle[variable_name] = recv_msg.value

        # The trigger: only evaluate when the buffer for the current cycle is full
        if len(self.current_cycle) == len(self.neighbors):
            self.evaluate_cycle()

    def evaluate_cycle(self):
        if len(self.current_cycle) == len(self.neighbors):
            self.logger.debug(
                "Full neighbors assignment for cycle %s : %s ",
                self.cycle_count,
                self.current_cycle,
            )

            self.current_cycle[self.variable.name] = self.current_value
            assignment = self.current_cycle.copy()
            costs = find_costs(
                variable=self.variable,
                assignment=assignment,
                constraints=self.constraints,
            )
            # convert to array
            costs_arr = np.array([costs[k] for k in self.ordered_domain])
            if self.mode == "min":
                utilities = -costs_arr
            else:
                utilities = costs_arr
            if self.use_ir_prm:
                new_strat = self.ir_prm_ioannis_version(utilities=utilities)
            else:
                new_strat = self.observe_utilities_get_next_strat(utilities=utilities)
            self.last_strategy = new_strat
            self.assign_sampled_value(self.last_strategy, costs)

            self.new_cycle()
            self.current_cycle = self.next_cycle
            self.next_cycle = {}

            self.post_to_all_neighbors(RMMessage(self.current_value))

            # Check if this was the last cycle
            if self.stop_cycle and self.cycle_count >= self.stop_cycle:
                self.finished()
                self.stop()
                return

            if len(self.current_cycle) == len(self.neighbors):
                self.evaluate_cycle()

    def observe_utilities_get_next_strat(self, utilities):
        # calc instantaneous regret
        instant_regrets = utilities - np.dot(utilities, self.last_strategy)

        # observe utility at time t
        pos_discount, neg_discount = self.get_positive_and_negative_discounts()
        # update regrets, (if context based, update based on neighbors context)
        cum_regrets = self.get_cum_regrets()
        if (pos_discount, neg_discount) != (1, 1):
            # multiply positive values by pos_discount
            # neg values by neg_discount
            discounted_old = np.where(
                cum_regrets > 0,
                cum_regrets * pos_discount,
                cum_regrets * neg_discount,
            )
        else:
            discounted_old = cum_regrets
        cum_regrets = discounted_old + instant_regrets
        if self.use_rm_plus:  # RM+
            cum_regrets = np.clip(cum_regrets, 0, np.inf)
        self.set_cum_regrets(cum_regrets)
        # next strategy (time t+1)
        # get probability distribution, (if predictive RM, use prediction of the most recent utilities obtained)
        if self.use_predictive:  # predictive RM
            regret_basis = cum_regrets + instant_regrets
        else:  # default vanilla RM
            regret_basis = cum_regrets

        positive_part = np.clip(regret_basis, 0, np.inf)
        pos_sum = np.sum(positive_part)

        if pos_sum <= 0:
            new_strat = self.get_uniform_policy()
        else:
            new_strat = positive_part / pos_sum

        if self.damping > 0:
            new_strat = self.damping * self.last_strategy + (1 - self.damping) * new_strat
        return new_strat

    def get_positive_and_negative_discounts(self):
        t = self.cycle_count + 1

        if self.use_discounted:
            pos_discount = (t**self.alpha) / (t**self.alpha + 1)
            neg_discount = (t**self.beta) / (t**self.beta + 1)
        else:
            pos_discount = 1
            neg_discount = 1

        return pos_discount, neg_discount

    def get_cum_regrets(self):
        if self.context_based:
            context = tuple(self.current_cycle[n] for n in self.neighbors)
            if context not in self.regrets:
                self.regrets[context] = self.get_zero_vector()
            return self.regrets[context]
        else:
            return self.regrets

    def set_cum_regrets(self, regrets):
        if self.context_based:
            context = tuple(self.current_cycle[n] for n in self.neighbors)
            self.regrets[context] = regrets
        else:
            self.regrets = regrets

    def assign_sampled_value(self, strategy, costs):
        if np.random.random() < self.update_prob:
            # CANNOT just do the following line, as it converts value to a np.int64 if it is an int
            #  this breaks json encoding
            # value = np.random.choice(self.ordered_domain, p=strategy)
            value = self.ordered_domain[np.random.choice(range(len(strategy)), p=strategy)]
        else:
            value = self.current_value
        self.value_selection(value, costs[value])
        return costs[value]

    def ir_prm_ioannis_version(self, utilities):
        """
        different enough to warrant new method
        """
        # ObserveUtility at timestep t
        u = utilities - self.ir_prm_prediction
        cum_regrets = self.get_cum_regrets() + u - np.dot(u, self.last_strategy)
        if self.use_rm_plus:  # RM+
            cum_regrets = np.clip(cum_regrets, 0, np.inf)
        # NextStrategy at timestep t+1
        if np.all(cum_regrets <= 0):
            new_strat = self.last_strategy
        else:
            self.ir_prm_prediction = utilities
            old_norm = np.linalg.norm(np.clip(cum_regrets, 0, np.inf))
            cum_regrets = cum_regrets + utilities
            gamma = self.ir_prm_get_gamma_slow(cum_regrets, old_norm)
            cum_regrets = cum_regrets - gamma
            positive_part = np.clip(cum_regrets, 0, np.inf)
            new_strat = positive_part / np.sum(positive_part)
        self.set_cum_regrets(cum_regrets)
        return new_strat

    def ir_prm_observe_utilities_get_next_strat(self, utilities):
        """
        different enough to warrant new method
        """
        # ObserveUtility at timestep t
        g = (utilities - self.ir_prm_prediction) - np.dot(utilities - self.ir_prm_prediction, self.last_strategy)
        cum_regrets = self.get_cum_regrets()
        r_hat = cum_regrets + g
        if self.use_rm_plus:  # RM+
            r_hat = np.clip(r_hat, 0, np.inf)
        # ignore calculating x_hat for now

        # the prediction of utilities for the t+1 step is the previous utility vector
        #  a bit awkward here since we are changing timesteps mid cycle
        self.ir_prm_prediction = utilities

        # NextStrategy at timestep t+1
        if np.all(r_hat <= 0):
            # TODO: is this supposed to update regrets?
            self.ir_prm_prediction = self.get_zero_vector()

            # TODO: setting new_strat=x_hat does not make sense here,
            #  as x_hat is not necessarily a prob distribution
            #  (it only works for rm+), changed to a different defintioin
            new_strat = self.last_strategy
        else:
            l2_norm = np.linalg.norm(r_hat)
            gamma = self.ir_prm_get_gamma_slow(
                v=r_hat + self.ir_prm_prediction,
                t=l2_norm,
            )
            cum_regrets = r_hat + self.ir_prm_prediction - gamma
            self.set_cum_regrets(cum_regrets)
            positive_part = np.clip(cum_regrets, 0, np.inf)
            new_strat = positive_part / np.sum(positive_part)
        return new_strat

    def ir_prm_get_gamma_slow(self, v, t):
        t = t**2
        # v sorted in descending order
        v = -np.sort(-v)
        s = 0
        s2 = 0
        for k, v_k in enumerate(v):
            # k is zero indexed, using (k+1) when this is relevant
            s += v_k
            s2 += v_k**2
            gamma = (s - np.sqrt(s**2 - (k + 1) * (s2 - t))) / (k + 1)
            if (k + 1) >= len(v) or gamma >= v[k + 1]:
                return gamma
        assert False
