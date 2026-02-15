# pydcop/algorithms/maxsum_advp.py
# BSD-3-Clause License
#
# Copyright 2017 Orange
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
MaxSum ADVP: Max-Sum with Advanced Value Propagation (Custom Variant)
---------------------------------------------------------------------

This is a **synchronous implementation** of Max-Sum.
It includes normalization of costs to prevent numerical instability.

Algorithm Parameters
^^^^^^^^^^^^^^^^^^^^

**damping**
  amount of damping [0-1] (Default: 0.5)

**damping_nodes**
  nodes that apply damping to messages: "vars", "factors", "both" or "none"

**stability**
  stability detection coefficient

**noise**
  noise level for variable

**start_messages**
  nodes that initiate messages : "leafs", "leafs_vars", "all"

**stop_cycle**
  Cycle at which to force stop (if not converged).

"""

import logging
from typing import Optional, List, Dict, Any, Tuple, Union
from collections import defaultdict
from operator import itemgetter

from pydcop.algorithms import ComputationDef, AlgoParameterDef
from pydcop.computations_graph.factor_graph import (
    FactorComputationNode,
    VariableComputationNode,
)
from pydcop.dcop.objects import Variable, VariableNoisyCostFunc
from pydcop.dcop.relations import Constraint, generate_assignment_as_dict
from pydcop.infrastructure.computations import (
    DcopComputation,
    SynchronousComputationMixin,
    VariableComputation,
    register,
    Message,
)

# --- CRITICAL: Define Graph Type ---
GRAPH_TYPE = "factor_graph"
logger = logging.getLogger("pydcop.maxsum_advp")

SAME_COUNT = 4
STABILITY_COEFF = 0.1
HEADER_SIZE = 0
UNIT_SIZE = 1

# constants for memory costs and capacity
FACTOR_UNIT_SIZE = 1
VARIABLE_UNIT_SIZE = 1


def build_computation(comp_def: ComputationDef):
    if comp_def.node.type == "VariableComputation":
        return MaxSumADVPVariableComputation(comp_def=comp_def)
    if comp_def.node.type == "FactorComputation":
        return MaxSumADVPFactorComputation(comp_def=comp_def)


def computation_memory(
    computation: Union[FactorComputationNode, VariableComputationNode]
) -> float:
    """Memory footprint associated with the maxsum computation node."""
    if isinstance(computation, FactorComputationNode):
        m = 0
        for v in computation.variables:
            domain_size = len(v.domain)
            m += domain_size * FACTOR_UNIT_SIZE
        return m

    elif isinstance(computation, VariableComputationNode):
        domain_size = len(computation.variable.domain)
        # Handle cases where neighbors might not be populated in ad-hoc distributions
        num_neighbors = len(list(computation.links)) if hasattr(computation, 'links') else 0
        return num_neighbors * domain_size * VARIABLE_UNIT_SIZE
    
    # Fallback for simple heuristics
    return 100.0


def communication_load(
    src: Union[FactorComputationNode, VariableComputationNode], target: str
) -> float:
    """The communication cost of an edge between a variable and a factor."""
    if isinstance(src, VariableComputationNode):
        d_size = len(src.variable.domain)
        return UNIT_SIZE * d_size + HEADER_SIZE

    elif isinstance(src, FactorComputationNode):
        for v in src.variables:
            if v.name == target:
                d_size = len(v.domain)
                return UNIT_SIZE * d_size + HEADER_SIZE
        # Fallback if target not found immediately
        return UNIT_SIZE * 10 + HEADER_SIZE

    return UNIT_SIZE


algo_params = [
    AlgoParameterDef("damping", "float", None, 0.5),
    AlgoParameterDef(
        "damping_nodes", "str", ["vars", "factors", "both", "none"], "both"
    ),
    AlgoParameterDef("stability", "float", None, STABILITY_COEFF),
    AlgoParameterDef("noise", "float", None, 0.01),
    AlgoParameterDef("start_messages", "str", ["leafs", "leafs_vars", "all"], "leafs"),
    AlgoParameterDef("stop_cycle", "int", None, 100),
]


class MaxSumADVPMessage(Message):
    def __init__(self, costs: Dict):
        # Register as "maxsum_advp" to avoid conflict with standard maxsum
        super().__init__("maxsum_advp", None)
        self._costs = costs

    @property
    def costs(self):
        return self._costs

    @property
    def size(self):
        return len(self._costs) * 2

    def __str__(self):
        return "MaxSumADVPMessage({})".format(self._costs)

    def __repr__(self):
        return "MaxSumADVPMessage({})".format(self._costs)

    def __eq__(self, other):
        if type(other) != MaxSumADVPMessage:
            return False
        return self.costs == other.costs


# Type definitions
VarName = str
FactorName = str
VarVal = Any
Cost = float


class MaxSumADVPFactorComputation(SynchronousComputationMixin, DcopComputation):
    def __init__(self, comp_def: ComputationDef):
        # Allow "maxsum_advp" or "maxsum" to support running with generic config
        assert comp_def.algo.algo in ["maxsum", "maxsum_advp"]
        super().__init__(comp_def.node.factor.name, comp_def)
        
        self.mode = comp_def.algo.mode
        self.factor = comp_def.node.factor
        self.variables = self.factor.dimensions

        self._costs: Dict[VarName, Dict[VarVal, Cost]] = {}

        self.damping = comp_def.algo.params.get("damping", 0.5)
        self.damping_nodes = comp_def.algo.params.get("damping_nodes", "both")
        self.stability_coef = comp_def.algo.params.get("stability", STABILITY_COEFF)
        self.start_messages = comp_def.algo.params.get("start_messages", "leafs")
        
        self._prev_messages = defaultdict(lambda: (None, 0))

    def on_start(self):
        # Start message logic
        should_send = False
        if len(self.variables) == 1 and self.start_messages in ["leafs", "leafs_vars"]:
            should_send = True
        elif self.start_messages == "all":
            should_send = True

        if should_send:
            for v in self.variables:
                costs_v = factor_costs_for_var(self.factor, v, self._costs, self.mode)
                self.post_msg(v.name, MaxSumADVPMessage(costs_v))

    @register("maxsum_advp")
    def on_msg(self, variable_name, recv_msg, t):
        pass

    def footprint(self) -> float:
        return computation_memory(self.computation_def.node)

    def on_new_cycle(self, messages, cycle_id) -> Optional[List]:
        for sender, (message, t) in messages.items():
            self._costs[sender] = message.costs

        for v in self.variables:
            costs_v = factor_costs_for_var(self.factor, v, self._costs, self.mode)
            prev_costs, count = self._prev_messages[v.name]

            if self.damping_nodes in ["factors", "both"]:
                costs_v = apply_damping(costs_v, prev_costs, self.damping)

            # Check stability
            if not approx_match(costs_v, prev_costs, self.stability_coef):
                self.post_msg(v.name, MaxSumADVPMessage(costs_v))
                self._prev_messages[v.name] = costs_v, 1

            elif count < SAME_COUNT:
                self.post_msg(v.name, MaxSumADVPMessage(costs_v))
                self._prev_messages[v.name] = costs_v, count + 1
            else:
                # Converged locally
                pass

        return None


def factor_costs_for_var(factor: Constraint, variable: Variable, recv_costs, mode: str):
    """
    Computes marginals from factor to variable.
    """
    costs = {}
    other_vars = [v for v in factor.dimensions if v != variable]
    
    # Optimization: If binary constraint, avoid generic assignment generation
    is_binary = len(factor.dimensions) == 2
    
    # Pre-fetch costs to avoid lookups in inner loop
    other_costs_cache = {}
    for other in other_vars:
        if other.name in recv_costs:
            other_costs_cache[other.name] = recv_costs[other.name]
        else:
            other_costs_cache[other.name] = {}

    for d in variable.domain:
        optimal_value = float("inf") if mode == "min" else -float("inf")

        # Optimization for Binary Constraints (Fast Path)
        if is_binary and len(other_vars) == 1:
            other_v = other_vars[0]
            other_name = other_v.name
            known_costs = other_costs_cache[other_name]
            
            for d_other in other_v.domain:
                # Calculate F(v=d, other=d_other)
                # Note: factor check assumes kwargs
                f_val = factor(**{variable.name: d, other_name: d_other})
                
                cost_from_other = known_costs.get(d_other, 0)
                current_val = f_val + cost_from_other
                
                if mode == "min":
                    if current_val < optimal_value: optimal_value = current_val
                else:
                    if current_val > optimal_value: optimal_value = current_val

        else:
            # Generic N-ary Path (Slower)
            for assignment in generate_assignment_as_dict(other_vars):
                assignment[variable.name] = d
                f_val = factor(**assignment)

                sum_cost = 0
                valid_assignment = True
                
                for another_var_obj in other_vars:
                    a_name = another_var_obj.name
                    val_in_asgt = assignment[a_name]
                    
                    # Add cost from this neighbor if available
                    known_costs = other_costs_cache[a_name]
                    if val_in_asgt in known_costs:
                        sum_cost += known_costs[val_in_asgt]
                    
                    # Note: Original logic allowed proceeding even if cost not received
                    # We stick to that logic here.

                if valid_assignment:
                    current_val = f_val + sum_cost
                    if mode == "min":
                        if current_val < optimal_value: optimal_value = current_val
                    else:
                        if current_val > optimal_value: optimal_value = current_val

        costs[d] = optimal_value

    return costs


class MaxSumADVPVariableComputation(SynchronousComputationMixin, VariableComputation):
    def __init__(self, comp_def: ComputationDef):
        super().__init__(comp_def.node.variable, comp_def)
        assert comp_def.algo.algo in ["maxsum", "maxsum_advp"]
        
        self.mode = comp_def.algo.mode
        self.damping = comp_def.algo.params.get("damping", 0.5)
        self.damping_nodes = comp_def.algo.params.get("damping_nodes", "both")
        self.stability_coef = comp_def.algo.params.get("stability", STABILITY_COEFF)
        self.start_messages = comp_def.algo.params.get("start_messages", "leafs")
        
        self.factors = [link.factor_node for link in comp_def.node.links]
        self.costs = {}
        self._prev_messages = defaultdict(lambda: (None, 0))

        noise = comp_def.algo.params.get("noise", 0.0)
        if noise != 0:
            self._variable = VariableNoisyCostFunc(
                self.variable.name,
                self.variable.domain,
                cost_func=lambda x: self.variable.cost_for_val(x),
                initial_value=self.variable.initial_value,
                noise_level=noise,
            )

    @register("maxsum_advp")
    def on_msg(self, variable_name, recv_msg, t):
        pass

    def on_start(self) -> None:
        if self.variable.initial_value is not None:
            self.value_selection(self.variable.initial_value)
        else:
            val, _ = select_value(self.variable, self.costs, self.mode)
            self.value_selection(val)

        if len(self.factors) == 1 and self.start_messages == "leafs":
            f = self.factors[0]
            costs_f = costs_for_factor(self.variable, f, self.factors, self.costs)
            self.post_msg(f, MaxSumADVPMessage(costs_f))

        elif self.start_messages in ["leafs_vars", "all"]:
            for f in self.factors:
                costs_f = costs_for_factor(self.variable, f, self.factors, self.costs)
                self.post_msg(f, MaxSumADVPMessage(costs_f))

    def on_new_cycle(self, messages, cycle_id) -> Optional[List]:
        for sender, (message, t) in messages.items():
            self.costs[sender] = message.costs

        # Select value
        val, _ = select_value(self.variable, self.costs, self.mode)
        self.value_selection(val)

        # Send messages
        for f_name in self.factors:
            costs_f = costs_for_factor(self.variable, f_name, self.factors, self.costs)
            prev_costs, count = self._prev_messages[f_name]

            if self.damping_nodes in ["vars", "both"]:
                costs_f = apply_damping(costs_f, prev_costs, self.damping)

            if not approx_match(costs_f, prev_costs, self.stability_coef):
                self.post_msg(f_name, MaxSumADVPMessage(costs_f))
                self._prev_messages[f_name] = costs_f, 1

            elif count < SAME_COUNT:
                self.post_msg(f_name, MaxSumADVPMessage(costs_f))
                self._prev_messages[f_name] = costs_f, count + 1

        return None


def select_value(variable: Variable, costs: Dict[str, Dict], mode: str) -> Tuple[Any, float]:
    """Select the value for `variable` with the best cost."""
    d_costs = {d: variable.cost_for_val(d) for d in variable.domain}
    
    for f_costs in costs.values():
        for d in variable.domain:
            if d in f_costs:
                d_costs[d] += f_costs[d]

    if mode == "min":
        optimal_d = min(d_costs.items(), key=itemgetter(1))
    else:
        optimal_d = max(d_costs.items(), key=itemgetter(1))

    return optimal_d[0], optimal_d[1]


def costs_for_factor(
    variable: Variable, factor: FactorName, factors: List[Constraint], costs: Dict
) -> Dict[VarVal, Cost]:
    
    # Base costs (unary)
    msg_costs = {d: variable.cost_for_val(d) for d in variable.domain}
    sum_cost = 0

    # Sum costs from all OTHER factors
    for f in factors:
        if f == factor: continue # Skip target
        if f not in costs: continue # Skip if no message yet
        
        f_costs = costs[f]
        for d in variable.domain:
            if d in f_costs:
                c = f_costs[d]
                sum_cost += c
                msg_costs[d] += c

    # --- ADVANCED PROPAGATION: Normalization ---
    # This prevents cost explosion in loopy graphs
    if len(msg_costs) > 0:
        total_sum = sum(msg_costs.values())
        avg_cost = total_sum / len(msg_costs)
        normalized_msg_costs = {d: c - avg_cost for d, c in msg_costs.items()}
        return normalized_msg_costs
    
    return msg_costs


def apply_damping(costs_f, prev_costs, damping):
    if prev_costs is not None:
        damped_costs = {}
        for d, c in costs_f.items():
            prev = prev_costs.get(d, 0)
            damped_costs[d] = damping * prev + (1 - damping) * c
        return damped_costs
    return costs_f


def approx_match(costs, prev_costs, stability_coef):
    if prev_costs is None:
        return False
    
    # Check keys match
    if set(costs.keys()) != set(prev_costs.keys()):
        return False

    for d, c in costs.items():
        prev_c = prev_costs[d]
        if prev_c != c:
            delta = abs(prev_c - c)
            # Avoid division by zero
            denom = abs(prev_c + c)
            if denom == 0:
                # Both zero -> match. One zero -> mismatch (caught by prev_c != c)
                return False 
            
            if not ((2 * delta / denom) < stability_coef):
                return False
    return True