from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Callable, Set, Optional
from collections import defaultdict
import random
import time
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constraint functions
# ---------------------------------------------------------------------------

def _make_reward_fn() -> Callable:
    """Returns the hardcoded per-request reward function.

    Utility = 1 when exactly one agent takes the request,
    1/n^2 when n>1 agents take it (discourages overlap), 0 when no one takes it.
    """
    def fn(var_indices: List[int], assignments: List[int]) -> float:
        n = sum(assignments[i] for i in var_indices)
        if n == 0:
            return 0.0
        if n == 1:
            return 1.0
        return 1.0 / (n * n)
    return fn


# ---------------------------------------------------------------------------
# Factory / dispatcher
# ---------------------------------------------------------------------------

def build_cosp(pydcop_dict: dict, algorithm_config: dict) -> "COSPSolver":
    """Instantiate the correct COSPSolver subclass."""
    name = algorithm_config.get("algorithm", "mgm").lower()
    if name == "mgm":
        return MGMSolver(algorithm_config, pydcop_dict)
    if name == "dsa":
        return DSASolver(algorithm_config, pydcop_dict)
    if name == "maxsum":
        return MaxSumSolver(algorithm_config, pydcop_dict)
    if name in ("regret_matching", "rm"):
        return RegretMatchingSolver(algorithm_config, pydcop_dict)
    raise ValueError(f"Unknown algorithm: {name}")


def cosp_run_global_dispatcher(pydcop_dict: dict, algorithm_config: dict):
    """Run solver in-process and return (result_dict, elapsed_seconds)."""
    solver = build_cosp(pydcop_dict, algorithm_config)
    t0 = time.time()
    solution = solver.solve()
    return solution, time.time() - t0


# ---------------------------------------------------------------------------
# Base solver
# ---------------------------------------------------------------------------

class COSPSolver(ABC):
    """
    Base class for all DCOP algorithm implementations.

    pydcop_dict schema
    ------------------
    {
        "name":        str,
        "variables":   List[str],            # all var names, e.g. ["v_a1_0", "v_a2_0"]
        "agents":      List[str],            # all agent ids
        "var_to_agent": {var_name: agent_id},# ownership map (required)
        "constraints": {
            c_name: [var_name, ...]           # list  -> hardcoded reward_fn applied
          | c_name: {"variables": [...], "fn": callable}         # callable already built
          | c_name: {"variables": [...], "function": str_expr}   # legacy string (step 3)
        }
    }
    """

    def __init__(self, algorithm_config: dict, pydcop_dict: dict):
        self.algorithm_config = algorithm_config

        variables: List[str] = pydcop_dict.get("variables", [])
        agents: List[str] = pydcop_dict.get("agents", [])

        self.variables = variables
        self.agents = agents
        self.n_vars = len(variables)
        self.n_agents = len(agents)

        self.var_to_idx: Dict[str, int] = {v: i for i, v in enumerate(variables)}
        self.agent_to_idx: Dict[str, int] = {a: i for i, a in enumerate(agents)}

        # Binary assignment vector: assignments[var_idx] in {0, 1}
        self.assignments: List[int] = [0] * self.n_vars

        # agent_var_indices[agent_idx] = sorted list of var indices owned by that agent
        self.agent_var_indices: List[List[int]] = [[] for _ in range(self.n_agents)]
        var_to_agent: Dict[str, str] = pydcop_dict.get("var_to_agent", {})
        for var_name, agent_id in var_to_agent.items():
            if var_name in self.var_to_idx and agent_id in self.agent_to_idx:
                ai = self.agent_to_idx[agent_id]
                vi = self.var_to_idx[var_name]
                self.agent_var_indices[ai].append(vi)

        # agent_of_var[var_idx] = agent_idx (-1 if unowned)
        self.agent_of_var: List[int] = [-1] * self.n_vars
        for ai, var_indices in enumerate(self.agent_var_indices):
            for vi in var_indices:
                self.agent_of_var[vi] = ai

        # constraints[c_idx] = (var_indices: List[int], fn: Callable)
        self.constraints: List[Tuple[List[int], Callable]] = []
        self._build_constraints(pydcop_dict.get("constraints", {}))

        # var_to_constraints[var_idx] = list of constraint indices that include this var
        self.var_to_constraints: List[List[int]] = [[] for _ in range(self.n_vars)]
        for c_idx, (var_indices, _) in enumerate(self.constraints):
            for vi in var_indices:
                self.var_to_constraints[vi].append(c_idx)

        # agent_neighbors[agent_idx] = list of agent indices sharing at least one constraint
        neighbor_sets: List[Set[int]] = [set() for _ in range(self.n_agents)]
        for var_indices, _ in self.constraints:
            agents_in_c = {self.agent_of_var[vi] for vi in var_indices if self.agent_of_var[vi] >= 0}
            for ai in agents_in_c:
                neighbor_sets[ai].update(agents_in_c - {ai})
        self.agent_neighbors: List[List[int]] = [list(s) for s in neighbor_sets]

        # Pre-compute would-be message count per iteration.
        # A message is counted each time an agent reads a variable owned by a different
        # agent while evaluating its constraints (one read = one would-be received message).
        # Constraints are fixed for the lifetime of a solver instance, so compute once.
        self.messages_per_iter: int = self._compute_messages_per_iter()
        self.total_messages: int = 0

    def _build_constraints(self, constraints_dict: dict):
        for c_name, c_spec in constraints_dict.items():
            if isinstance(c_spec, list):
                var_names = c_spec
                fn = _make_reward_fn()
            elif isinstance(c_spec, dict):
                var_names = c_spec.get("variables", [])
                fn = c_spec.get("fn", _make_reward_fn())
            else:
                continue

            var_indices = [self.var_to_idx[v] for v in var_names if v in self.var_to_idx]
            if var_indices:
                self.constraints.append((var_indices, fn))

    def _compute_messages_per_iter(self) -> int:
        """
        Count would-be messages for one iteration.

        For each agent, walk the unique set of constraints it participates in
        and count variables in those constraints owned by *other* agents.
        Each such variable is a value that would have been received as a message
        in a real distributed implementation.
        """
        count = 0
        for ai in range(self.n_agents):
            seen: Set[int] = set()
            for vi in self.agent_var_indices[ai]:
                for c_idx in self.var_to_constraints[vi]:
                    if c_idx not in seen:
                        seen.add(c_idx)
                        var_indices, _ = self.constraints[c_idx]
                        count += sum(1 for vj in var_indices if self.agent_of_var[vj] != ai)
        return count

    def _constraint_value(self, c_idx: int, assignments: List[int]) -> float:
        var_indices, fn = self.constraints[c_idx]
        return fn(var_indices, assignments)

    def _agent_utility(self, agent_idx: int, assignments: List[int]) -> float:
        """Sum of all constraint values that involve at least one variable owned by agent_idx."""
        seen: Set[int] = set()
        total = 0.0
        for vi in self.agent_var_indices[agent_idx]:
            for c_idx in self.var_to_constraints[vi]:
                if c_idx not in seen:
                    seen.add(c_idx)
                    total += self._constraint_value(c_idx, assignments)
        return total

    def _extract_solution(self) -> Dict:
        return {
            agent_id: [
                self.variables[vi]
                for vi in self.agent_var_indices[ai]
                if self.assignments[vi] == 1
            ]
            for ai, agent_id in enumerate(self.agents)
        }

    @abstractmethod
    def solve(self) -> Dict:
        pass


# ---------------------------------------------------------------------------
# MGM2 Solver
# ---------------------------------------------------------------------------

class MGMSolver(COSPSolver):
    """
    MGM2 (Maximum Gain Message, 2-agent coordination) solver.

    Each iteration runs a 3-phase protocol:

    Phase 1 — unilateral: every variable computes its best individual move
        and the corresponding gain (same as plain MGM per-variable).

    Phase 2 — pair offers: each variable independently decides to become an
        *offerer* with probability `threshold` (default 0.5).  An offerer
        picks one random constraint-neighbour as partner and searches the 4
        joint binary assignments to find the one that maximises the combined
        utility of both variables' constraint neighbourhoods.  The offer is
        accepted by the partner if the joint gain for the partner strictly
        exceeds the partner's unilateral gain.  Accepted pairs commit to the
        joint assignment and report individual joint gains for the next phase.

    Phase 3 — GO/NO-GO: a variable applies its move (joint or unilateral)
        only if its effective gain beats the effective gains of all
        constraint-local competitors (same max-gain rule as plain MGM, but
        *excluding* the committed partner from the competitor set).  For
        committed pairs both partners must individually win their respective
        competitions before either executes.

    Parameters
    ----------
    threshold : float (default 0.5)
        Probability that a variable becomes an offerer each iteration.
    max_iterations : int (default 100)
    stop_cycle : int (default 0 = disabled)
    """

    def __init__(self, algorithm_config: dict, pydcop_dict: dict):
        super().__init__(algorithm_config, pydcop_dict)
        self.max_iterations = algorithm_config.get("max_iterations", 100)
        self.stop_cycle = algorithm_config.get("stop_cycle", 0)
        self.threshold = float(algorithm_config.get("threshold", 0.5))

    def _update(self):
        self.total_messages += self.messages_per_iter
        snapshot = list(self.assignments)

        # ------------------------------------------------------------------
        # Phase 1: unilateral gains (identical to plain MGM per-variable)
        # ------------------------------------------------------------------
        uni_gains: List[float] = [0.0] * self.n_vars
        uni_best: List[int] = list(snapshot)

        for vi in range(self.n_vars):
            orig = snapshot[vi]
            cur_u = sum(self._constraint_value(c, snapshot) for c in self.var_to_constraints[vi])
            alt = 1 - orig
            snapshot[vi] = alt
            alt_u = sum(self._constraint_value(c, snapshot) for c in self.var_to_constraints[vi])
            snapshot[vi] = orig
            if alt_u > cur_u:
                uni_gains[vi] = alt_u - cur_u + random.uniform(0, 1e-9)
                uni_best[vi] = alt
            else:
                uni_gains[vi] = random.uniform(0, 1e-9)  # ≤0 gain, just noise

        # ------------------------------------------------------------------
        # Phase 2: pair offers
        # ------------------------------------------------------------------
        # paired_gain[vi]    = effective gain vi will report in the GO phase
        #                      (None if vi is unpaired → falls back to uni_gains[vi])
        # paired_best[vi]    = value vi intends to take under the joint plan
        # paired_partner[vi] = partner index (-1 if unpaired)
        paired_gain: List[Optional[float]] = [None] * self.n_vars
        paired_best: List[int] = list(snapshot)
        paired_partner: List[int] = [-1] * self.n_vars
        in_pair: List[bool] = [False] * self.n_vars

        order = list(range(self.n_vars))
        random.shuffle(order)

        for vi in order:
            if in_pair[vi] or random.random() >= self.threshold:
                continue

            # Find unpaired constraint neighbours
            neighbors: Set[int] = set()
            for c_idx in self.var_to_constraints[vi]:
                for vj in self.constraints[c_idx][0]:
                    if vj != vi and not in_pair[vj]:
                        neighbors.add(vj)
            if not neighbors:
                continue

            vj = random.choice(list(neighbors))
            orig_i, orig_j = snapshot[vi], snapshot[vj]

            # Best joint assignment: search all 4 binary combinations for (vi, vj)
            joint_c = list(set(self.var_to_constraints[vi]) | set(self.var_to_constraints[vj]))
            cur_joint = sum(self._constraint_value(c, snapshot) for c in joint_c)
            best_vi_val, best_vj_val = orig_i, orig_j
            best_joint = cur_joint

            for val_i in (0, 1):
                for val_j in (0, 1):
                    if val_i == orig_i and val_j == orig_j:
                        continue
                    snapshot[vi] = val_i
                    snapshot[vj] = val_j
                    u = sum(self._constraint_value(c, snapshot) for c in joint_c)
                    if u > best_joint:
                        best_joint = u
                        best_vi_val, best_vj_val = val_i, val_j
            snapshot[vi] = orig_i
            snapshot[vj] = orig_j

            if best_joint == cur_joint:
                continue  # No joint improvement — no offer

            # Individual gains for each variable under the proposed joint assignment
            snapshot[vi] = best_vi_val
            snapshot[vj] = best_vj_val
            new_ui = sum(self._constraint_value(c, snapshot) for c in self.var_to_constraints[vi])
            new_uj = sum(self._constraint_value(c, snapshot) for c in self.var_to_constraints[vj])
            snapshot[vi] = orig_i
            snapshot[vj] = orig_j
            cur_ui = sum(self._constraint_value(c, snapshot) for c in self.var_to_constraints[vi])
            cur_uj = sum(self._constraint_value(c, snapshot) for c in self.var_to_constraints[vj])

            gain_vi_j = new_ui - cur_ui + random.uniform(0, 1e-9)
            gain_vj_j = new_uj - cur_uj + random.uniform(0, 1e-9)

            # Partner (vj) accepts offer only if joint gain > own unilateral gain
            if gain_vj_j > uni_gains[vj]:
                paired_gain[vi] = gain_vi_j
                paired_gain[vj] = gain_vj_j
                paired_best[vi] = best_vi_val
                paired_best[vj] = best_vj_val
                paired_partner[vi] = vj
                paired_partner[vj] = vi
                in_pair[vi] = True
                in_pair[vj] = True

        # Effective gain used in GO-phase competition: joint if paired, else unilateral
        eff_gains: List[float] = [
            paired_gain[vi] if paired_gain[vi] is not None else uni_gains[vi]
            for vi in range(self.n_vars)
        ]

        # ------------------------------------------------------------------
        # Phase 3: GO / NO-GO — apply moves
        # ------------------------------------------------------------------
        def _wins_competition(vx: int, exclude_partner: int) -> bool:
            comps: Set[int] = set()
            for c_idx in self.var_to_constraints[vx]:
                comps.update(self.constraints[c_idx][0])
            comps.discard(vx)
            if exclude_partner >= 0:
                comps.discard(exclude_partner)
            return all(eff_gains[vx] >= eff_gains[vk] for vk in comps)

        for vi in range(self.n_vars):
            partner = paired_partner[vi]

            if partner >= 0:
                # Paired move: both partners must win their respective competitions
                if not _wins_competition(vi, partner):
                    continue
                if not _wins_competition(partner, vi):
                    continue
                self.assignments[vi] = paired_best[vi]
            else:
                # Unilateral move: vi must win against all constraint-local competitors
                if uni_best[vi] == snapshot[vi]:
                    continue
                if _wins_competition(vi, -1):
                    self.assignments[vi] = uni_best[vi]

    def solve(self) -> Dict:
        prev = list(self.assignments)
        for iteration in range(self.max_iterations):
            if self.stop_cycle > 0 and iteration >= self.stop_cycle:
                break
            self._update()
            if self.assignments == prev:
                logger.info(f"MGM2 converged at iteration {iteration}")
                return {
                    "algorithm": "MGM2",
                    "iterations": iteration,
                    "solution": self._extract_solution(),
                    "converged": True,
                    "messages_per_iter": self.messages_per_iter,
                    "total_messages": self.total_messages,
                }
            prev = list(self.assignments)

        return {
            "algorithm": "MGM2",
            "iterations": self.max_iterations,
            "solution": self._extract_solution(),
            "converged": False,
            "messages_per_iter": self.messages_per_iter,
            "total_messages": self.total_messages,
        }


# ---------------------------------------------------------------------------
# DSA Solver
# ---------------------------------------------------------------------------

class DSASolver(COSPSolver):
    """
    DSA-C (Distributed Stochastic Algorithm, variant C) solver.

    Variant C is the most aggressive DSA variant:

    * **delta > 0** (strict improvement): change to best value with probability p.
    * **delta == 0** (tie): remove the current value from the candidate set so
      the agent is forced to pick an *alternative* equally-good value, then
      change with probability p.  This escapes symmetric local optima that
      variants A and B would keep revisiting.
    * **delta < 0** (degradation): no change.

    For binary variables the tie-break simplifies to: when the alternative
    value yields identical utility, switch to it with probability p.

    Parameters
    ----------
    probability : float (default 0.7)
    max_iterations : int (default 100)
    patience : int (default 3)
        Early-stop after this many consecutive iterations where < stability_threshold
        fraction of variables change.
    stability_threshold : float (default 0.01)
    stop_cycle : int (default 0 = disabled)
    """

    def __init__(self, algorithm_config: dict, pydcop_dict: dict):
        super().__init__(algorithm_config, pydcop_dict)
        self.max_iterations = algorithm_config.get("max_iterations", 100)
        self.probability = algorithm_config.get("probability", 0.7)
        self.stop_cycle = algorithm_config.get("stop_cycle", 0)
        self.patience = algorithm_config.get("patience", 8)
        self.stability_threshold = algorithm_config.get("stability_threshold", 0.01)

    def _update(self):
        self.total_messages += self.messages_per_iter
        snapshot = list(self.assignments)

        for ai in range(self.n_agents):
            for vi in self.agent_var_indices[ai]:
                orig = snapshot[vi]
                alt = 1 - orig

                # Only evaluate constraints containing vi (51× cheaper than agent_utility).
                cur_u = sum(self._constraint_value(c, snapshot) for c in self.var_to_constraints[vi])
                snapshot[vi] = alt
                alt_u = sum(self._constraint_value(c, snapshot) for c in self.var_to_constraints[vi])
                snapshot[vi] = orig  # restore

                delta = alt_u - cur_u

                if delta > 0:
                    # DSA-C variant: improvement → change with probability p
                    if random.random() < self.probability:
                        self.assignments[vi] = alt
                elif delta == 0:
                    # DSA-C variant: tie → force switch to the alternative value
                    # (removes current_val from candidates per the pydcop spec)
                    if random.random() < self.probability:
                        self.assignments[vi] = alt
                # delta < 0: no change (all three variants agree)

    def solve(self) -> Dict:
        prev = list(self.assignments)
        stable_iters = 0
        for iteration in range(self.max_iterations):
            if self.stop_cycle > 0 and iteration >= self.stop_cycle:
                break
            self._update()
            # Exact convergence: nothing changed at all.
            if self.assignments == prev:
                logger.info(f"DSA-C converged at iteration {iteration}")
                return {
                    "algorithm": "DSA-C",
                    "iterations": iteration,
                    "solution": self._extract_solution(),
                    "converged": True,
                    "messages_per_iter": self.messages_per_iter,
                    "total_messages": self.total_messages,
                }
            # Patience: stop if fewer than stability_threshold fraction of variables
            # changed for `patience` consecutive iterations.
            changed_frac = sum(a != b for a, b in zip(self.assignments, prev)) / max(self.n_vars, 1)
            stable_iters = stable_iters + 1 if changed_frac < self.stability_threshold else 0
            if stable_iters >= self.patience:
                logger.info(f"DSA-C stabilised at iteration {iteration} (changed_frac={changed_frac:.4f})")
                return {
                    "algorithm": "DSA-C",
                    "iterations": iteration,
                    "solution": self._extract_solution(),
                    "converged": True,
                    "messages_per_iter": self.messages_per_iter,
                    "total_messages": self.total_messages,
                }
            prev = list(self.assignments)

        return {
            "algorithm": "DSA-C",
            "iterations": self.max_iterations,
            "solution": self._extract_solution(),
            "converged": False,
            "messages_per_iter": self.messages_per_iter,
            "total_messages": self.total_messages,
        }


# ---------------------------------------------------------------------------
# MaxSum Solver
# ---------------------------------------------------------------------------

class MaxSumSolver(COSPSolver):
    """
    MaxSum ADVP (Adaptive Damping and Variable Pruning) solver.

    ADVP adds three improvements over plain MaxSum:

    1. **Damping** — each variable's belief score is an EMA blend of the
       newly-computed incoming score and the previous score:
           score_new = (1 − damping) · incoming + damping · score_old
       Damping reduces oscillation in loopy graphs.

    2. **Normalization** — for each factor→variable message the mean of the
       message values is subtracted before delivery.  For binary variables the
       differential (u1 − u0) is unchanged, so numerical stability is
       improved without affecting the argmax.  In our scalar-score
       implementation this is implicit.

    3. **Per-variable stability pruning** (the "VP" part) — once a variable's
       score has been stable (relative change < `stability`) for SAME_COUNT
       consecutive iterations, that variable is skipped in subsequent
       iterations.  This lets the solver keep running for max_iterations but
       do progressively less work as variables freeze.  The SAME_COUNT minimum
       (default 4, matching pydcop) guarantees neighbours receive enough
       consistent messages before a variable stops broadcasting.

    Update order: async Gauss-Seidel (random shuffle each iteration) to avoid
    the synchronous-Jacobi oscillation where all co-covering variables flip in
    lockstep.

    Parameters
    ----------
    damping : float (default 0.5)
    stability : float (default 1e-4)
        Relative-change threshold for per-variable pruning:
        ``2·|prev−curr| / (|prev|+|curr|+ε) < stability``
    same_count : int (default 4)
        Minimum number of stable consecutive iterations before a variable
        is pruned (matches SAME_COUNT in pydcop maxsum_advp).
    max_iterations : int (default 100)
    """

    _SAME_COUNT_DEFAULT = 8

    def __init__(self, algorithm_config: dict, pydcop_dict: dict):
        super().__init__(algorithm_config, pydcop_dict)
        self.max_iterations = algorithm_config.get("max_iterations", 100)
        self.damping = float(algorithm_config.get("damping", 0.5))
        self.stability = float(algorithm_config.get("stability", 1e-4))
        self.same_count = int(algorithm_config.get("same_count", self._SAME_COUNT_DEFAULT))
        # message_scores[vi] = current damped belief score for variable vi
        self.message_scores: List[float] = [0.0] * self.n_vars
        # stable_iters[vi] = consecutive iterations where vi's score was stable
        self._stable_iters: List[int] = [0] * self.n_vars

    @staticmethod
    def _approx_stable(prev: float, curr: float, tol: float) -> bool:
        """Relative-change stability test matching pydcop approx_match."""
        denom = abs(prev) + abs(curr)
        if denom == 0.0:
            return True
        return 2.0 * abs(prev - curr) / denom < tol

    def _update(self):
        self.total_messages += self.messages_per_iter
        # Async (Gauss-Seidel): random order, each variable sees the latest live values.
        order = list(range(self.n_vars))
        random.shuffle(order)

        for vi in order:
            # VP: skip variables that have already stabilised for ≥ same_count iters
            if self._stable_iters[vi] >= self.same_count:
                continue

            orig = self.assignments[vi]

            self.assignments[vi] = 1
            u1 = sum(self._constraint_value(c_idx, self.assignments)
                     for c_idx in self.var_to_constraints[vi])

            self.assignments[vi] = 0
            u0 = sum(self._constraint_value(c_idx, self.assignments)
                     for c_idx in self.var_to_constraints[vi])

            self.assignments[vi] = orig

            # Normalization: subtract mean so scores don't drift unboundedly.
            # For binary {u0, u1}: mean = (u0+u1)/2, normalised diff = u1−u0
            # (unchanged), but we track normalised_u1 = (u1−u0)/2 as the score
            # so scores stay bounded even in dense loopy graphs.
            incoming = (u1 - u0) / 2.0 + random.uniform(0, 1e-9)

            prev_score = self.message_scores[vi]
            new_score = (1.0 - self.damping) * incoming + self.damping * prev_score
            self.message_scores[vi] = new_score
            self.assignments[vi] = 1 if new_score > 0 else 0

            # Per-variable stability tracking
            if self._approx_stable(prev_score, new_score, self.stability):
                self._stable_iters[vi] += 1
            else:
                self._stable_iters[vi] = 0

    def solve(self) -> Dict:
        for iteration in range(self.max_iterations):
            self._update()
            # Global convergence: all variables have stabilised for ≥ same_count iters
            if all(self._stable_iters[vi] >= self.same_count for vi in range(self.n_vars)):
                logger.info(f"MaxSum ADVP converged at iteration {iteration}")
                return {
                    "algorithm": "MaxSum-ADVP",
                    "iterations": iteration,
                    "solution": self._extract_solution(),
                    "converged": True,
                    "messages_per_iter": self.messages_per_iter,
                    "total_messages": self.total_messages,
                }

        return {
            "algorithm": "MaxSum-ADVP",
            "iterations": self.max_iterations,
            "solution": self._extract_solution(),
            "converged": False,
            "messages_per_iter": self.messages_per_iter,
            "total_messages": self.total_messages,
        }


# ---------------------------------------------------------------------------
# Regret Matching Solver
# ---------------------------------------------------------------------------

class RegretMatchingSolver(COSPSolver):
    """
    Regret Matching (RM) and variants.

    Each agent maintains a mixed strategy (probability distribution over its
    binary domain) and updates it based on cumulative regret.  Supported
    variants (all via algorithm_config / algo_params):

        rm_plus            (bool, default False)  — clip negative regrets to 0 (RM+)
        predictive         (bool, default False)  — add instant regret to basis (Predictive RM)
        ir_prm             (bool, default False)  — IR-PRM update rule
        discounted_rm      (bool, default False)  — discount old regrets over time
        alpha              (float, default 1.5)   — discount exponent for positive regrets
        beta               (float, default 0.0)   — discount exponent for negative regrets
        damping            (float, default 0.0)   — blend new strategy with previous
        update_prob        (float, default 1.0)   — probability of actually updating value
        deterministic_start(bool, default False)  — start at best deterministic value instead of random
        stop_cycle         (int,   default 0)     — hard iteration cap (0 = use max_iterations)
        max_iterations     (int,   default 100)   — maximum iterations
        context_based      (bool, default False)  — maintain separate regret table per
                                                    neighbor-value context (memory-intensive)

    All per-variable state is kept as plain Python floats (not numpy arrays).
    For a binary domain {0, 1}, a 2-element strategy [p0, p1] reduces to a
    single scalar p1 (since p0 = 1 - p1), and cumulative regrets [r0, r1]
    reduce to the differential r1 - r0.  This eliminates all numpy overhead
    in the inner loop (~51x speedup on scenario_1 vs the numpy version).
    """

    def __init__(self, algorithm_config: dict, pydcop_dict: dict):
        super().__init__(algorithm_config, pydcop_dict)

        self.max_iterations   = algorithm_config.get("max_iterations", 100)
        self.stop_cycle       = algorithm_config.get("stop_cycle", 20)
        self.use_rm_plus      = bool(algorithm_config.get("rm_plus", False))
        self.use_predictive   = bool(algorithm_config.get("predictive", False))
        self.use_ir_prm       = bool(algorithm_config.get("ir_prm", False))
        self.context_based    = bool(algorithm_config.get("context_based", False))
        self.det_start        = bool(algorithm_config.get("deterministic_start", False))
        self.update_prob      = float(algorithm_config.get("update_prob", 1.0))
        self.use_discounted   = bool(algorithm_config.get("discounted_rm", False))
        self.alpha            = float(algorithm_config.get("alpha", 1.5))
        self.beta             = float(algorithm_config.get("beta", 0.0))
        self.damping          = float(algorithm_config.get("damping", 0.0))

        # strategy_p1[vi] = P(vi=1);  P(vi=0) = 1 - p1.
        self.strategy_p1: List[float] = [0.5] * self.n_vars

        # Cumulative regrets stored as (r0, r1) plain Python float pairs.
        # context_based: dict[context_tuple] -> (r0, r1)
        self.cum_r0: List = [dict() if self.context_based else 0.0] * self.n_vars
        self.cum_r1: List = [dict() if self.context_based else 0.0] * self.n_vars
        # Avoid aliasing from list multiplication for the dict case
        if self.context_based:
            self.cum_r0 = [dict() for _ in range(self.n_vars)]
            self.cum_r1 = [dict() for _ in range(self.n_vars)]

        # IR-PRM: predicted utilities (u0_pred, u1_pred) per variable
        self.ir_prm_pred0: Optional[List[float]] = (
            [0.0] * self.n_vars if self.use_ir_prm else None
        )
        self.ir_prm_pred1: Optional[List[float]] = (
            [0.0] * self.n_vars if self.use_ir_prm else None
        )

        if self.det_start:
            self._deterministic_init()
        else:
            for vi in range(self.n_vars):
                self.assignments[vi] = random.randint(0, 1)

    # ------------------------------------------------------------------
    # Plain-Python helpers (no numpy)
    # ------------------------------------------------------------------

    def _get_regrets(self, vi: int):
        """Return (r0, r1) for variable vi."""
        if self.context_based:
            ctx = self._context(vi)
            return self.cum_r0[vi].get(ctx, 0.0), self.cum_r1[vi].get(ctx, 0.0)
        return self.cum_r0[vi], self.cum_r1[vi]

    def _set_regrets(self, vi: int, r0: float, r1: float):
        if self.context_based:
            ctx = self._context(vi)
            self.cum_r0[vi][ctx] = r0
            self.cum_r1[vi][ctx] = r1
        else:
            self.cum_r0[vi] = r0
            self.cum_r1[vi] = r1

    def _context(self, vi: int):
        nbr_vars: Set[int] = set()
        for c_idx in self.var_to_constraints[vi]:
            nbr_vars.update(self.constraints[c_idx][0])
        nbr_vars.discard(vi)
        return tuple(self.assignments[vj] for vj in sorted(nbr_vars))

    def _compute_utils(self, vi: int, snapshot: List[int]):
        """Return (u0, u1) using snapshot for all other variables."""
        orig = snapshot[vi]
        snapshot[vi] = 1
        u1 = sum(self._constraint_value(c, snapshot) for c in self.var_to_constraints[vi])
        snapshot[vi] = 0
        u0 = sum(self._constraint_value(c, snapshot) for c in self.var_to_constraints[vi])
        snapshot[vi] = orig
        return u0, u1

    def _deterministic_init(self):
        snap = list(self.assignments)
        for vi in range(self.n_vars):
            u0, u1 = self._compute_utils(vi, snap)
            self.assignments[vi] = 1 if u1 >= u0 else 0

    def _get_discounts(self, t: int):
        if self.use_discounted:
            ta  = t ** self.alpha
            pos = ta / (ta + 1.0)
            neg = 0.0
            if self.beta > 0:
                tb  = t ** self.beta
                neg = tb / (tb + 1.0)
        else:
            pos = neg = 1.0
        return pos, neg

    @staticmethod
    def _strategy_from_regrets(r0: float, r1: float) -> float:
        """Convert (r0, r1) cumulative regrets to p1 probability."""
        p0 = max(r0, 0.0)
        p1 = max(r1, 0.0)
        s  = p0 + p1
        return (p1 / s) if s > 0 else 0.5

    def _sample_assignment(self, vi: int, p1: float) -> int:
        if random.random() >= self.update_prob:
            return self.assignments[vi]
        return 1 if random.random() < p1 else 0

    # ------------------------------------------------------------------
    # Update rules — plain Python floats, no numpy allocations
    # ------------------------------------------------------------------

    def _vanilla_rm_update(self, vi: int, u0: float, u1: float, t: int) -> float:
        """Standard / discounted / predictive RM. Returns new p1."""
        p1   = self.strategy_p1[vi]
        exp  = u0 * (1.0 - p1) + u1 * p1          # expected utility
        ir0  = u0 - exp                             # instant regret for val=0
        ir1  = u1 - exp                             # instant regret for val=1

        pos_d, neg_d = self._get_discounts(t)
        r0, r1 = self._get_regrets(vi)

        if (pos_d, neg_d) != (1.0, 1.0):
            r0 = r0 * pos_d if r0 > 0 else r0 * neg_d
            r1 = r1 * pos_d if r1 > 0 else r1 * neg_d
        r0 += ir0
        r1 += ir1

        if self.use_rm_plus:
            r0 = max(r0, 0.0)
            r1 = max(r1, 0.0)
        self._set_regrets(vi, r0, r1)

        if self.use_predictive:
            b0, b1 = r0 + ir0, r1 + ir1
        else:
            b0, b1 = r0, r1

        new_p1 = self._strategy_from_regrets(b0, b1)

        if self.damping > 0:
            new_p1 = self.damping * p1 + (1.0 - self.damping) * new_p1
        return new_p1

    @staticmethod
    def _ir_prm_gamma(v0: float, v1: float, t_sq: float) -> float:
        """Closed-form gamma for 2-element vector, replacing pydcop's slow loop."""
        a, b = (v0, v1) if v0 >= v1 else (v1, v0)
        # k=0 candidate
        g0 = a - (t_sq ** 0.5)
        if g0 >= b:
            return g0
        # k=1 candidate (use both elements)
        s    = a + b
        s2   = a * a + b * b
        disc = s * s - 2.0 * (s2 - t_sq)
        return (s - (max(disc, 0.0) ** 0.5)) / 2.0

    def _ir_prm_update(self, vi: int, u0: float, u1: float) -> float:
        """IR-PRM update (Ioannis version). Returns new p1."""
        pred0, pred1 = self.ir_prm_pred0[vi], self.ir_prm_pred1[vi]
        p1           = self.strategy_p1[vi]
        g0 = u0 - pred0
        g1 = u1 - pred1
        exp_g = g0 * (1.0 - p1) + g1 * p1
        r0, r1 = self._get_regrets(vi)
        r0 += g0 - exp_g
        r1 += g1 - exp_g

        if self.use_rm_plus:
            r0, r1 = max(r0, 0.0), max(r1, 0.0)

        if r0 <= 0 and r1 <= 0:
            self._set_regrets(vi, r0, r1)
            return p1

        self.ir_prm_pred0[vi] = u0
        self.ir_prm_pred1[vi] = u1
        old_norm_sq = max(r0, 0.0) ** 2 + max(r1, 0.0) ** 2
        r0 += u0
        r1 += u1
        gamma = self._ir_prm_gamma(r0, r1, old_norm_sq)
        r0 -= gamma
        r1 -= gamma
        self._set_regrets(vi, r0, r1)
        return self._strategy_from_regrets(r0, r1)

    # ------------------------------------------------------------------
    # Main iteration
    # ------------------------------------------------------------------

    def _update(self, t: int):
        self.total_messages += self.messages_per_iter
        snapshot = list(self.assignments)
        new_assignments = list(self.assignments)

        for vi in range(self.n_vars):
            u0, u1 = self._compute_utils(vi, snapshot)

            if self.use_ir_prm:
                new_p1 = self._ir_prm_update(vi, u0, u1)
            else:
                new_p1 = self._vanilla_rm_update(vi, u0, u1, t)

            self.strategy_p1[vi] = new_p1
            new_assignments[vi] = self._sample_assignment(vi, new_p1)

        self.assignments = new_assignments

    def solve(self) -> Dict:
        # RM does not have a clean convergence signal in contested environments:
        # - Assignment-based checks fail because stochastic sampling keeps
        #   jittering even after strategies stabilise.
        # - Strategy-change checks (max/mean dp) fail because a small subset of
        #   "contested" variables permanently swing as neighbours re-sample.
        # - Pure-strategy checks fire after a single iteration (all vars greedily
        #   jump to p1≈1 before contention resolves) giving terrible solutions.
        #
        # The correct budget control is stop_cycle (set in algo_params).  RM
        # quality plateaus within ~20–30 iterations; stop_cycle:30 gives good
        # results at a message cost comparable to DSA-C.
        n_iters = self.stop_cycle if self.stop_cycle > 0 else self.max_iterations
        for iteration in range(n_iters):
            self._update(t=iteration + 1)

        return self._result(n_iters, converged=False)

    def _extract_solution(self) -> Dict:
        """
        Per-constraint argmax extraction.

        RM's mixed strategy causes many variables to be sampled as 1
        simultaneously (heavy overcoverage), which makes OR-tools scheduling
        slow.  Instead of using the last stochastic sample, we deterministically
        assign 1 to the variable with the highest strategy_p1 within each
        constraint (ties broken by index), giving at most one agent per request.
        Variables not participating in any reward constraint keep their sampled
        assignment.
        """
        # Start with all zeros.
        final_assignments = [0] * self.n_vars

        # For each constraint, award 1 to the variable with the highest p1.
        for var_indices, _ in self.constraints:
            if len(var_indices) == 1:
                # Unary constraint (penalty): preserve the stochastic assignment.
                vi = var_indices[0]
                final_assignments[vi] = self.assignments[vi]
            else:
                # Reward constraint: highest-p1 variable wins.
                winner = max(var_indices, key=lambda vi: self.strategy_p1[vi])
                if self.strategy_p1[winner] > 0:
                    final_assignments[winner] = 1

        return {
            agent_id: [
                self.variables[vi]
                for vi in self.agent_var_indices[ai]
                if final_assignments[vi] == 1
            ]
            for ai, agent_id in enumerate(self.agents)
        }

    def _result(self, iterations: int, converged: bool) -> Dict:
        return {
            "algorithm": "RegretMatching",
            "iterations": iterations,
            "solution": self._extract_solution(),
            "converged": converged,
            "messages_per_iter": self.messages_per_iter,
            "total_messages": self.total_messages,
        }


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def cosp_parse_json_to_dcop_and_overlaps(json_filepath: str):
    """
    Reads a scenario JSON, builds a pydcop_dict for COSPSolver, and extracts
    per-agent task / downlink / capacity data for the local scheduler.
    """
    with open(json_filepath, "r") as f:
        data = json.load(f)

    requests = data.get("requests", [])

    pydcop: Dict = {
        "name": "Satellite_Request_Allocation",
        "variables": [],
        "agents": [],
        "var_to_agent": {},
        "constraints": {},
    }

    agent_tasks: Dict = defaultdict(list)
    agent_downlinks: Dict = defaultdict(list)
    agent_capacities: Dict = {}

    req_to_agents: Dict = defaultdict(list)
    agent_to_reqs: Dict = defaultdict(set)

    for agent_data in data.get("agents", []):
        agent_id = str(agent_data.get("agent_id", agent_data.get("identifier")))
        agent_capacities[agent_id] = float(agent_data.get("data_volume_MB", 1_000_000.0))
        pydcop["agents"].append(agent_id)

        for dl in agent_data.get("downlinks", []):
            agent_downlinks[agent_id].append({
                "start": float(dl["start"]),
                "end": float(dl["end"]),
                "data": float(dl["data_volume_MB"]),
            })

        for f in agent_data.get("fulfillments", []):
            req_id = str(f["request_id"])
            task_id = str(f["fulfillment_id"])
            agent_to_reqs[agent_id].add(req_id)
            if agent_id not in req_to_agents[req_id]:
                req_to_agents[req_id].append(agent_id)
            agent_tasks[agent_id].append({
                "task_id": task_id,
                "req_id": req_id,
                "start": float(f["start_time"]),
                "end": float(f["end_time"]),
                "data": float(f["data_volume_MB"]),
            })

    req_to_vars: Dict = defaultdict(list)
    var_to_details: Dict = {}

    for agent_id, reqs in agent_to_reqs.items():
        for var_counter, req_id in enumerate(reqs):
            var_name = f"v_{agent_id}_{var_counter}"
            pydcop["variables"].append(var_name)
            pydcop["var_to_agent"][var_name] = agent_id
            req_to_vars[req_id].append(var_name)
            var_to_details[var_name] = {"req_id": req_id, "agent_id": agent_id}

    # One reward constraint per request — hardcoded fn applied by _build_constraints
    for req_id, var_list in req_to_vars.items():
        pydcop["constraints"][f"reward_req_{req_id}"] = var_list

    return pydcop, agent_tasks, agent_downlinks, agent_capacities, var_to_details, requests


if __name__ == "__main__":
    fp = "scenarios/scenario_0.json"
    pydcop, *_ = cosp_parse_json_to_dcop_and_overlaps(fp)
    print(pydcop.keys())
    solver = build_cosp(pydcop, {"algorithm": "mgm", "max_iterations": 10})
    result = solver.solve()
    print(result)
