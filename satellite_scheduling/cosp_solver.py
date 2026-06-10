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
# MGM Solver
# ---------------------------------------------------------------------------

class MGMSolver(COSPSolver):
    """
    MGM (Maximum Gain Message) solver.

    Each iteration:
      1. All agents simultaneously compute the best value (0 or 1) for each
         owned variable based on the current shared state (snapshot).
      2. Gains are computed; an agent applies its move only if its gain is
         >= the gain of every neighboring agent (max-gain wins protocol).
    """

    def __init__(self, algorithm_config: dict, pydcop_dict: dict):
        super().__init__(algorithm_config, pydcop_dict)
        self.max_iterations = algorithm_config.get("max_iterations", 100)
        self.stop_cycle = algorithm_config.get("stop_cycle", 0)

    def _update(self):
        self.total_messages += self.messages_per_iter
        snapshot = list(self.assignments)

        # Per-variable gain: how much does flipping vi improve the local constraint sum?
        var_gains: List[float] = [0.0] * self.n_vars
        var_best_val: List[int] = list(snapshot)

        for vi in range(self.n_vars):
            orig = snapshot[vi]
            current_u = sum(self._constraint_value(c, snapshot) for c in self.var_to_constraints[vi])
            best_val, best_u = orig, current_u
            for val in (0, 1):
                if val == orig:
                    continue
                snapshot[vi] = val
                u = sum(self._constraint_value(c, snapshot) for c in self.var_to_constraints[vi])
                if u > best_u:
                    best_u, best_val = u, val
            snapshot[vi] = orig
            # Tiny noise breaks ties so symmetric agents don't all act/all freeze.
            var_gains[vi] = best_u - current_u + random.uniform(0, 1e-9)
            var_best_val[vi] = best_val

        # Per-variable coordination: vi is updated only if its gain beats the gain of
        # every other variable that shares a constraint with it.  This allows one
        # variable per constraint to move each iteration instead of serialising through
        # a single per-agent all-neighbors check (which blocks ~58/60 agents in dense
        # scenarios where every agent has ~43 global neighbors).
        for vi in range(self.n_vars):
            if var_gains[vi] <= 0 or var_best_val[vi] == snapshot[vi]:
                continue
            competitors: Set[int] = set()
            for c_idx in self.var_to_constraints[vi]:
                competitors.update(self.constraints[c_idx][0])
            competitors.discard(vi)
            if all(var_gains[vi] >= var_gains[vj] for vj in competitors):
                self.assignments[vi] = var_best_val[vi]

    def solve(self) -> Dict:
        prev = list(self.assignments)
        for iteration in range(self.max_iterations):
            if self.stop_cycle > 0 and iteration >= self.stop_cycle:
                break
            self._update()
            if self.assignments == prev:
                logger.info(f"MGM converged at iteration {iteration}")
                return {
                    "algorithm": "MGM",
                    "iterations": iteration,
                    "solution": self._extract_solution(),
                    "converged": True,
                    "messages_per_iter": self.messages_per_iter,
                    "total_messages": self.total_messages,
                }
            prev = list(self.assignments)

        return {
            "algorithm": "MGM",
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
    DSA (Distributed Stochastic Algorithm) solver.

    Each iteration every agent independently and stochastically decides
    whether to adopt its locally best value, with probability `probability`.
    """

    def __init__(self, algorithm_config: dict, pydcop_dict: dict):
        super().__init__(algorithm_config, pydcop_dict)
        self.max_iterations = algorithm_config.get("max_iterations", 100)
        self.probability = algorithm_config.get("probability", 0.7)
        self.stop_cycle = algorithm_config.get("stop_cycle", 0)
        # Stop early if the number of assignments that changed is below this
        # fraction of variables for `patience` consecutive iterations.
        # With p=0.7 DSA keeps jittering even at a good solution, so "no change"
        # convergence rarely fires; patience gives a practical cutoff.
        self.patience = algorithm_config.get("patience", 3)
        self.stability_threshold = algorithm_config.get("stability_threshold", 0.01)

    def _update(self):
        self.total_messages += self.messages_per_iter
        snapshot = list(self.assignments)

        for ai in range(self.n_agents):
            for vi in self.agent_var_indices[ai]:
                orig = snapshot[vi]

                # Only evaluate constraints containing vi — flipping vi cannot
                # affect any other constraint, so agent_utility (which sums ALL
                # of the agent's constraints) was doing ~51x redundant work.
                current_u = sum(self._constraint_value(c, snapshot) for c in self.var_to_constraints[vi])
                best_val, best_u = orig, current_u

                for val in (0, 1):
                    if val == orig:
                        continue
                    snapshot[vi] = val
                    u = sum(self._constraint_value(c, snapshot) for c in self.var_to_constraints[vi])
                    if u > best_u:
                        best_u, best_val = u, val
                snapshot[vi] = orig  # restore

                if best_val != orig and random.random() < self.probability:
                    self.assignments[vi] = best_val

    def solve(self) -> Dict:
        prev = list(self.assignments)
        stable_iters = 0
        for iteration in range(self.max_iterations):
            if self.stop_cycle > 0 and iteration >= self.stop_cycle:
                break
            self._update()
            # Exact convergence: nothing changed at all.
            if self.assignments == prev:
                logger.info(f"DSA converged at iteration {iteration}")
                return {
                    "algorithm": "DSA",
                    "variant": self.algorithm_config.get("variant", "B"),
                    "iterations": iteration,
                    "solution": self._extract_solution(),
                    "converged": True,
                    "messages_per_iter": self.messages_per_iter,
                    "total_messages": self.total_messages,
                }
            # Patience: stop if fewer than stability_threshold fraction of variables
            # changed for `patience` consecutive iterations.  With p=0.7, DSA keeps
            # jittering near a good solution and rarely hits exact convergence, so this
            # provides a practical early-stop without waiting for max_iterations.
            changed_frac = sum(a != b for a, b in zip(self.assignments, prev)) / max(self.n_vars, 1)
            stable_iters = stable_iters + 1 if changed_frac < self.stability_threshold else 0
            if stable_iters >= self.patience:
                logger.info(f"DSA stabilised at iteration {iteration} (changed_frac={changed_frac:.4f})")
                return {
                    "algorithm": "DSA",
                    "variant": self.algorithm_config.get("variant", "B"),
                    "iterations": iteration,
                    "solution": self._extract_solution(),
                    "converged": True,
                    "messages_per_iter": self.messages_per_iter,
                    "total_messages": self.total_messages,
                }
            prev = list(self.assignments)

        return {
            "algorithm": "DSA",
            "variant": self.algorithm_config.get("variant", "B"),
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
    MaxSum (belief propagation) solver.

    Each agent maintains a message score per owned variable.  Messages are
    damped by `damping` to improve stability.  Convergence is declared when
    no message changes by more than `stability`.
    """

    def __init__(self, algorithm_config: dict, pydcop_dict: dict):
        super().__init__(algorithm_config, pydcop_dict)
        self.max_iterations = algorithm_config.get("max_iterations", 100)
        self.damping = algorithm_config.get("damping", 0.0)
        self.stability = algorithm_config.get("stability", 1e-4)
        # message_scores[vi] = current belief score for variable vi
        self.message_scores: List[float] = [0.0] * self.n_vars

    def _update(self):
        self.total_messages += self.messages_per_iter
        # Async (Gauss-Seidel) update: process variables one at a time in random
        # order, each seeing the most recent live assignments.
        #
        # The synchronous (Jacobi) alternative causes oscillation: all variables
        # covering the same request see everyone else at 0, score 1.0, all flip to 1.
        # Next round they all see 1, score negative, all flip to 0.  Random shuffle
        # breaks this symmetry: the first variable in the order claims the request;
        # later variables see it taken and correctly back off.
        order = list(range(self.n_vars))
        random.shuffle(order)

        for vi in order:
            orig = self.assignments[vi]

            self.assignments[vi] = 1
            u1 = sum(self._constraint_value(c_idx, self.assignments)
                     for c_idx in self.var_to_constraints[vi])

            self.assignments[vi] = 0
            u0 = sum(self._constraint_value(c_idx, self.assignments)
                     for c_idx in self.var_to_constraints[vi])

            self.assignments[vi] = orig  # restore before committing new value

            # Small tie-breaking noise so equal-score variables don't systematically
            # all land on 0 or all land on 1.
            incoming = u1 - u0 + random.uniform(0, 1e-9)
            new_score = (1.0 - self.damping) * incoming + self.damping * self.message_scores[vi]
            self.message_scores[vi] = new_score
            self.assignments[vi] = 1 if new_score > 0 else 0

    def solve(self) -> Dict:
        prev_scores: List[float] = list(self.message_scores)
        for iteration in range(self.max_iterations):
            self._update()
            if all(
                abs(self.message_scores[vi] - prev_scores[vi]) <= self.stability
                for vi in range(self.n_vars)
            ):
                logger.info(f"MaxSum converged at iteration {iteration}")
                return {
                    "algorithm": "MaxSum",
                    "iterations": iteration,
                    "solution": self._extract_solution(),
                    "converged": True,
                    "messages_per_iter": self.messages_per_iter,
                    "total_messages": self.total_messages,
                }
            prev_scores = list(self.message_scores)

        return {
            "algorithm": "MaxSum",
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
    """

    # Binary domain used by all variables.
    _DOMAIN = (0, 1)

    def __init__(self, algorithm_config: dict, pydcop_dict: dict):
        super().__init__(algorithm_config, pydcop_dict)

        self.max_iterations   = algorithm_config.get("max_iterations", 100)
        self.stop_cycle       = algorithm_config.get("stop_cycle", 0)
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

        # Per-variable mixed strategy: strategy[vi] = [p(0), p(1)]
        self.strategy: List[np.ndarray] = [
            np.array([0.5, 0.5]) for _ in range(self.n_vars)
        ]

        # Cumulative regrets.
        # Non-context: regrets[vi] = np.array of shape (2,)
        # Context-based: regrets[vi] = dict mapping context_tuple -> np.array(2,)
        self.regrets: List = [
            (dict() if self.context_based else np.zeros(2))
            for _ in range(self.n_vars)
        ]

        # IR-PRM: prediction of utility vector from previous cycle
        self.ir_prm_pred: Optional[List[np.ndarray]] = (
            [np.zeros(2) for _ in range(self.n_vars)] if self.use_ir_prm else None
        )

        # Initialise assignments
        if self.det_start:
            self._deterministic_init()
        else:
            for vi in range(self.n_vars):
                self.assignments[vi] = random.randint(0, 1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_cum_regrets(self, vi: int) -> np.ndarray:
        if self.context_based:
            ctx = self._context(vi)
            if ctx not in self.regrets[vi]:
                self.regrets[vi][ctx] = np.zeros(2)
            return self.regrets[vi][ctx]
        return self.regrets[vi]

    def _set_cum_regrets(self, vi: int, r: np.ndarray):
        if self.context_based:
            self.regrets[vi][self._context(vi)] = r
        else:
            self.regrets[vi] = r

    def _context(self, vi: int):
        """Tuple of current assignments of all agents sharing a constraint with vi."""
        neighbors_vi: List[int] = []
        for c_idx in self.var_to_constraints[vi]:
            for vj in self.constraints[c_idx][0]:
                if vj != vi:
                    neighbors_vi.append(vj)
        return tuple(self.assignments[vj] for vj in sorted(set(neighbors_vi)))

    def _compute_utilities(self, vi: int) -> np.ndarray:
        """
        Return utility array [u(vi=0), u(vi=1)] using the current assignment
        snapshot for all other variables.
        """
        orig = self.assignments[vi]
        utils = np.zeros(2)
        for val in (0, 1):
            self.assignments[vi] = val
            utils[val] = sum(
                self._constraint_value(c, self.assignments)
                for c in self.var_to_constraints[vi]
            )
        self.assignments[vi] = orig
        return utils

    def _strategy_to_assignment(self, vi: int, strategy: np.ndarray) -> int:
        """Sample a value from the mixed strategy."""
        if random.random() >= self.update_prob:
            return self.assignments[vi]
        return int(np.random.choice(2, p=strategy))

    def _deterministic_init(self):
        """Start at the locally best value for each variable (greedy)."""
        for vi in range(self.n_vars):
            utils = self._compute_utilities(vi)
            self.assignments[vi] = int(np.argmax(utils))

    def _get_discounts(self, t: int):
        """Return (pos_discount, neg_discount) for iteration t (1-indexed)."""
        if self.use_discounted:
            pos = (t ** self.alpha) / (t ** self.alpha + 1)
            neg = (t ** self.beta) / (t ** self.beta + 1) if self.beta > 0 else 0.0
        else:
            pos = neg = 1.0
        return pos, neg

    # ------------------------------------------------------------------
    # Update rules (ported from pydcop RMComputation)
    # ------------------------------------------------------------------

    def _vanilla_rm_update(self, vi: int, utilities: np.ndarray, t: int) -> np.ndarray:
        """Standard / discounted / predictive RM update. Returns new strategy."""
        last_strat  = self.strategy[vi]
        instant_reg = utilities - float(np.dot(utilities, last_strat))

        pos_d, neg_d = self._get_discounts(t)
        cum_reg = self._get_cum_regrets(vi)

        if (pos_d, neg_d) != (1.0, 1.0):
            cum_reg = np.where(cum_reg > 0, cum_reg * pos_d, cum_reg * neg_d)
        cum_reg = cum_reg + instant_reg

        if self.use_rm_plus:
            cum_reg = np.clip(cum_reg, 0, None)
        self._set_cum_regrets(vi, cum_reg)

        basis = cum_reg + instant_reg if self.use_predictive else cum_reg
        pos_part = np.clip(basis, 0, None)
        pos_sum  = float(np.sum(pos_part))

        new_strat = np.array([0.5, 0.5]) if pos_sum <= 0 else pos_part / pos_sum

        if self.damping > 0:
            new_strat = self.damping * last_strat + (1 - self.damping) * new_strat
        return new_strat

    @staticmethod
    def _ir_prm_gamma(v: np.ndarray, t: float) -> float:
        """Slow but exact gamma computation (from pydcop ir_prm_get_gamma_slow)."""
        t2 = t ** 2
        v_sorted = -np.sort(-v)   # descending
        s = s2 = 0.0
        for k, vk in enumerate(v_sorted):
            s  += vk
            s2 += vk ** 2
            gamma = (s - np.sqrt(max(s ** 2 - (k + 1) * (s2 - t2), 0.0))) / (k + 1)
            if (k + 1) >= len(v_sorted) or gamma >= v_sorted[k + 1]:
                return gamma
        return gamma  # unreachable but satisfies type checker

    def _ir_prm_update(self, vi: int, utilities: np.ndarray) -> np.ndarray:
        """IR-PRM update (Ioannis version from pydcop). Returns new strategy."""
        pred       = self.ir_prm_pred[vi]
        last_strat = self.strategy[vi]
        u          = utilities - pred
        cum_reg    = self._get_cum_regrets(vi) + u - float(np.dot(u, last_strat))

        if self.use_rm_plus:
            cum_reg = np.clip(cum_reg, 0, None)

        if np.all(cum_reg <= 0):
            self._set_cum_regrets(vi, cum_reg)
            return last_strat

        self.ir_prm_pred[vi] = utilities
        old_norm = float(np.linalg.norm(np.clip(cum_reg, 0, None)))
        cum_reg  = cum_reg + utilities
        gamma    = self._ir_prm_gamma(cum_reg, old_norm)
        cum_reg  = cum_reg - gamma
        self._set_cum_regrets(vi, cum_reg)
        pos_part = np.clip(cum_reg, 0, None)
        return pos_part / float(np.sum(pos_part))

    # ------------------------------------------------------------------
    # Main iteration
    # ------------------------------------------------------------------

    def _update(self, t: int):
        """One synchronous iteration over all variables."""
        self.total_messages += self.messages_per_iter
        # Take a snapshot so all agents evaluate from the same state.
        snapshot = list(self.assignments)

        new_assignments = list(self.assignments)

        for vi in range(self.n_vars):
            # Compute utilities from snapshot
            orig = snapshot[vi]
            utils = np.zeros(2)
            for val in (0, 1):
                snapshot[vi] = val
                utils[val] = sum(
                    self._constraint_value(c, snapshot)
                    for c in self.var_to_constraints[vi]
                )
            snapshot[vi] = orig

            # Choose update rule
            if self.use_ir_prm:
                new_strat = self._ir_prm_update(vi, utils)
            else:
                new_strat = self._vanilla_rm_update(vi, utils, t)

            self.strategy[vi] = new_strat
            new_assignments[vi] = self._strategy_to_assignment(vi, new_strat)

        self.assignments = new_assignments

    def solve(self) -> Dict:
        prev = list(self.assignments)
        stable_iters = 0
        for iteration in range(self.max_iterations):
            if self.stop_cycle > 0 and iteration >= self.stop_cycle:
                break
            self._update(t=iteration + 1)

            if self.assignments == prev:
                logger.info(f"RM converged at iteration {iteration}")
                return self._result(iteration, converged=True)

            changed_frac = sum(a != b for a, b in zip(self.assignments, prev)) / max(self.n_vars, 1)
            stable_iters = stable_iters + 1 if changed_frac < 0.01 else 0
            if stable_iters >= 3:
                logger.info(f"RM stabilised at iteration {iteration}")
                return self._result(iteration, converged=True)

            prev = list(self.assignments)

        return self._result(self.max_iterations, converged=False)

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
