#!/usr/bin/env python3
"""Test COSPSolver implementations."""

import sys

sys.path.insert(0, ".")

from cosp_solver import MGMSolver, DSASolver, MaxSumSolver, build_cosp


SIMPLE_PYDCOP = {
    "name": "test",
    "variables": ["v_a1_0", "v_a2_0"],
    "agents": ["a1", "a2"],
    "var_to_agent": {"v_a1_0": "a1", "v_a2_0": "a2"},
    "constraints": {"req_1": ["v_a1_0", "v_a2_0"]},
}


def test_solver_creation():
    config_mgm = {"algorithm": "mgm", "max_iterations": 5}
    solver_mgm = build_cosp(SIMPLE_PYDCOP, config_mgm)
    assert isinstance(solver_mgm, MGMSolver)
    assert solver_mgm.n_agents == 2
    assert solver_mgm.n_vars == 2
    print("✓ MGMSolver created successfully")

    config_dsa = {"algorithm": "dsa", "max_iterations": 5}
    solver_dsa = build_cosp(SIMPLE_PYDCOP, config_dsa)
    assert isinstance(solver_dsa, DSASolver)
    assert solver_dsa.n_agents == 2
    print("✓ DSASolver created successfully")

    config_maxsum = {"algorithm": "maxsum", "max_iterations": 5}
    solver_maxsum = build_cosp(SIMPLE_PYDCOP, config_maxsum)
    assert isinstance(solver_maxsum, MaxSumSolver)
    assert solver_maxsum.n_agents == 2
    print("✓ MaxSumSolver created successfully")


def test_solve():
    config_mgm = {"algorithm": "mgm", "max_iterations": 5}
    result_mgm = build_cosp(SIMPLE_PYDCOP, config_mgm).solve()
    assert result_mgm["algorithm"] == "MGM"
    assert "solution" in result_mgm
    print("✓ MGM solve completed:", result_mgm["iterations"], "iterations")

    config_dsa = {"algorithm": "dsa", "max_iterations": 5}
    result_dsa = build_cosp(SIMPLE_PYDCOP, config_dsa).solve()
    assert result_dsa["algorithm"] == "DSA"
    assert "solution" in result_dsa
    print("✓ DSA solve completed:", result_dsa["iterations"], "iterations")

    config_maxsum = {"algorithm": "maxsum", "max_iterations": 5}
    result_maxsum = build_cosp(SIMPLE_PYDCOP, config_maxsum).solve()
    assert result_maxsum["algorithm"] == "MaxSum"
    assert "solution" in result_maxsum
    print("✓ MaxSum solve completed:", result_maxsum["iterations"], "iterations")


def test_reward_fn():
    """Reward function: 1 when exactly one agent takes request, 1/n^2 for n>1."""
    pydcop = {
        "name": "test",
        "variables": ["v_a1_0", "v_a2_0", "v_a3_0"],
        "agents": ["a1", "a2", "a3"],
        "var_to_agent": {"v_a1_0": "a1", "v_a2_0": "a2", "v_a3_0": "a3"},
        "constraints": {"req_1": ["v_a1_0", "v_a2_0", "v_a3_0"]},
    }
    solver = build_cosp(pydcop, {"algorithm": "mgm", "max_iterations": 0})
    var_indices, fn = solver.constraints[0]

    assert fn(var_indices, [0, 0, 0]) == 0.0, "0 agents -> 0"
    assert fn(var_indices, [1, 0, 0]) == 1.0, "1 agent -> 1"
    assert abs(fn(var_indices, [1, 1, 0]) - 0.25) < 1e-9, "2 agents -> 0.25"
    assert abs(fn(var_indices, [1, 1, 1]) - 1 / 9) < 1e-9, "3 agents -> 1/9"
    print("✓ Reward function correct")


def test_index_structures():
    solver = build_cosp(SIMPLE_PYDCOP, {"algorithm": "mgm", "max_iterations": 1})
    assert solver.var_to_idx["v_a1_0"] == 0
    assert solver.var_to_idx["v_a2_0"] == 1
    assert solver.agent_of_var[0] == solver.agent_to_idx["a1"]
    assert solver.agent_of_var[1] == solver.agent_to_idx["a2"]
    a1 = solver.agent_to_idx["a1"]
    a2 = solver.agent_to_idx["a2"]
    assert a2 in solver.agent_neighbors[a1]
    assert a1 in solver.agent_neighbors[a2]
    print("✓ Index structures correct")


if __name__ == "__main__":
    print("Testing COSPSolver...\n")
    try:
        test_solver_creation()
        print()
        test_solve()
        print()
        test_reward_fn()
        print()
        test_index_structures()
        print("\n✓ All tests passed!")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
