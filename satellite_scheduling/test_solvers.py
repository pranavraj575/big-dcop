#!/usr/bin/env python3
"""Test COSPSolver implementations."""
import sys
sys.path.insert(0, '.')

from cosp_solver import COSPSolver, MGMSolver, DSASolver, MaxSumSolver
from cosp_solver import MGMAgent, DSAAgent, MaxSumAgent
from cosp_solver import build_cosp


def test_solver_creation():
    """Test creating solver instances."""
    pydcop_dict = {
        "name": "test",
        "variables": ["v_a1_0", "v_a2_0"],
        "agents": ["a1", "a2"],
        "constraints": {
            "req_1": ["v_a1_0", "v_a2_0"]
        }
    }

    # Test MGM
    config_mgm = {"algorithm": "mgm", "max_iterations": 5}
    solver_mgm = build_cosp(pydcop_dict, config_mgm)
    assert isinstance(solver_mgm, MGMSolver), "MGMSolver creation failed"
    assert len(solver_mgm.agents) == 2, "Wrong number of agents"
    print("✓ MGMSolver created successfully")

    # Test DSA
    config_dsa = {"algorithm": "dsa", "max_iterations": 5}
    solver_dsa = build_cosp(pydcop_dict, config_dsa)
    assert isinstance(solver_dsa, DSASolver), "DSASolver creation failed"
    assert len(solver_dsa.agents) == 2, "Wrong number of agents"
    print("✓ DSASolver created successfully")

    # Test MaxSum
    config_maxsum = {"algorithm": "maxsum", "max_iterations": 5}
    solver_maxsum = build_cosp(pydcop_dict, config_maxsum)
    assert isinstance(solver_maxsum, MaxSumSolver), "MaxSumSolver creation failed"
    assert len(solver_maxsum.agents) == 2, "Wrong number of agents"
    print("✓ MaxSumSolver created successfully")


def test_solve():
    """Test running solve methods."""
    pydcop_dict = {
        "name": "test",
        "variables": ["v_a1_0", "v_a2_0"],
        "agents": ["a1", "a2"],
        "constraints": {
            "req_1": ["v_a1_0", "v_a2_0"]
        }
    }

    # Test MGM solve
    config_mgm = {"algorithm": "mgm", "max_iterations": 5}
    solver_mgm = build_cosp(pydcop_dict, config_mgm)
    result_mgm = solver_mgm.solve()
    assert result_mgm["algorithm"] == "MGM", "MGM result incorrect"
    assert "solution" in result_mgm, "Solution missing from result"
    print("✓ MGM solve completed:", result_mgm["iterations"], "iterations")

    # Test DSA solve
    config_dsa = {"algorithm": "dsa", "max_iterations": 5}
    solver_dsa = build_cosp(pydcop_dict, config_dsa)
    result_dsa = solver_dsa.solve()
    assert result_dsa["algorithm"] == "DSA", "DSA result incorrect"
    assert "solution" in result_dsa, "Solution missing from result"
    print("✓ DSA solve completed:", result_dsa["iterations"], "iterations")

    # Test MaxSum solve
    config_maxsum = {"algorithm": "maxsum", "max_iterations": 5}
    solver_maxsum = build_cosp(pydcop_dict, config_maxsum)
    result_maxsum = solver_maxsum.solve()
    assert result_maxsum["algorithm"] == "MaxSum", "MaxSum result incorrect"
    assert "solution" in result_maxsum, "Solution missing from result"
    print("✓ MaxSum solve completed:", result_maxsum["iterations"], "iterations")


def test_agent_types():
    """Test agent type initialization."""
    config = {"variant": "B", "probability": 0.7, "damping": 0.5}

    mgm_agent = MGMAgent("a1", config)
    assert isinstance(mgm_agent, MGMAgent), "MGMAgent creation failed"
    print("✓ MGMAgent created successfully")

    dsa_agent = DSAAgent("a1", config)
    assert isinstance(dsa_agent, DSAAgent), "DSAAgent creation failed"
    print("✓ DSAAgent created successfully")

    maxsum_agent = MaxSumAgent("a1", config)
    assert isinstance(maxsum_agent, MaxSumAgent), "MaxSumAgent creation failed"
    print("✓ MaxSumAgent created successfully")


if __name__ == "__main__":
    print("Testing COSPSolver implementations...\n")
    
    try:
        test_solver_creation()
        print()
        test_solve()
        print()
        test_agent_types()
        print("\n✓ All tests passed!")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
