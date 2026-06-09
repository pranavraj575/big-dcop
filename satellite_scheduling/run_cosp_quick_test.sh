#!/bin/bash
# Quick test: Minimal scenario with iterative pricing

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=================================="
echo "COSPSolver Quick Test"
echo "Scenario: test_minimal.json"
echo "Framework: iterative_pricing"
echo "Max Iterations: 2"
echo "=================================="
echo

python3 main.py \
  --scenario test_minimal.json \
  --algorithms_json cosp_algorithm_configs.json \
  --output_json output/cosp_quick_test.json \
  --framework iterative_pricing \
  --max_iterations 2 \
  --timeout -1

echo
echo "✓ Test completed!"
echo "Results saved to: output/cosp_quick_test.json"
