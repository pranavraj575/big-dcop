#!/bin/bash
# Full test: Larger scenario with all algorithms

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=================================="
echo "COSPSolver Full Test"
echo "Scenario: test.json (38.9K)"
echo "Framework: constraint_generation"
echo "Max Iterations: 4"
echo "=================================="
echo

python3 main.py \
  --scenario test.json \
  --algorithms_json cosp_algorithm_configs.json \
  --output_json output/cosp_full_test.json \
  --framework constraint_generation \
  --max_iterations 4 \
  --timeout -1

echo
echo "✓ Test completed!"
echo "Results saved to: output/cosp_full_test.json"
