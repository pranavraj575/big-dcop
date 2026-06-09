#!/bin/bash
# Performance test: Large scenario with limited iterations

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=================================="
echo "COSPSolver Performance Test"
echo "Scenario: test_large.json (702K)"
echo "Framework: iterative_pricing"
echo "Max Iterations: 3"
echo "Timeout: 300 seconds"
echo "=================================="
echo

python3 main.py \
  --scenario test_large.json \
  --algorithms_json cosp_algorithm_configs.json \
  --output_json output/cosp_perf_test.json \
  --framework iterative_pricing \
  --max_iterations 3 \
  --timeout 300

echo
echo "✓ Performance test completed!"
echo "Results saved to: output/cosp_perf_test.json"
