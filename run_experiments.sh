#!/usr/bin/env bash
# run_experiments.sh
#
# Runs every algorithm in cosp_algorithm_configs.json on every scenario in
# satellite_scheduling/scenarios/ under both the iterative_pricing and
# constraint_generation frameworks.
#
# Output JSONs are written to:
#   output/<framework>/<scenario_stem>.json
#
# Usage:
#   bash run_experiments.sh [--max-iterations N] [--output-dir DIR]
#
# Defaults:
#   --max-iterations  4
#   --output-dir      output

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MAX_ITER=4
OUTPUT_DIR="output"
PYTHON="/Users/itai/big-dcop/venv/bin/python3"
SCRIPT="satellite_scheduling/main.py"
ALGORITHMS_JSON="satellite_scheduling/cosp_algorithm_configs.json"
SCENARIOS_DIR="satellite_scheduling/scenarios"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-iterations)
      MAX_ITER="$2"; shift 2;;
    --output-dir)
      OUTPUT_DIR="$2"; shift 2;;
    --python)
      PYTHON="$2"; shift 2;;
    *)
      echo "Unknown argument: $1" >&2; exit 1;;
  esac
done

FRAMEWORKS=("constraint_generation" "iterative_pricing")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
total=0
passed=0
failed=0
failed_list=()

for framework in "${FRAMEWORKS[@]}"; do
  out_dir="${OUTPUT_DIR}/${framework}"
  mkdir -p "${out_dir}"

  for scenario_path in "${SCENARIOS_DIR}"/scenario_*.json; do
    scenario_stem=$(basename "${scenario_path}" .json)
    output_json="${out_dir}/${scenario_stem}.json"

    total=$((total + 1))
    echo "--------------------------------------------------------------"
    echo "  framework : ${framework}"
    echo "  scenario  : ${scenario_stem}"
    echo "  output    : ${output_json}"
    echo "--------------------------------------------------------------"

    # Remove stale output so main.py can write fresh results
    rm -f "${output_json}"

    if "python" "${SCRIPT}" \
        --scenario "${scenario_path}" \
        --output_json "${output_json}" \
        --algorithms_json "${ALGORITHMS_JSON}" \
        --framework "${framework}" \
        --max_iterations "${MAX_ITER}"; then
      passed=$((passed + 1))
    else
      failed=$((failed + 1))
      failed_list+=("${framework}/${scenario_stem}")
      echo "  [FAILED] ${framework}/${scenario_stem}" >&2
    fi

    echo ""
  done
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  Experiments complete"
echo "  Total  : ${total}"
echo "  Passed : ${passed}"
echo "  Failed : ${failed}"
if [[ ${failed} -gt 0 ]]; then
  echo "  Failed runs:"
  for f in "${failed_list[@]}"; do
    echo "    - ${f}"
  done
fi
echo "============================================================"
