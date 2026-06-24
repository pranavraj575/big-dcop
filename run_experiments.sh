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
#   bash run_experiments.sh [--scenarios SCENARIOS_DIR] [--max-iterations N] [--output-dir DIR] [--trials NUM_TRIALS]
#
# Defaults:
#   --max-iterations  4
#   --output-dir      output
#   --scenarios       satellite_scheduling/scenarios_larger
#   --trials          2

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MAX_ITER=4
TRIALS=2
USE_SLURM_JOBS=1
PROJECT_DIR=$(readlink -e $(dirname $0))
OUTPUT_DIR="$PROJECT_DIR/output"
SCRIPT="$PROJECT_DIR/satellite_scheduling/main.py"
ALGORITHMS_JSON="$PROJECT_DIR/satellite_scheduling/cosp_algorithm_configs.json"
SCENARIOS_DIR="$PROJECT_DIR/satellite_scheduling/scenarios_larger"


# Auto-detect a Python interpreter that has ortools installed.
# Override with --python /path/to/python if needed.
_find_python() {
  for candidate in \
      "$(dirname "$0")/venv/bin/python3" \
      "${VIRTUAL_ENV:-__none__}/bin/python3" \
      python3 python; do
    [[ "$candidate" == __none__* ]] && continue
    if command -v "$candidate" &>/dev/null && \
       "$candidate" -c "import ortools" 2>/dev/null; then
      echo "$candidate"
      return
    fi
  done
  echo ""
}
PYTHON=""

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-iterations)
      MAX_ITER="$2"; shift 2;;
    --output-dir)
      OUTPUT_DIR="$2"; shift 2;;
    --trials)
      TRIALS="$2"; shift 2;;
    --scenarios)
      SCENARIOS_DIR="$2"; shift 2;;
    --python)
      PYTHON="$2"; shift 2;;   # explicit override skips auto-detect
    *)
      echo "Unknown argument: $1" >&2; exit 1;;
  esac
done

# Resolve Python interpreter (auto-detect if not set via --python)
if [[ -z "${PYTHON}" ]]; then
  PYTHON="$(_find_python)"
fi
if [[ -z "${PYTHON}" ]]; then
  echo "ERROR: Could not find a Python interpreter with ortools installed." >&2
  echo "       Install ortools in your active environment or pass --python /path/to/python3" >&2
  exit 1
fi
echo "Using Python: ${PYTHON}"
echo ""

FRAMEWORKS=("iterative_pricing" "constraint_generation")

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
    current_trial=0
    while [[ $current_trial -lt $TRIALS ]]; do
      output_json="${out_dir}/${scenario_stem}_t${current_trial}.json"

      total=$((total + 1))
      echo "--------------------------------------------------------------"
      echo "  framework : ${framework}"
      echo "  scenario  : ${scenario_path}  (trial ${current_trial})"
      echo "  output    : ${output_json}"
      echo "--------------------------------------------------------------"

      # Remove stale output so main.py can write fresh results
      rm -f "${output_json}"

      if [[ USE_SLURM_JOBS ]]
      then
        echo "sending job to slurm"
        sbatch "$PROJECT_DIR/slurm_template.sh" "python" "${SCRIPT}" \
                                           --scenario "${scenario_path}" \
                                           --output_json "${output_json}" \
                                           --algorithms_json "${ALGORITHMS_JSON}" \
                                           --framework "${framework}" \
                                           --max_iterations "${MAX_ITER}"
      else
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
      fi
      echo ""
      current_trial=$(($current_trial+1))
    done
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
