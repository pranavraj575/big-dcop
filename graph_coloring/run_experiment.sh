#!/bin/bash

PROJECT_DIR="$( dirname -- "$( dirname -- "$( readlink -f -- "${BASH_SOURCE[0]}"; )"; )"; )"

SCRIPT="$PROJECT_DIR/graph_coloring/graph_coloring_runner.py"
OUTPUT_DIR="$PROJECT_DIR/output"
GRAPH_INSTANCE_DIR="$OUTPUT_DIR/graph_coloring_instances_hard"
ALGORITHMS="$PROJECT_DIR/graph_coloring/configs/algorithm_configs.json"
TRIALS=2
START_TRIAL=0
OVERWRITE=false
USE_SLURM_JOBS=false

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="$2"; shift 2;;
    --algorithms)
      ALGORITHMS="$2"; shift 2;;
    --graph-instances)
      GRAPH_INSTANCE_DIR="$2"; shift 2;;
    --trials)
      TRIALS="$2"; shift 2;;
    --start-trial)
      START_TRIAL="$2"; shift 2;;
    --slurm)
      USE_SLURM_JOBS=true; shift 1;;
    --overwrite)
      OVERWRITE=true; shift 1;;
    *)
      echo "Unknown argument: $1" >&2; exit 1;;
  esac
done

source ~/.bashrc
source ~/miniconda3/bin/activate
conda activate big-dcop

#python graph_coloring/graph_coloring_generator.py \
#  --output_dir "${GRAPH_INSTANCE_DIR}" \
# --num_problems 5 \
#  --graph_n 10 20 30 50 100 \
#  --color_count 3 \
#  --graph_type random scalefree
total=0
passed=0
skipped=0
slurmed=0

counter=0
for instance_path in "${GRAPH_INSTANCE_DIR}"/*.yaml; do
  current_trial="${START_TRIAL}"
  while [[ $current_trial -lt $(($TRIALS + $START_TRIAL)) ]]; do
    output_csv="${OUTPUT_DIR}/graph_coloring_results_i_${counter}_t_${current_trial}.csv"
    echo "--------------------------------------------------------------"
    echo "  instance  : ${instance_path}  (trial ${current_trial})"
    echo "--------------------------------------------------------------"
    total=$((total + 1))

    if [[ -e "$output_csv" ]] && [[ $OVERWRITE == false ]]; then
      echo "WARNING: $output_csv exists, skipping this trial (use --overwrite to overwrite this)"
      skipped=$((skipped + 1))
    else
      if [[ -e "$output_csv" ]]; then
        echo "WARNING: $output_csv exists, python script will overwrite this"
      fi
      if [[ $USE_SLURM_JOBS == true ]]; then
        echo "sending job to slurm"
        sbatch "$PROJECT_DIR/slurm_template.sh" "python" "${SCRIPT}" \
          --algorithms "${ALGORITHMS}" \
          --collect_on value_change \
          --instances "${instance_path}" \
          --output_csv "${output_csv}"
        slurmed=$((slurmed + 1))
      else
        python "${SCRIPT}" \
            --algorithms "${ALGORITHMS}" \
            --trials "${TRIALS}" \
            --collect_on value_change \
            --instances "${instance_path}" \
            --output_csv "${output_csv}"
        passed=$((passed + 1))
      fi
    fi
    current_trial=$(($current_trial+1))
  done
  counter=$((counter + 1))
done


echo "============================================================"
echo "  Experiments complete"
echo "  Total   : ${total}"
echo "  Passed  : ${passed}"
echo "  Skipped : ${skipped}"
echo "  Slurmed : ${slurmed}"
if [[ ${failed} -gt 0 ]]; then
  echo "  Failed runs:"
  for f in "${failed_list[@]}"; do
    echo "    - ${f}"
  done
fi
echo "============================================================"