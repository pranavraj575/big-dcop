#!/bin/bash

source ~/.bashrc
source ~/miniconda3/bin/activate
conda activate big-dcop

PROJECT_DIR="$( dirname -- "$( dirname -- "$( readlink -f -- "${BASH_SOURCE[0]}"; )"; )"; )"

SCRIPT="$PROJECT_DIR/graph_coloring/graph_coloring_runner.py"
OUTPUT_DIR="$PROJECT_DIR/output"
GRAPH_INSTANCE_DIR="$OUTPUT_DIR/graph_coloring_instances_hard"
ALGORITHMS="$PROJECT_DIR/graph_coloring/configs/algorithm_configs.json"
TRIALS=2

#python graph_coloring/graph_coloring_generator.py \
#  --output_dir "${GRAPH_INSTANCE_DIR}" \
# --num_problems 5 \
#  --graph_n 10 20 30 50 100 \
#  --color_count 3 \
#  --graph_type random scalefree

counter=0
for instance_path in "${GRAPH_INSTANCE_DIR}"/*.yaml; do
  python "${SCRIPT}" \
    --algorithms "${ALGORITHMS}" \
    --trials "${TRIALS}" \
    --collect_on value_change \
    --instances "${instance_path}" \
    --output_csv "${OUTPUT_DIR}/graph_coloring_results_${counter}.csv"

  counter=$((counter + 1))
done


