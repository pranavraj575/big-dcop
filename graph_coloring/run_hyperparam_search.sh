#!/bin/bash

source ~/.bashrc
source ~/miniconda3/bin/activate
conda activate big-dcop

PROJECT_DIR="$( dirname -- "$( dirname -- "$( readlink -f -- "${BASH_SOURCE[0]}"; )"; )"; )"

SCRIPT="$PROJECT_DIR/graph_coloring/graph_coloring_runner.py"
OUTPUT_DIR="$PROJECT_DIR/output"
GRAPH_INSTANCE_DIR="$OUTPUT_DIR/graph_coloring_instances_hard"
ALGORITHMS="$PROJECT_DIR/graph_coloring/configs/hyperparam_configs.json"
TRIALS=3
OUTPUT_CSV="$OUTPUT_DIR/hyperparam_results.csv"

INSTANCES=(
  "${GRAPH_INSTANCE_DIR}/gc_n10_k3_random_1.yaml"
  "${GRAPH_INSTANCE_DIR}/gc_n20_k3_random_1.yaml"
)

python "${SCRIPT}" \
  --algorithms "${ALGORITHMS}" \
  --trials "${TRIALS}" \
  --collect_on value_change \
  --instances "${INSTANCES[@]}" \
  --output_csv "${OUTPUT_CSV}"
