#!/bin/bash

source ~/.bashrc
conda activate big-dcop

BIG_DCOP_DIR="$( dirname -- "$( dirname -- "$( readlink -f -- "${BASH_SOURCE[0]}"; )"; )"; )"
cd $BIG_DCOP_DIR
python satellite_scheduling_pydcop/main.py \
	--scenario satellite_scheduling_pydcop/scenarios/scenario_$1.json \
	--output_json output/sat_sched/output_rm_scen_$1.json \
	--algorithms_json satellite_scheduling_pydcop/rm_algorithm_configs.json \
	--pydcop_mode thread \
	--framework iterative_pricing \
	--timeout 4000

python satellite_scheduling_pydcop/main.py \
  --scenario satellite_scheduling_pydcop/scenarios/scenario_$1.json \
  --output_json output/sat_sched/output_baseline_scen_$1.json \
  --algorithms_json satellite_scheduling_pydcop/baseline_algorithm_configs.json \
  --pydcop_mode thread \
	--framework iterative_pricing \
  --timeout 4000

