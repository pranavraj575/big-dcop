BIG_DCOP_DIR="$( dirname -- "$( dirname -- "$( readlink -f -- "${BASH_SOURCE[0]}"; )"; )"; )"
cd $BIG_DCOP_DIR
conda activate big-dcop

python satellite_scheduling_pydcop/main.py \
	--scenario satellite_scheduling_pydcop/scenarios/scenario_$1.json \
	--output_json output/sat_sched/output_cg_rm_scen_$1.json \
	--algorithms_json satellite_scheduling_pydcop/rm_algorithm_configs.json \
	--pydcop_mode thread \
	--framework constraint_generation \
	--timeout 4000

python satellite_scheduling_pydcop/main.py \
  --scenario satellite_scheduling_pydcop/scenarios/scenario_$1.json \
  --output_json output/sat_sched/output_cg_baseline_scen_$1.json \
  --algorithms_json satellite_scheduling_pydcop/baseline_algorithm_configs.json \
  --pydcop_mode thread \
	--framework constraint_generation \
  --timeout 4000

