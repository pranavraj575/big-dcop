BIG_DCOP_DIR="$( dirname -- "$( dirname -- "$( readlink -f -- "${BASH_SOURCE[0]}"; )"; )"; )"
cd $BIG_DCOP_DIR
conda activate big-dcop

python satellite_scheduling/main.py \
	--scenario satellite_scheduling/scenarios/scenario_$1.json \
	--output_json output/sat_sched/output_cg_rm_scen_$1.json \
	--algorithms_json satellite_scheduling/rm_algorithm_configs.json \
	--pydcop_mode thread \
	--framework constraint_generation \
	--timeout 4000

python satellite_scheduling/main.py \
  --scenario satellite_scheduling/scenarios/scenario_$1.json \
  --output_json output/sat_sched/output_cg_baseline_scen_$1.json \
  --algorithms_json satellite_scheduling/baseline_algorithm_configs.json \
  --pydcop_mode thread \
	--framework constraint_generation \
  --timeout 4000

