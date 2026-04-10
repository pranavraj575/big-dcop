cd ~/projects/big-dcop
conda activate big-dcop

python satellite_scheduling/main.py \
	--scenario satellite_scheduling/scenarios/scenario_$1.json \
	--output_json output/sat_sched/output_rm_scen_$1.json \
	--algorithms_json satellite_scheduling/rm_algorithm_configs.json \
	--pydcop_mode thread \
	--timeout 4000

python satellite_scheduling/main.py \
        --scenario satellite_scheduling/scenarios/scenario_$1.json \
        --output_json output/sat_sched/output_baseline_scen_$1.json \
        --algorithms_json satellite_scheduling/baseline_algorithm_configs.json \
        --pydcop_mode thread \
        --timeout 4000

