
# bash satellite_scheduling/run_experiments.sh  --output-dir output/cosp_main --trials 20 --max-iterations 100 --slurm
# bash satellite_scheduling/run_experiments.sh  --output-dir output/cosp_c_tuning/c_<C VAL STRING> --trials 20 --max-iterations 100 --slurm --step-size-c <C VALUE>

MAX_ITER=25
TEST_MAX_ITER=100
MAIN_EXP=output/cosp_main
MAIN_TEMP="${MAIN_EXP}_no_iter_TEMP"
HP_OPTIM_EXP=output/cosp_c_tuning
HP_OPTIM_TEMP="${HP_OPTIM_EXP}_TEMP"

rm -r ${HP_OPTIM_TEMP}
rm -r ${MAIN_TEMP}

cp -r ${MAIN_EXP} ${MAIN_TEMP}
rm -r "${MAIN_TEMP}/iterative_pricing"
cp -r ${HP_OPTIM_EXP} ${HP_OPTIM_TEMP}
rm -r "${HP_OPTIM_TEMP}/c_0" "${HP_OPTIM_TEMP}/c_100"

echo "PLOTTING main experiment"
python satellite_scheduling/analyze_run_experiments.py --hyperparam-optim --output ${HP_OPTIM_TEMP}/* ${MAIN_TEMP} --plot output/cosp_plots/context --algorithms satellite_scheduling/algo_configs/plot_context_based_algs.json --max-iteration "$MAX_ITER" &
python satellite_scheduling/analyze_run_experiments.py --hyperparam-optim --output ${HP_OPTIM_TEMP}/* ${MAIN_TEMP} --plot output/cosp_plots/no_context --algorithms satellite_scheduling/algo_configs/plot_default_algs.json --max-iteration "$MAX_ITER" &
python satellite_scheduling/analyze_run_experiments.py --hyperparam-optim --output ${HP_OPTIM_TEMP}/* ${MAIN_TEMP} --plot output/cosp_plots/all --max-iteration "$MAX_ITER" &
python satellite_scheduling/analyze_run_experiments.py --hyperparam-optim --output ${HP_OPTIM_TEMP}/* ${MAIN_TEMP} --plot output/cosp_plots/ctx_cmp  --algorithms satellite_scheduling/algo_configs/ctx_comparison.json --max-iteration "$MAX_ITER" &
python satellite_scheduling/analyze_run_experiments.py --hyperparam-optim --output ${HP_OPTIM_TEMP}/* ${MAIN_TEMP} --plot output/cosp_plots/ONLY_ctx  --algorithms satellite_scheduling/algo_configs/ONLY_ctx_based.json --max-iteration "$MAX_ITER" &
python satellite_scheduling/analyze_run_experiments.py --hyperparam-optim --output ${HP_OPTIM_TEMP}/* ${MAIN_TEMP} --plot output/cosp_plots/"$TEST_MAX_ITER"_max_iter --algorithms satellite_scheduling/algo_configs/plot_context_based_algs.json --max-iteration "$TEST_MAX_ITER"


echo "PLOTTING all c"
python satellite_scheduling/analyze_run_experiments.py --output ${HP_OPTIM_TEMP}/* --plot output/cosp_plots_c_tuning/all_c/no_context --algorithms satellite_scheduling/algo_configs/plot_default_algs.json --max-iteration "$MAX_ITER" &
python satellite_scheduling/analyze_run_experiments.py --output ${HP_OPTIM_TEMP}/* --plot output/cosp_plots_c_tuning/all_c/context --algorithms satellite_scheduling/algo_configs/plot_context_based_algs.json --max-iteration "$MAX_ITER"

echo "PLOTTING c tuning in parallel"
for output_path in ${HP_OPTIM_TEMP}/c_*; do
  c_var=${output_path:$(( ${#HP_OPTIM_TEMP} + 3 )):${#output_path}}
  echo "plotting c=${c_var}" &
  python satellite_scheduling/analyze_run_experiments.py --output "${output_path}" --plot output/cosp_plots_c_tuning/c_"${c_var}"/all --max-iteration "$MAX_ITER" &
  python satellite_scheduling/analyze_run_experiments.py --output "${output_path}" --plot output/cosp_plots_c_tuning/c_"${c_var}"/no_context --algorithms satellite_scheduling/algo_configs/plot_default_algs.json --max-iteration "$MAX_ITER" &
  python satellite_scheduling/analyze_run_experiments.py --output "${output_path}" --plot output/cosp_plots_c_tuning/c_"${c_var}"/context --algorithms satellite_scheduling/algo_configs/plot_context_based_algs.json --max-iteration "$MAX_ITER" --show-numbers &
done
echo "running";

