
# bash run_experiments.sh  --output-dir output/cosp_final_100_iter --trials 20 --max-iterations 100 --slurm
# bash run_experiments.sh  --output-dir output/cosp_final_c_tuning/c_<C VAL STRING> --trials 20 --max-iterations 100 --slurm --step-size-c <C VALUE>

MAX_ITER=100
SMALL_MAX_ITER=25

rm -r output/cosp_final_100_iter_no_iterative/
cp -r output/cosp_final_100_iter output/cosp_final_100_iter_no_iterative
rm -r output/cosp_final_100_iter_no_iterative/iterative_pricing

echo "PLOTTING main experiment"
python satellite_scheduling/analyze_run_experiments.py --hyperparam-optim --output output/cosp_final_c_tuning/* output/cosp_final_100_iter_no_iterative --plot output/cosp_solver_plots_final/context --algorithms satellite_scheduling/algo_configs/plot_context_based_algs.json --max-iteration "$MAX_ITER" &
python satellite_scheduling/analyze_run_experiments.py --hyperparam-optim --output output/cosp_final_c_tuning/* output/cosp_final_100_iter_no_iterative --plot output/cosp_solver_plots_final/no_context --algorithms satellite_scheduling/algo_configs/plot_default_algs.json --max-iteration "$MAX_ITER" &
python satellite_scheduling/analyze_run_experiments.py --hyperparam-optim --output output/cosp_final_c_tuning/* output/cosp_final_100_iter_no_iterative --plot output/cosp_solver_plots_final/all --max-iteration "$MAX_ITER" &
python satellite_scheduling/analyze_run_experiments.py --hyperparam-optim --output output/cosp_final_c_tuning/* output/cosp_final_100_iter_no_iterative --plot output/cosp_solver_plots_final/ctx_cmp  --algorithms satellite_scheduling/algo_configs/ctx_comparison.json --max-iteration "$MAX_ITER" &
python satellite_scheduling/analyze_run_experiments.py --hyperparam-optim --output output/cosp_final_c_tuning/* output/cosp_final_100_iter_no_iterative --plot output/cosp_solver_plots_final/ONLY_ctx  --algorithms satellite_scheduling/algo_configs/ONLY_ctx_based.json --max-iteration "$MAX_ITER" &

python satellite_scheduling/analyze_run_experiments.py --hyperparam-optim --output output/cosp_final_c_tuning/* output/cosp_final_100_iter_no_iterative --plot output/cosp_solver_plots_final/"$SMALL_MAX_ITER"_max_iter --algorithms satellite_scheduling/algo_configs/plot_context_based_algs.json --max-iteration "$SMALL_MAX_ITER"




#python satellite_scheduling/analyze_run_experiments.py --output output/cosp_final_100_iter --plot output/cosp_solver_plots_final_100_iter/same_scale --max-iteration 100 &
#python satellite_scheduling/analyze_run_experiments.py --output output/cosp_final_100_iter --plot output/cosp_solver_plots_final_100_iter/diff_scale --not-same --max-iteration 100 &
#python satellite_scheduling/analyze_run_experiments.py --output output/cosp_final_100_iter --plot output/cosp_solver_plots_final_100_iter/no_context --not-same --max-iteration 100 --algorithms satellite_scheduling/algo_configs/plot_default_algs.json &
#python satellite_scheduling/analyze_run_experiments.py --output output/cosp_final_100_iter --plot output/cosp_solver_plots_final_100_iter/context --not-same --max-iteration 100 --algorithms satellite_scheduling/algo_configs/plot_context_based_algs.json

echo "PLOTTING all c"
python satellite_scheduling/analyze_run_experiments.py --output output/cosp_final_c_tuning/* --plot output/cosp_solver_plots_c_tuning/all_c/no_context --algorithms satellite_scheduling/algo_configs/plot_default_algs.json --max-iteration "$MAX_ITER" &
python satellite_scheduling/analyze_run_experiments.py --output output/cosp_final_c_tuning/* --plot output/cosp_solver_plots_c_tuning/all_c/context --algorithms satellite_scheduling/algo_configs/plot_context_based_algs.json --max-iteration "$MAX_ITER"

echo "PLOTTING c tuning in parallel"
for output_path in output/cosp_final_c_tuning/c_*; do
  c_var=$(cut -c 30-100 <<< $output_path)
  echo "plotting c=${c_var}" &
  python satellite_scheduling/analyze_run_experiments.py --output "${output_path}" --plot output/cosp_solver_plots_c_tuning/c_"${c_var}"/all --max-iteration "$MAX_ITER" &
  python satellite_scheduling/analyze_run_experiments.py --output "${output_path}" --plot output/cosp_solver_plots_c_tuning/c_"${c_var}"/no_context --algorithms satellite_scheduling/algo_configs/plot_default_algs.json --max-iteration "$MAX_ITER" &
  python satellite_scheduling/analyze_run_experiments.py --output "${output_path}" --plot output/cosp_solver_plots_c_tuning/c_"${c_var}"/context --algorithms satellite_scheduling/algo_configs/plot_context_based_algs.json --max-iteration "$MAX_ITER" &
done
echo "running"
