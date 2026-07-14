
# bash run_experiments.sh  --output-dir output/cosp_final_100_iter --trials 20 --max-iterations 100 --slurm

# bash run_experiments.sh  --output-dir output/cosp_final_c_tuning/c_<C VAL STRING> --trials 20 --max-iterations 100 --slurm --step-size-c <C VALUE>

echo "plotting main experiment"
python satellite_scheduling/analyze_run_experiments.py --hyperparam-optim --output output/cosp_final_c_tuning/* output/cosp_final_100_iter_no_iterative --plot output/cosp_solver_plots_final/context --algorithms satellite_scheduling/algo_configs/plot_context_based_algs.json &
python satellite_scheduling/analyze_run_experiments.py --hyperparam-optim --output output/cosp_final_c_tuning/* output/cosp_final_100_iter_no_iterative --plot output/cosp_solver_plots_final/no_context --algorithms satellite_scheduling/algo_configs/plot_default_algs.json &
python satellite_scheduling/analyze_run_experiments.py --hyperparam-optim --output output/cosp_final_c_tuning/* output/cosp_final_100_iter_no_iterative --plot output/cosp_solver_plots_final/all &
python satellite_scheduling/analyze_run_experiments.py --hyperparam-optim --output output/cosp_final_c_tuning/* output/cosp_final_100_iter_no_iterative --plot output/cosp_solver_plots_final/ctx_cmp  --algorithms satellite_scheduling/algo_configs/ctx_comparison.json &
python satellite_scheduling/analyze_run_experiments.py --hyperparam-optim --output output/cosp_final_c_tuning/* output/cosp_final_100_iter_no_iterative --plot output/cosp_solver_plots_final/ONLY_ctx  --algorithms satellite_scheduling/algo_configs/ONLY_ctx_based.json




#python satellite_scheduling/analyze_run_experiments.py --output output/cosp_final_100_iter --plot output/cosp_solver_plots_final_100_iter/same_scale --max-iteration 100 &
#python satellite_scheduling/analyze_run_experiments.py --output output/cosp_final_100_iter --plot output/cosp_solver_plots_final_100_iter/diff_scale --not-same --max-iteration 100 &
#python satellite_scheduling/analyze_run_experiments.py --output output/cosp_final_100_iter --plot output/cosp_solver_plots_final_100_iter/no_context --not-same --max-iteration 100 --algorithms satellite_scheduling/algo_configs/plot_default_algs.json &
#python satellite_scheduling/analyze_run_experiments.py --output output/cosp_final_100_iter --plot output/cosp_solver_plots_final_100_iter/context --not-same --max-iteration 100 --algorithms satellite_scheduling/algo_configs/plot_context_based_algs.json

#echo "plotting all c"
#python satellite_scheduling/analyze_run_experiments.py --output output/cosp_final_c_tuning/* --plot output/cosp_solver_plots_c_tuning/all_c/no_context --algorithms satellite_scheduling/algo_configs/plot_default_algs.json
#python satellite_scheduling/analyze_run_experiments.py --output output/cosp_final_c_tuning/* --plot output/cosp_solver_plots_c_tuning/all_c/context --algorithms satellite_scheduling/algo_configs/plot_context_based_algs.json

#echo "plotting c tuning in parallel"
#for output_path in output/cosp_final_c_tuning/c_1_*; do
#  c_var=$(cut -c 30-100 <<< $output_path)
#  echo "plotting c=${c_var}" &
#  python satellite_scheduling/analyze_run_experiments.py --output "${output_path}" --plot output/cosp_solver_plots_c_tuning/c_"${c_var}"&
#done
