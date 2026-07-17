
# bash graph_coloring/run_experiment.sh --slurm --output-dir output/graph_col_final --algorithms graph_coloring/configs/algorithm_configs.json --trials 10
# bash graph_coloring/run_experiment.sh --slurm --output-dir output/graph_col_final_hyperparams --algorithms graph_coloring/configs/hyperparam_configs.json --trials 10

for algo_path in graph_coloring/configs/hyperparam_configs_*; do
  nm=${algo_path:42:-5}
  echo "plotting" $nm
  python graph_coloring/plot_results.py \
    --path output/graph_col_final_hyperparams/* \
    --algorithms "${algo_path}" \
    --y_keys cost \
    --output output/graph_col_plot_hp/"${nm}";
done

echo "plotting main results"
python graph_coloring/plot_results.py --path output/graph_col_final/* --algorithms graph_coloring/configs/algorithm_configs_to_plot.json --y_keys cost msg_count msg_count:log --output output/graph_coloring
