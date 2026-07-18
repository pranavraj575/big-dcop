
# C_VALS=(0 0.01 0.05 0.1 0.15 0.3 0.5 0.75 1 1.25 1.5 1.6 1.7 1.75 1.8 1.9 2 2.25 2.5 3 4 5 6 6.9 8.5 10 15 25 50 100)
C_VALS=(0 0.01 0.03 0.05 0.1 0.3 0.5 0.75 1 1.5 1.75 2 2.5 3 4 5 6 6.9 8.5 10 15 25 50 100)

for C_VAL in "${C_VALS[@]}"; do
  C_STRING="${C_VAL//./_}"
  bash satellite_scheduling/run_experiments.sh \
      --no-cg \
      --output-dir output/cosp_c_tuning/c_"${C_STRING}" \
      --trials 20 \
      --max-iterations 100 \
      --slurm \
      --step-size-c "${C_VAL}"
done
