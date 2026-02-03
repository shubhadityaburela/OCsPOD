#!/bin/bash

type_of_problem="Shifting"
grid_str="3201 2400 1"

num_control_sets=(
  "20"
)

for nc in "${num_control_sets[@]}"; do
  read -r n_c <<< "$nc"
  echo "Submitting modes: type_of_problem=$type_of_problem, num_controls=($n_c), grid=\"$grid_str\""
  sbatch run_cluster_inner_FOM.sh "$type_of_problem" "$n_c" "$grid_str"
done
