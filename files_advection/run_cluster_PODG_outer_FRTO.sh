#!/bin/bash

type_of_problem="Shifting"
grid_str="3201 2400 1"

mode_sets=(
  "500"
  "750"
  "1000"
  "1250"
  "1500"
  "1750"
  "2000"
)

for common_basis in False; do
  for ms in "${mode_sets[@]}"; do
    read -r mode1 <<< "$ms"
    echo "Submitting modes: type_of_problem=$type_of_problem, common_basis=$common_basis, modes=($mode1), grid=\"$grid_str\""
    sbatch run_cluster_PODG_inner_FRTO.sh "$type_of_problem" "$common_basis" "$mode1" "$grid_str"
  done
done




