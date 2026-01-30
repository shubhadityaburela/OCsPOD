#!/bin/bash

type_of_problem="Shifting"
grid_str="3201 2400 1"

mode_sets=(
  "5"
  "10"
  "20"
  "30"
  "40"
  "50"
  "60"
  "70"
  "80"
  "90"
  "100"
  "200"
  "300"
  "400"
  "500"
)

tolerances=(
  "1e-2"
  "5e-3"
  "1e-3"
  "5e-4"
  "1e-4"
  "5e-5"
  "1e-5"
  "5e-6"
  "1e-6"
  "5e-7"
  "1e-7"
  "5e-8"
  "1e-8"
  "5e-9"
  "1e-9"
  "5e-10"
  "1e-10"
)


for common_basis in False; do
  for ms in "${mode_sets[@]}"; do
    read -r mode1 <<< "$ms"
    echo "Submitting modes: type_of_problem=$type_of_problem, common_basis=$common_basis, modes=($mode1), grid=\"$grid_str\""
    sbatch run_cluster_PODG_inner_FRTO_a.sh "$type_of_problem" "$common_basis" "modes" "$mode1" "$grid_str"
  done
  for tol in "${tolerances[@]}"; do
    echo "Submitting tol: type_of_problem=$type_of_problem, common_basis=$common_basis, tol=$tol, grid=\"$grid_str\""
    sbatch run_cluster_PODG_inner_FRTO_a.sh "$type_of_problem" "$common_basis" "tol" "$tol" "$grid_str"
  done
done




