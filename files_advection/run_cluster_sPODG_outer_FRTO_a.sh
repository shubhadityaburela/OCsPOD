#!/bin/bash

type_of_problem="Shifting"   # Shifting or Shifting_3
grid_str="3201 2400 1"

mode_sets=(
  "2"
  "5"
  "8"
  "10"
  "12"
  "15"
  "20"
  "25"
  "30"
  "35"
  "40"
  "45"
  "50"
)

tolerances=(
  "1e-2" "5e-3" "1e-3" "5e-4" "1e-4"
  "5e-5" "1e-5" "5e-6" "1e-6"
)

for common_basis in False; do
  for ms in "${mode_sets[@]}"; do
    read -r mode1 <<< "$ms"
    echo "Submitting modes: type_of_problem=$type_of_problem, common_basis=$common_basis, modes=($mode1), grid=\"$grid_str\""
    sbatch run_cluster_sPODG_inner_FRTO_a.sh "$type_of_problem" "$common_basis" "modes" "$mode1" "$grid_str"
  done
  for tol in "${tolerances[@]}"; do
    echo "Submitting tol: type_of_problem=$type_of_problem, common_basis=$common_basis, tol=$tol, grid=\"$grid_str\""
    sbatch run_cluster_sPODG_inner_FRTO_a.sh "$type_of_problem" "$common_basis" "tol" "$tol" "$grid_str"
  done
done