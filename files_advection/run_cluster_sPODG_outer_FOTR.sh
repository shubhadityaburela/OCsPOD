#!/bin/bash

type_of_problem="Shifting"
grid_str="3201 2400 1"

mode_sets=(
  "2 2"
  "5 5"
  "8 8"
  "10 10"
  "12 12"
  "15 15"
  "20 20"
  "25 25"
  "30 30"
  "35 35"
  "40 40"
  "45 45"
  "50 50"
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

for script_type in fixed adaptive; do
  for common_basis in True False; do
    for ms in "${mode_sets[@]}"; do
      read -r mode1 mode2 <<< "$ms"
      echo "Submitting modes: script_type=$script_type, type_of_problem=$type_of_problem, common_basis=$common_basis, modes=($mode1,$mode2), grid=\"$grid_str\""
      sbatch run_cluster_sPODG_inner_FOTR.sh "$type_of_problem" "$common_basis" "modes" "$mode1" "$mode2" "$script_type" "$grid_str"
    done
    for tol in "${tolerances[@]}"; do
      echo "Submitting tol: script_type=$script_type, type_of_problem=$type_of_problem, common_basis=$common_basis, tol=$tol, grid=\"$grid_str\""
      sbatch run_cluster_sPODG_inner_FOTR.sh "$type_of_problem" "$common_basis" "tol" "$tol" "$script_type" "$grid_str"
    done
  done
done


