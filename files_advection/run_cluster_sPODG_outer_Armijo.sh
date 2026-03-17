#!/bin/bash

type_of_problem="Shifting"
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

for ms in "${mode_sets[@]}"; do
  read -r mode1 <<< "$ms"
  echo "Submitting modes: type_of_problem=$type_of_problem, modes=($mode1), grid=\"$grid_str\""
  sbatch run_cluster_sPODG_inner_Armijo.sh "$type_of_problem" "modes" "$mode1" "$grid_str"
done





