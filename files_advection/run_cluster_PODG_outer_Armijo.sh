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

for ms in "${mode_sets[@]}"; do
  read -r mode1 <<< "$ms"
  echo "Submitting modes: type_of_problem=$type_of_problem, modes=($mode1), grid=\"$grid_str\""
  sbatch run_cluster_PODG_inner_Armijo.sh "$type_of_problem" "modes" "$mode1" "$grid_str"
done





