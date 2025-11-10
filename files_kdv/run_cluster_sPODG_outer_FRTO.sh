#!/bin/bash

fully_nonlinear="True"   # e.g. True for nonlinear problem and False for fully linear
CTC_mask="False"            # "True" or "False"
grid_str="1000 120000 16"

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

for script_type in fixed adaptive; do
  for common_basis in True False; do
    for ms in "${mode_sets[@]}"; do
      read -r mode1 <<< "$ms"
      echo "Submitting modes: fully_nonlinear=$fully_nonlinear, script_type=$script_type, common_basis=$common_basis, CTC_mask=$CTC_mask modes=($mode1), grid=\"$grid_str\""
      sbatch run_cluster_sPODG_inner_FRTO.sh "$fully_nonlinear" "$common_basis" "modes" "$CTC_mask" "$mode1" "$script_type" "$grid_str"
    done
  done
done



