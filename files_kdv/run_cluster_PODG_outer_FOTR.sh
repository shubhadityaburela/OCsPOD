#!/bin/bash

fully_nonlinear="True"   # e.g. True for nonlinear problem and False for fully linear
CTC_mask="False"            # "True" or "False"
grid_str="1000 120000 16"

mode_sets=(
  "10 10"
  "50 50"
  "100 100"
  "150 150"
  "200 200"
  "250 250"
  "300 300"
  "350 350"
  "400 400"
  "450 450"
  "500 500"
)

for script_type in fixed adaptive; do
  for common_basis in True False; do
    for ms in "${mode_sets[@]}"; do
      read -r mode1 mode2 <<< "$ms"
      echo "Submitting modes: fully_nonlinear=$fully_nonlinear, script_type=$script_type, common_basis=$common_basis, CTC_mask=$CTC_mask modes=($mode1,$mode2), grid=\"$grid_str\""
      sbatch run_cluster_PODG_inner_FOTR.sh "$fully_nonlinear" "$common_basis" "modes" "$CTC_mask" "$mode1" "$mode2" "$script_type" "$grid_str"
    done
  done
done



