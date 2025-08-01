#!/bin/bash

mode_sets=(
  "10 10"
  "50 50"
  "100 100"
  "150 150"
  "200 200"
  "250 250"
  "300 300"
)


for script_type in fixed adaptive; do
  for common_basis in True False; do
    for ms in "${mode_sets[@]}"; do
      # unpack the four modes
      read -r mode1 mode2 <<< "$ms"

      echo "Submitting job: script_type=$script_type, common_basis=$common_basis, modes=($mode1,$mode2)"
      sbatch run_cluster_PODG_inner_FRTO.sh \
        "$common_basis" \
        "modes" \
        "$mode1" "$mode2" \
        "$script_type"
    done
  done
done
