#!/bin/bash
mode_sets=(
  "2 2 2 2"
  "5 5 5 5"
  "8 8 8 8"
  "10 10 10 10"
  "12 12 12 12"
  "15 15 15 15"
)

for script_type in fixed adaptive; do
  for common_basis in True False; do
    for ms in "${mode_sets[@]}"; do
      # unpack the four modes
      read -r mode1 mode2 mode3 mode4 <<< "$ms"

      echo "Submitting job: script_type=$script_type, common_basis=$common_basis, modes=($mode1,$mode2,$mode3,$mode4)"
      sbatch run_cluster_sPODG_inner_FOTR.sh \
        "$common_basis" \
        "modes" \
        "$mode1" "$mode2" "$mode3" "$mode4" \
        "$script_type"
    done
  done
done




