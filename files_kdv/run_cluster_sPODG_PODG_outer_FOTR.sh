#!/bin/bash
mode_sets=(
  "2 2 500 2"
  "5 5 500 5"
  "8 8 500 8"
  "10 10 500 10"
  "12 12 500 12"
  "15 15 500 15"
  "20 20 500 20"
  "25 25 500 25"
  "30 30 500 30"
  "35 35 500 35"
  "40 40 500 40"
  "45 45 500 45"
  "50 50 500 50"
)

for script_type in fixed adaptive; do
  for common_basis in True False; do
    for ms in "${mode_sets[@]}"; do
      # unpack the four modes
      read -r mode1 mode2 mode3 mode4 <<< "$ms"

      echo "Submitting job: script_type=$script_type, common_basis=$common_basis, modes=($mode1,$mode2,$mode3,$mode4)"
      sbatch run_cluster_sPODG_PODG_inner_FOTR.sh \
        "$common_basis" \
        "modes" \
        "$mode1" "$mode2" "$mode3" "$mode4" \
        "$script_type"
    done
  done
done




