#!/bin/bash

type_of_problem="Constant_shift"   # "Constant_shift" => mode study; anything else => tol study
CTC_mask="False"                   # used only for tol study
grid_str="1000 8000 1"             # 1000 8000 1   or     3200 3360 1

mode_sets=(
  "10"
  "50"
  "100"
  "150"
  "200"
  "250"
  "300"
  "350"
  "400"
  "450"
  "500"
)

tolerances=(
  "1e-2" "5e-3" "1e-3" "5e-4" "1e-4"
  "5e-5" "1e-5" "5e-6" "1e-6" "5e-7" "1e-7"
)

for script_type in fixed adaptive; do
  for common_basis in True False; do

    if [ "$type_of_problem" = "Constant_shift" ]; then
      for ms in "${mode_sets[@]}"; do
        read -r mode1 <<< "$ms"
        echo "Submitting modes: type_of_problem=$type_of_problem, script_type=$script_type, common_basis=$common_basis, CTC_mask=$CTC_mask modes=($mode1), grid=\"$grid_str\""
        sbatch run_cluster_PODG_inner_FRTO.sh "$type_of_problem" "$common_basis" "modes" "$CTC_mask" "$mode1" "$script_type" "$grid_str"
      done
    else
      for tol in "${tolerances[@]}"; do
        echo "Submitting tol: type_of_problem=$type_of_problem, script_type=$script_type, common_basis=$common_basis, CTC_mask=$CTC_mask, tol=$tol, grid=\"$grid_str\""
        sbatch run_cluster_PODG_inner_FRTO.sh "$type_of_problem" "$common_basis" "tol" "$CTC_mask" "$tol" "$script_type" "$grid_str"
      done
    fi

  done
done



