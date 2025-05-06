#!/bin/bash

# Loop over the two script types: "adaptive" and "fixed"
for script_type in adaptive fixed; do
    for problem in 1 2 3; do
        for conv_accel in True False; do
            for target_for_basis in False True; do
                for interp_scheme in "CubSpl" "Lagr"; do
                    # Define an array of mode pairs (each pair represents --modes value1 value2)
                    mode_values=( "5 155" "10 160" "15 165" "20 170" "25 175" "30 180" "35 185" "40 190" "45 195" "50 200" "55 205" "60 210")

                    if [ "$script_type" = "adaptive" ]; then
                        # For adaptive, also loop over refine_acc_cost values
                        for refine_acc_cost in True False; do
                            # Submit jobs using --modes values
#                            for mode_pair in "${mode_values[@]}"; do
#                                echo "Submitting job: script_type=$script_type, problem=$problem, convergence_acceleration=$conv_accel, include_target_for_basis=$target_for_basis, interpolation_scheme=$interp_scheme, type_of_study=modes, values=($mode_pair), refine_acc_cost=$refine_acc_cost"
#                                sbatch run_cluster_sPODG_PODG_inner_FOTR.sh $problem $conv_accel $target_for_basis $interp_scheme modes $mode_pair $script_type $refine_acc_cost
#                            done
                            # Submit jobs using --tol values
                            for tol in 1e-2 7e-3 5e-3 3e-3 1e-3 7e-4 5e-4 3e-4 1e-4; do
                                echo "Submitting job: script_type=$script_type, problem=$problem, convergence_acceleration=$conv_accel, include_target_for_basis=$target_for_basis, interpolation_scheme=$interp_scheme, type_of_study=tol, value=$tol, refine_acc_cost=$refine_acc_cost"
                                sbatch run_cluster_sPODG_PODG_inner_FOTR.sh $problem $conv_accel $target_for_basis $interp_scheme tol $tol $script_type $refine_acc_cost
                            done
                        done
                    else
                        # For fixed, do not include the refine_acc_cost parameter
#                        for mode_pair in "${mode_values[@]}"; do
#                            echo "Submitting job: script_type=$script_type, problem=$problem, convergence_acceleration=$conv_accel, include_target_for_basis=$target_for_basis, interpolation_scheme=$interp_scheme, type_of_study=modes, values=($mode_pair)"
#                            sbatch run_cluster_sPODG_PODG_inner_FOTR.sh $problem $conv_accel $target_for_basis $interp_scheme modes $mode_pair $script_type
#                        done
                        # Submit jobs using --tol values
                        for tol in 1e-2 7e-3 5e-3 3e-3 1e-3 7e-4 5e-4 3e-4 1e-4; do
                            echo "Submitting job: script_type=$script_type, problem=$problem, convergence_acceleration=$conv_accel, include_target_for_basis=$target_for_basis, interpolation_scheme=$interp_scheme, type_of_study=tol, value=$tol"
                            sbatch run_cluster_sPODG_PODG_inner_FOTR.sh $problem $conv_accel $target_for_basis $interp_scheme tol $tol $script_type
                        done
                    fi

                done
            done
        done
    done
done



