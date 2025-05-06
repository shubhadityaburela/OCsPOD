#!/bin/bash
# Loop over the two script types: "adaptive" and "fixed"
for script_type in adaptive fixed everytime; do
    for problem in 1 2 3; do
        for conv_accel in True False; do
            for target_for_basis in False True; do
                # Define an array of mode pairs (each pair represents --modes value1 value2)
                mode_values=( "5 5" "20 20" "35 35" "50 50" "65 65" "90 90" "120 120" "150 150" "180 180")

                if [ "$script_type" = "adaptive" ]; then
                    # For adaptive, also loop over refine_acc_cost values
                    for refine_acc_cost in True False; do
                        # Submit jobs using --modes values
                        for mode_pair in "${mode_values[@]}"; do
                            echo "Submitting job: script_type=$script_type, problem=$problem, convergence_acceleration=$conv_accel, include_target_for_basis=$target_for_basis, type_of_study=modes, values=($mode_pair), refine_acc_cost=$refine_acc_cost"
                            sbatch run_cluster_PODG_inner_FOTR.sh $problem $conv_accel $target_for_basis modes $mode_pair $script_type $refine_acc_cost
                        done
                        # Submit jobs using --tol values
                        for tol in 1e-2 7e-3 3e-3 1e-3 7e-4 5e-4 3e-4 1e-4 5e-5 1e-5 5e-6 1e-6; do
                            echo "Submitting job: script_type=$script_type, problem=$problem, convergence_acceleration=$conv_accel, include_target_for_basis=$target_for_basis, type_of_study=tol, value=$tol, refine_acc_cost=$refine_acc_cost"
                            sbatch run_cluster_PODG_inner_FOTR.sh $problem $conv_accel $target_for_basis tol $tol $script_type $refine_acc_cost
                        done
                    done
                else
                    # For fixed, do not include the refine_acc_cost parameter
                    for mode_pair in "${mode_values[@]}"; do
                        echo "Submitting job: script_type=$script_type, problem=$problem, convergence_acceleration=$conv_accel, include_target_for_basis=$target_for_basis, type_of_study=modes, values=($mode_pair)"
                        sbatch run_cluster_PODG_inner_FOTR.sh $problem $conv_accel $target_for_basis modes $mode_pair $script_type
                    done
                    for tol in 1e-2 7e-3 3e-3 1e-3 7e-4 5e-4 3e-4 1e-4 5e-5 1e-5 5e-6 1e-6; do
                        echo "Submitting job: script_type=$script_type, problem=$problem, convergence_acceleration=$conv_accel, include_target_for_basis=$target_for_basis, type_of_study=tol, value=$tol"
                        sbatch run_cluster_PODG_inner_FOTR.sh $problem $conv_accel $target_for_basis tol $tol $script_type
                    done
                fi

            done
        done
    done
done



