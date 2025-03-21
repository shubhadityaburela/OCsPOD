#!/bin/bash

# Loop over the two script types: "adaptive" and "fixed"
for script_type in adaptive fixed; do
    for problem in 1 2 3; do
        for conv_accel in True; do
            for target_for_basis in False True; do
                # Define an array of mode pairs (each pair represents --modes value1 value2)
                mode_values=( "5 5" "20 20" "35 35" "50 50" "65 65" "90 90" "120 120" "150 150" "180 180")
                for mode_pair in "${mode_values[@]}"; do
                    echo "Submitting job: script_type=$script_type, problem=$problem, convergence_acceleration=$conv_accel, include_target_for_basis=$target_for_basis, type_of_study=modes, values=($mode_pair)"
                    sbatch run_cluster_PODG_inner_FOTR.sh $problem $conv_accel $target_for_basis modes $mode_pair $script_type
                done
                # Submit jobs using --tol values
                for tol in 1e-2 1e-3; do
                    echo "Submitting job: script_type=$script_type, problem=$problem, convergence_acceleration=$conv_accel, include_target_for_basis=$target_for_basis, type_of_study=tol, value=$tol"
                    sbatch run_cluster_PODG_inner_FOTR.sh $problem $conv_accel $target_for_basis tol $tol $script_type
                done
            done
        done
    done
done



