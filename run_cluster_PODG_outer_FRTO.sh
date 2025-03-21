#!/bin/bash

# Loop over the two script types: "adaptive" and "fixed"
for script_type in adaptive fixed; do
    for problem in 1 2 3; do
        for conv_accel in True; do
            for target_for_basis in False True; do
                for mode in 5 20 35 50 65 90 120 150 180; do
                    echo "Submitting job: script_type=$script_type, problem=$problem, convergence_acceleration=$conv_accel, include_target_for_basis=$target_for_basis, type_of_study=modes, value=$mode"
                    sbatch run_cluster_PODG_inner_FRTO.sh $problem $conv_accel $target_for_basis modes $mode $script_type
                done
                # Submit jobs using --tol values
                for tol in 1e-2 1e-3; do
                    echo "Submitting job: script_type=$script_type, problem=$problem, convergence_acceleration=$conv_accel, include_target_for_basis=$target_for_basis, type_of_study=tol, value=$tol"
                    sbatch run_cluster_PODG_inner_FRTO.sh $problem $conv_accel $target_for_basis tol $tol $script_type
                done
            done
        done
    done
done



