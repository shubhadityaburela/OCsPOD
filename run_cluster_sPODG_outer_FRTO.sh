#!/bin/bash

# Loop over the two script types: "adaptive" and "fixed"
for script_type in adaptive fixed; do
    for problem in 1 2 3; do
        for conv_accel in True; do
            for target_for_basis in False True; do
                for interp_scheme in "CubSpl" "Lagr"; do
                    for mode in 5 10 15 20 25 30 35 40 45 50 55 60; do
                        echo "Submitting job: script_type=$script_type, problem=$problem, convergence_acceleration=$conv_accel, include_target_for_basis=$target_for_basis, interpolation_scheme=$interp_scheme, type_of_study=modes, values=$mode"
                        sbatch run_cluster_sPODG_inner_FRTO.sh $problem $conv_accel $target_for_basis $interp_scheme modes $mode $script_type
                    done
                    # Submit jobs using --tol values
                    for tol in 1e-2 1e-3; do
                        echo "Submitting job: script_type=$script_type, problem=$problem, convergence_acceleration=$conv_accel, include_target_for_basis=$target_for_basis, interpolation_scheme=$interp_scheme, type_of_study=tol, value=$tol"
                        sbatch run_cluster_sPODG_inner_FRTO.sh $problem $conv_accel $target_for_basis $interp_scheme tol $tol $script_type
                    done
                done
            done
        done
    done
done



