#!/bin/bash

# Loop over the two script types: "adaptive" and "fixed"
for script_type in adaptive fixed; do
    for problem in 1 2 3; do
        for common_basis in False True; do
            for interp_scheme in "CubSpl" "Lagr"; do
                # Submit jobs using --tol values
                for tol in 1e-2 7e-3 3e-3 1e-3 7e-4 5e-4 3e-4 1e-4 5e-5 1e-5 5e-6 1e-6; do
                    echo "Submitting job: script_type=$script_type, problem=$problem, common_basis=$common_basis, interpolation_scheme=$interp_scheme, type_of_study=tol, value=$tol"
                    sbatch run_cluster_sPODG_inner_FOTR.sh $problem $common_basis $interp_scheme tol $tol $script_type
                done

            done
        done
    done
done



