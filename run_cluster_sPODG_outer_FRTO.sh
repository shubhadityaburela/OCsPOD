#!/bin/bash

# Loop over the two script types: "adaptive" and "fixed"
for script_type in adaptive fixed; do
    for problem in 1 2 3; do
        for common_basis in False True; do
            for interp_scheme in "CubSpl" "Lagr"; do
              # Define mode values list
                mode_values=(1 3 5 8 13 17 22 27 35 40)
                for mode in "${mode_values[@]}"; do
                    echo "Submitting job: script_type=$script_type, problem=$problem, common_basis=$common_basis, interpolation_scheme=$interp_scheme, type_of_study=modes, value=$mode"
                    sbatch run_cluster_sPODG_inner_FRTO.sh $problem $common_basis $interp_scheme modes $mode $script_type
                done
                # Submit jobs using --tol values
                for tol in 1e-2 5e-3 1e-3 5e-4 1e-4 5e-5 1e-5; do
                    echo "Submitting job: script_type=$script_type, problem=$problem, common_basis=$common_basis, interpolation_scheme=$interp_scheme, type_of_study=tol, value=$tol"
                    sbatch run_cluster_sPODG_inner_FRTO.sh $problem $common_basis $interp_scheme tol $tol $script_type
                done
            done
        done
    done
done



