#!/bin/bash

# Loop over the two script types: "adaptive" and "fixed"
for script_type in fixed adaptive; do
    for problem in 1 2 3; do
        for common_basis in True False; do
            for CTC_mask in True False; do
                for interp_scheme in "Lagr" "CubSpl"; do
                  # Define mode values list
                    mode_values=(1 3 5 8 13 17 22 27 35 40)
                    for mode in "${mode_values[@]}"; do
                        echo "Submitting job: script_type=$script_type, problem=$problem, common_basis=$common_basis, interpolation_scheme=$interp_scheme, type_of_study=modes, CTC_mask=$CTC_mask, value=$mode"
                        sbatch run_cluster_sPODG_inner_FRTO.sh $problem $common_basis $interp_scheme modes $CTC_mask $mode $script_type
                    done
                    # Submit jobs using --tol values
                    for tol in 1e-2 5e-3 1e-3 5e-4 1e-4 5e-5 1e-5 5e-6 1e-6 5e-7 1e-7; do
                        echo "Submitting job: script_type=$script_type, problem=$problem, common_basis=$common_basis, interpolation_scheme=$interp_scheme, type_of_study=tol, CTC_mask=$CTC_mask, value=$tol"
                        sbatch run_cluster_sPODG_inner_FRTO.sh $problem $common_basis $interp_scheme tol $CTC_mask $tol $script_type
                    done
                done
            done
        done
    done
done



