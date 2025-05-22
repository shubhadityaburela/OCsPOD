#!/bin/bash
# Loop over the two script types: "adaptive" and "fixed"
for script_type in adaptive fixed; do
    for problem in 1 2 3; do
        for common_basis in False True; do
            # Define mode values list
            mode_values=(5 10 20 30 50 70 90 110 130 150 170 190 210 230 250 270 300)
            for mode in "${mode_values[@]}"; do
                echo "Submitting job: script_type=$script_type, problem=$problem, common_basis=$common_basis, type_of_study=modes, value=$mode"
                sbatch run_cluster_PODG_inner_FRTO.sh $problem $common_basis modes $mode $script_type
            done
            # Submit jobs for different tolerances
            for tol in 1e-2 5e-3 1e-3 5e-4 1e-4 5e-5 1e-5 5e-6 1e-6 5e-7 1e-7 5e-8 1e-8; do
                echo "Submitting job: script_type=$script_type, problem=$problem, common_basis=$common_basis, type_of_study=tol, value=$tol"
                sbatch run_cluster_PODG_inner_FRTO.sh $problem $common_basis tol $tol $script_type
            done
        done
    done
done



