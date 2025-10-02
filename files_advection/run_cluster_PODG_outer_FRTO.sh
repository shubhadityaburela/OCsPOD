#!/bin/bash
# Loop over the two script types: "adaptive" and "fixed"
for script_type in fixed adaptive; do
    for common_basis in True False; do
        for CTC_mask in False; do
            # Submit jobs for different tolerances
            for tol in 1e-2 5e-3 1e-3 5e-4 1e-4 5e-5 1e-5 5e-6 1e-6 5e-7 1e-7; do
                echo "Submitting job: script_type=$script_type, common_basis=$common_basis, type_of_study=tol, CTC_mask=$CTC_mask, value=$tol"
                sbatch run_cluster_PODG_inner_FRTO.sh $common_basis tol $CTC_mask $tol $script_type
            done
        done
    done
done



