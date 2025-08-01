#!/bin/bash
for problem in 1 2 3; do
    for CTC_mask in True; do
        echo "Submitting job: problem=$problem, CTC_mask=$CTC_mask"
        sbatch run_cluster_inner_FOM.sh $problem $CTC_mask
    done
done



