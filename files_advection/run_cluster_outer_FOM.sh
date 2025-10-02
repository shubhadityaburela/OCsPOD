#!/bin/bash
for CTC_mask in False; do
    echo "Submitting job: CTC_mask=$CTC_mask"
    sbatch run_cluster_inner_FOM.sh $CTC_mask
done




