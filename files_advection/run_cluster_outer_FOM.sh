#!/bin/bash

type_of_problem="Constant_shift"   # e.g. "Shifting" or "Constant_shift"
CTC_mask="False"            # "True" or "False"
grid_str="1000 8000 1"  # 1000 8000 1   or     3200 3360 1

echo "Submitting job: type_of_problem=$type_of_problem, CTC_mask=$CTC_mask, grid=\"$grid_str\""
sbatch run_cluster_inner_FOM.sh "$type_of_problem" "$CTC_mask" "$grid_str"


