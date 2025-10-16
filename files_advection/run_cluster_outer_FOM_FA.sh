#!/bin/bash

type_of_problem="Constant_shift"   # e.g. "Shifting" or "Constant_shift"
problem_number=1           # 1, 2, or 3 for only "Shifting" type problem
CTC_mask="False"            # "True" or "False"
grid_str="1000 8000 1"         # 1000 8000 1   or     3200 3360 1

echo "Submitting job: type_of_problem=$type_of_problem, problem_number=$problem_number, CTC_mask=$CTC_mask, grid=\"$grid_str\""
sbatch run_cluster_inner_FOM_FA.sh "$type_of_problem" "$CTC_mask" "$problem_number" "$grid_str"


