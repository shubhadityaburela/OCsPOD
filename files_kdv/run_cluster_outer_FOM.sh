#!/bin/bash

fully_nonlinear="True"   # e.g. True for nonlinear problem and False for fully linear
CTC_mask="False"            # "True" or "False"
grid_str="1000 120000 16"

echo "Submitting job: fully_nonlinear=$fully_nonlinear, CTC_mask=$CTC_mask, grid=\"$grid_str\""
sbatch run_cluster_inner_FOM.sh "$fully_nonlinear" "$CTC_mask" "$grid_str"


