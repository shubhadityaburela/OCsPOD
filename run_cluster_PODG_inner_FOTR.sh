#!/bin/bash
#SBATCH --job-name=PODG       # Job name
#SBATCH --ntasks=1            # Run on a single CPU
#SBATCH --mem=16gb            # Job memory request
#SBATCH --time=150:00:00       # Time limit
#SBATCH --output=/work/burela/PODG_%j.log  # Standard output log
#SBATCH --partition=gbr
#SBATCH --nodelist=node748

# Print job info
pwd; hostname; date

echo "PODG run with 40 controls"

# Common command-line arguments
problem=$1
conv_accel=$2
target_for_basis=$3
param_type=$4    # Should be either "modes" or "tol"

# Depending on the parameter type, capture the proper values.
if [ "$param_type" = "tol" ]; then
    tol_value=$5      # Single tolerance value
    script_type=$6    # "adaptive" or "fixed" or "everytime"
    refine_acc_cost=$7    # For adaptive only: refine_acc_cost value
else
    mode1=$5          # First mode value
    mode2=$6          # Second mode value
    script_type=$7    # "adaptive" or "fixed" or "everytime"
    refine_acc_cost=$8    # For adaptive only: refine_acc_cost value
fi

# Decide which Python script to run based on the script_type parameter.
if [ "$script_type" = "adaptive" ]; then
    # For adaptive, pass refine_acc_cost immediately after target_for_basis.
    if [ "$param_type" = "tol" ]; then
        python3 PODG_FOTR_RA_adaptive.py $problem $conv_accel $target_for_basis $refine_acc_cost 75000 "/work/burela" --tol $tol_value
    else
        python3 PODG_FOTR_RA_adaptive.py $problem $conv_accel $target_for_basis $refine_acc_cost 75000 "/work/burela" --modes $mode1 $mode2
    fi
elif [ "$script_type" = "everytime" ]; then
    if [ "$param_type" = "tol" ]; then
        python3 PODG_FOTR_RA_everytime.py $problem $conv_accel $target_for_basis 75000 "/work/burela" --tol $tol_value
    else
        python3 PODG_FOTR_RA_everytime.py $problem $conv_accel $target_for_basis 75000 "/work/burela" --modes $mode1 $mode2
    fi
else
    # For fixed, simply call the corresponding Python script without refine_acc_cost.
    if [ "$param_type" = "tol" ]; then
        python3 PODG_FOTR_RA.py $problem $conv_accel $target_for_basis 75000 "/work/burela" --tol $tol_value
    else
        python3 PODG_FOTR_RA.py $problem $conv_accel $target_for_basis 75000 "/work/burela" --modes $mode1 $mode2
    fi
fi

date

