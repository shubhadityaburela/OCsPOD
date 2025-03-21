#!/bin/bash
#SBATCH --job-name=sPODG_PODG       # Job name
#SBATCH --ntasks=1            # Run on a single CPU
#SBATCH --mem=16gb            # Job memory request
#SBATCH --time=150:00:00       # Time limit
#SBATCH --output=/work/burela/sPODG_PODG_%j.log  # Standard output log
#SBATCH --partition=gbr
#SBATCH --nodelist=node748

# Print job info
pwd; hostname; date

echo "sPODG + PODG run with 40 controls"

# Common command-line arguments
problem=$1
conv_accel=$2
target_for_basis=$3
interp_scheme=$4
param_type=$5    # Should be either "modes" or "tol"

# Depending on the parameter type, capture the proper values
if [ "$param_type" = "tol" ]; then
    tol_value=$6      # Single tolerance value
    script_type=$7    # "adaptive" or "fixed"
else
    mode1=$6          # First mode value
    mode2=$7          # Second mode value
    script_type=$8    # "adaptive" or "fixed"
fi

# Decide which Python script to run based on the script_type parameter
if [ "$script_type" = "adaptive" ]; then
    if [ "$param_type" = "tol" ]; then
        python3 sPODG_PODG_FOTR_RA_adaptive.py $problem $conv_accel $target_for_basis $interp_scheme --tol $tol_value
    else
        python3 sPODG_PODG_FOTR_RA_adaptive.py $problem $conv_accel $target_for_basis $interp_scheme --modes $mode1 $mode2
    fi
else
    if [ "$param_type" = "tol" ]; then
        python3 sPODG_PODG_FOTR_RA.py $problem $conv_accel $target_for_basis $interp_scheme --tol $tol_value
    else
        python3 sPODG_PODG_FOTR_RA.py $problem $conv_accel $target_for_basis $interp_scheme --modes $mode1 $mode2
    fi
fi

date

