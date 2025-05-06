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
common_basis=$2  # Should be either true or false
param_type=$3    # Should be either "modes" or "tol"

# Depending on the parameter type, capture the proper values.
if [ "$param_type" = "tol" ]; then
    tol_value=$4      # Single tolerance value
    script_type=$5    # "adaptive" or "fixed" or "everytime"
else
    mode=$4          # First mode value
    script_type=$5    # "adaptive" or "fixed" or "everytime"
fi

# Decide which Python script to run based on the script_type parameter.
if [ "$script_type" = "adaptive" ]; then
    if [ "$param_type" = "tol" ]; then
        python3 PODG_FRTO_adaptive.py $problem $common_basis 50000 "/work/burela" --tol $tol_value
    else
        python3 PODG_FRTO_adaptive.py $problem $common_basis 50000 "/work/burela" --modes $mode
    fi
else
    if [ "$param_type" = "tol" ]; then
        python3 PODG_FRTO.py $problem $common_basis 50000 "/work/burela" --tol $tol_value
    else
        python3 PODG_FRTO.py $problem $common_basis 50000 "/work/burela" --modes $mode
    fi
fi

date

