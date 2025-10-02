#!/bin/bash
#SBATCH --job-name=PODG       # Job name
#SBATCH --ntasks=1            # Run on a single CPU
#SBATCH --mem=20gb            # Job memory request
#SBATCH --time=150:00:00       # Time limit
#SBATCH --output=/work/burela/PODG_%j.log  # Standard output log
#SBATCH --partition=gbr
#SBATCH --nodelist=node748
#SBATCH --chdir=/homes/math/burela/KDV_burgers

export PYTHONUNBUFFERED=1

# Print job info
pwd; hostname; date

export PYTHONPATH="$PWD:$PYTHONPATH"

echo "PODG run with 40 controls"

# Common command-line arguments
common_basis=$1
param_type=$2    # Should be either "modes" or "tol"
CTC_mask=$3

# Depending on the parameter type, capture the proper values.
if [ "$param_type" = "tol" ]; then
    tol_value=$4      # Single tolerance value
    script_type=$5    # "adaptive" or "fixed"
else
    mode1=$4          # First mode value
    mode2=$5          # Second mode value
    script_type=$6    # "adaptive" or "fixed"
fi

# Decide which Python script to run based on the script_type parameter.
if [ "$script_type" = "adaptive" ]; then
    if [ "$param_type" = "tol" ]; then
        python3 files_advection/PODG_FOTR_adaptive.py $common_basis 1000 16000 8 20000 "/work/burela" 0 1e-3 $CTC_mask --tol $tol_value
    else
        python3 files_advection/PODG_FOTR_adaptive.py $common_basis 1000 16000 8 20000 "/work/burela" 0 1e-3 $CTC_mask --modes $mode1 $mode2
    fi
else
    if [ "$param_type" = "tol" ]; then
        python3 files_advection/PODG_FOTR.py $common_basis 1000 16000 8 20000 "/work/burela" 0 1e-3 $CTC_mask --tol $tol_value
    else
        python3 files_advection/PODG_FOTR.py $common_basis 1000 16000 8 20000 "/work/burela" 0 1e-3 $CTC_mask --modes $mode1 $mode2
    fi
fi

date

