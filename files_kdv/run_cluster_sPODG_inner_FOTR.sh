#!/bin/bash
#SBATCH --job-name=sPODG_FOTR       # Job name
#SBATCH --ntasks=1            # Run on a single CPU
#SBATCH --mem=48gb            # Job memory request
#SBATCH --time=150:00:00       # Time limit
#SBATCH --output=/work/burela/sPODG_%j.log  # Standard output log
#SBATCH --partition=gbr
#SBATCH --nodelist=node748
#SBATCH --chdir=/homes/math/burela/KDV_burgers

export PYTHONUNBUFFERED=1

# Print job info
pwd; hostname; date

export PYTHONPATH="$PWD:$PYTHONPATH"

echo "sPODG run FOTR"

# Common command-line arguments
fully_nonlinear=$1
common_basis=$2
param_type=$3    # Should be either "modes" or "tol"
CTC_mask=$4

# Depending on the parameter type, capture the proper values
if [ "$param_type" = "tol" ]; then
    tol_value=$5      # Single tolerance value
    script_type=$6    # "adaptive" or "fixed"
else
    mode1=$5        # First mode value
    mode2=$6          # Second mode value
    script_type=$7    # "adaptive" or "fixed"
fi


grid_str="${@: -1}"
read -r -a grid_params <<< "$grid_str"


# Decide which Python script to run based on the script_type parameter
if [ "$script_type" = "adaptive" ]; then
    if [ "$param_type" = "tol" ]; then
        python3 files_kdv/sPODG_FOTR_kdv_adaptive.py $fully_nonlinear $common_basis "${grid_params[@]:0:3}" 20000 "/work/burela" 0 1e-3 $CTC_mask --tol $tol_value
    else
        python3 files_kdv/sPODG_FOTR_kdv_adaptive.py $fully_nonlinear $common_basis "${grid_params[@]:0:3}" 20000 "/work/burela" 0 1e-3 $CTC_mask --modes $mode1 $mode2
    fi
else
    if [ "$param_type" = "tol" ]; then
        python3 files_kdv/sPODG_FOTR_kdv.py $fully_nonlinear $common_basis "${grid_params[@]:0:3}" 20000 "/work/burela" 0 1e-3 $CTC_mask --tol $tol_value
    else
        python3 files_kdv/sPODG_FOTR_kdv.py $fully_nonlinear $common_basis "${grid_params[@]:0:3}" 20000 "/work/burela" 0 1e-3 $CTC_mask --modes $mode1 $mode2
    fi
fi

date

