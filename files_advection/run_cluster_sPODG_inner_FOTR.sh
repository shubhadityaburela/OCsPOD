#!/bin/bash
#SBATCH --job-name=sPODG_FOTR       # Job name
#SBATCH --ntasks=1            # Run on a single CPU
#SBATCH --mem=48gb            # Job memory request
#SBATCH --time=150:00:00       # Time limit
#SBATCH --output=/work/burela/sPODG_%j.log  # Standard output log
##SBATCH --partition=gbr
##SBATCH --nodelist=node748
#SBATCH --chdir=/homes/math/burela/FOTR_advection

export PYTHONUNBUFFERED=1

# Print job info
pwd; hostname; date

export PYTHONPATH="$PWD:$PYTHONPATH"

echo "sPODG run FOTR"

# Common command-line arguments
type_of_problem=$1
common_basis=$2  # Should be either true or false
param_type=$3    # Should be either "modes" or "tol"


# Depending on the parameter type, capture the proper values.
if [ "$param_type" = "tol" ]; then
    tol_value=$4      # Single tolerance value
    script_type=$5    # "adaptive" or "fixed"
else
    mode1=$4          # First mode value
    mode2=$5          # Second mode value
    script_type=$6    # "adaptive" or "fixed"
fi

grid_str="${@: -1}"
read -r -a grid_params <<< "$grid_str"

# Decide which Python script to run based on the script_type parameter.
if [ "$script_type" = "adaptive" ]; then
    if [ "$param_type" = "tol" ]; then
        python3 files_advection/sPODG_FOTR_adaptive.py $type_of_problem $common_basis "${grid_params[@]:0:3}" 20000 "/work/burela" 0 1e-3 --tol $tol_value
    else
        python3 files_advection/sPODG_FOTR_adaptive.py $type_of_problem $common_basis "${grid_params[@]:0:3}" 20000 "/work/burela" 0 1e-3 --modes $mode1 $mode2
    fi
else
    if [ "$param_type" = "tol" ]; then
        python3 files_advection/sPODG_FOTR.py $type_of_problem $common_basis "${grid_params[@]:0:3}" 20000 "/work/burela" 0 1e-3 --tol $tol_value
    else
        python3 files_advection/sPODG_FOTR.py $type_of_problem $common_basis "${grid_params[@]:0:3}" 20000 "/work/burela" 0 1e-3 --modes $mode1 $mode2
    fi
fi

date
