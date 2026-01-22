#!/bin/bash
#SBATCH --job-name=PODG_FRTO       # Job name
#SBATCH --ntasks=1            # Run on a single CPU
#SBATCH --mem=20gb            # Job memory request
#SBATCH --time=150:00:00       # Time limit
#SBATCH --output=/work/burela/PODG_%j.log  # Standard output log
##SBATCH --partition=gbr
##SBATCH --nodelist=node748
#SBATCH --chdir=/homes/math/burela/KDV_burgers

export PYTHONUNBUFFERED=1

# Print job info
pwd; hostname; date

export PYTHONPATH="$PWD:$PYTHONPATH"

echo "PODG run FRTO"

# Common command-line arguments
type_of_problem=$1
common_basis=$2  # Should be either true or false
param_type=$3    # Should be either "modes" or "tol"

# Depending on the parameter type, capture the proper values.
if [ "$param_type" = "tol" ]; then
    tol_value=$4      # Single tolerance value
else
    mode=$4          # First mode value
fi

grid_str="${@: -1}"
read -r -a grid_params <<< "$grid_str"

# Decide which Python script to run based on the script_type parameter.
if [ "$param_type" = "tol" ]; then
    python3 files_advection/PODG_FRTO_adaptive.py $type_of_problem $common_basis "${grid_params[@]:0:3}" 20000 "/work/burela" 0 1e-3 --tol $tol_value
else
    python3 files_advection/PODG_FRTO_adaptive.py $type_of_problem $common_basis "${grid_params[@]:0:3}" 20000 "/work/burela" 0 1e-3 --modes $mode
fi


date

