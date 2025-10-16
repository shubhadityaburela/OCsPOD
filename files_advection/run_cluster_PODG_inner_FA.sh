#!/bin/bash
#SBATCH --job-name=PODG_FOTR_FA       # Job name
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

echo "PODG run FA"

# Common command-line arguments
type_of_problem=$1
common_basis=$2  # Should be either true or false
param_type=$3    # Should be either "modes" or "tol"
CTC_mask=$4

# Depending on the parameter type, capture the proper values.
if [ "$param_type" = "tol" ]; then
    tol_value=$5      # Single tolerance value
else
    mode=$5          # First mode value
fi

problem_number=$6

grid_str="${@: -1}"
read -r -a grid_params <<< "$grid_str"

# Decide which Python script to run based on the script_type parameter.

if [ "$param_type" = "tol" ]; then
    python3 files_advection/PODG_FOTR_FA.py $type_of_problem $problem_number $common_basis "${grid_params[@]:0:3}" 20000 "/work/burela" 0 1e-3 $CTC_mask --tol $tol_value
else
    python3 files_advection/PODG_FOTR_FA.py $type_of_problem $problem_number $common_basis "${grid_params[@]:0:3}" 20000 "/work/burela" 0 1e-3 $CTC_mask --modes $mode
fi

date

