#!/bin/bash
#SBATCH --job-name=sPODG       # Job name
#SBATCH --ntasks=1            # Run on a single CPU
#SBATCH --mem=48gb            # Job memory request
#SBATCH --time=150:00:00       # Time limit
#SBATCH --output=/work/burela/sPODG_%j.log  # Standard output log
#SBATCH --partition=gbr
#SBATCH --nodelist=node748

export PYTHONUNBUFFERED=1


# Print job info
pwd; hostname; date

echo "sPODG run with 40 controls"

# Common command-line arguments
problem=$1
common_basis=$2
interp_scheme=$3
param_type=$4    # Should be either "modes" or "tol"
CTC_mask=$5

# Depending on the parameter type, capture the proper values
if [ "$param_type" = "tol" ]; then
    tol_value=$6      # Single tolerance value
    script_type=$7    # "adaptive" or "fixed"
else
    mode=$6         # mode value
    script_type=$7    # "adaptive" or "fixed"
fi

# Decide which Python script to run based on the script_type parameter
if [ "$script_type" = "adaptive" ]; then
    if [ "$param_type" = "tol" ]; then
        python3 sPODG_FRTO_adaptive.py $problem $common_basis 20000 "/work/burela" 0 1e-3 $CTC_mask $interp_scheme --tol $tol_value
    else
        python3 sPODG_FRTO_adaptive.py $problem $common_basis 20000 "/work/burela" 0 1e-3 $CTC_mask $interp_scheme --modes $mode
    fi
else
    if [ "$param_type" = "tol" ]; then
        python3 sPODG_FRTO.py $problem $common_basis 20000 "/work/burela" 0 1e-3 $CTC_mask $interp_scheme --tol $tol_value
    else
        python3 sPODG_FRTO.py $problem $common_basis 20000 "/work/burela" 0 1e-3 $CTC_mask $interp_scheme --modes $mode
    fi
fi

date

