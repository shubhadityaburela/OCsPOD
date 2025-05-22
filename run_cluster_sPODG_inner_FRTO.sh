#!/bin/bash
#SBATCH --job-name=sPODG       # Job name
#SBATCH --ntasks=1            # Run on a single CPU
#SBATCH --mem=28gb            # Job memory request
#SBATCH --time=150:00:00       # Time limit
#SBATCH --output=/work/burela/sPODG_%j.log  # Standard output log
#SBATCH --partition=gbr
#SBATCH --nodelist=node748

# Print job info
pwd; hostname; date

echo "sPODG run with 40 controls"

# Common command-line arguments
problem=$1
common_basis=$2
interp_scheme=$3
param_type=$4    # Should be either "modes" or "tol"

# Depending on the parameter type, capture the proper values
if [ "$param_type" = "tol" ]; then
    tol_value=$5      # Single tolerance value
    script_type=$6    # "adaptive" or "fixed"
else
    mode1=$5         # mode value
    script_type=$6    # "adaptive" or "fixed"
fi

# Decide which Python script to run based on the script_type parameter
if [ "$script_type" = "adaptive" ]; then
    if [ "$param_type" = "tol" ]; then
        python3 sPODG_FRTO_adaptive.py $problem $common_basis $interp_scheme 20000 "/work/burela" --tol $tol_value
    else
        python3 sPODG_FRTO_adaptive.py $problem $common_basis $interp_scheme 20000 "/work/burela" --modes $mode
    fi
else
    if [ "$param_type" = "tol" ]; then
        python3 sPODG_FRTO.py $problem $common_basis $interp_scheme 20000 "/work/burela" --tol $tol_value
    else
        python3 sPODG_FRTO.py $problem $common_basis $interp_scheme 20000 "/work/burela" --modes $mode
    fi
fi

date

