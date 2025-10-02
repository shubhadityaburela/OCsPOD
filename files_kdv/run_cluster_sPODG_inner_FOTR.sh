#!/bin/bash
#SBATCH --job-name=sPODG       # Job name
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

echo "sPODG run"

# Common command-line arguments
common_basis=$1
param_type=$2    # Should be either "modes" or "tol"

# Depending on the parameter type, capture the proper values
if [ "$param_type" = "tol" ]; then
    tol1=$3      # First tolerance value
    tol2=$4      # Second tolerance value
    script_type=$5    # "adaptive" or "fixed"
else
    mode1=$3          # First mode value
    mode2=$4          # Second mode value
    mode3=$5          # Third mode value
    mode4=$6          # Fourth mode value
    script_type=$7    # "adaptive" or "fixed"
fi

# Decide which Python script to run based on the script_type parameter
if [ "$script_type" = "adaptive" ]; then
    if [ "$param_type" = "tol" ]; then
        python3 files_kdv/sPODG_FOTR_kdv_adaptive.py False $common_basis 1000 8000 4 20000 "/work/burela" 0 1e-3 False --tol $tol1 $tol2
    else
        python3 files_kdv/sPODG_FOTR_kdv_adaptive.py False $common_basis 1000 8000 4 20000 "/work/burela" 0 1e-3 False --modes $mode1 $mode2 $mode3 $mode4
    fi
else
    if [ "$param_type" = "tol" ]; then
        python3 files_kdv/sPODG_FOTR_kdv.py False $common_basis 1000 8000 4 20000 "/work/burela" 0 1e-3 False --tol $tol1 $tol2
    else
        python3 files_kdv/sPODG_FOTR_kdv.py False $common_basis 1000 8000 4 20000 "/work/burela" 0 1e-3 False --modes $mode1 $mode2 $mode3 $mode4
    fi
fi

date

