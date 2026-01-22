#!/bin/bash
#SBATCH --job-name=sPODG_FRTO       # Job name
#SBATCH --ntasks=1            # Run on a single CPU
#SBATCH --mem=48gb            # Job memory request
#SBATCH --time=150:00:00       # Time limit
#SBATCH --output=/work/burela/sPODG_%j.log  # Standard output log
##SBATCH --partition=gbr
##SBATCH --nodelist=node748
#SBATCH --chdir=/homes/math/burela/KDV_burgers

export PYTHONUNBUFFERED=1


# Print job info
pwd; hostname; date

export PYTHONPATH="$PWD:$PYTHONPATH"

echo "sPODG run FRTO"

# Common command-line arguments
type_of_problem=$1
num_controls=$2

grid_str="${@: -1}"
read -r -a grid_params <<< "$grid_str"


python3 files_advection/sPODG_FRTO.py $type_of_problem "${grid_params[@]:0:3}" 20000 "/work/burela" 0 1e-3 $num_controls

date

