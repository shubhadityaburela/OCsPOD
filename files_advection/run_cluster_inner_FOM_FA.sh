#!/bin/bash
#SBATCH --job-name=FOM       # Job name
#SBATCH --ntasks=1            # Run on a single CPU
#SBATCH --mem=20gb            # Job memory request
#SBATCH --time=150:00:00       # Time limit
#SBATCH --output=/work/burela/FOM_%j.log  # Standard output log
#SBATCH --partition=gbr
#SBATCH --nodelist=node748
#SBATCH --chdir=/homes/math/burela/KDV_burgers

export PYTHONUNBUFFERED=1

# Print job info
pwd; hostname; date

export PYTHONPATH="$PWD:$PYTHONPATH"

echo "FOM run"

# Common command-line arguments
type_of_problem=$1
CTC_mask=$2
problem_number=$3

grid_str="${@: -1}"
read -r -a grid_params <<< "$grid_str"

python3 files_advection/FOM_FA.py $type_of_problem $problem_number "${grid_params[@]:0:3}" 20000 "/work/burela" $CTC_mask 0 1e-3

date

