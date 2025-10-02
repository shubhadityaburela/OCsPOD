#!/bin/bash
#SBATCH --job-name=FOM       # Job name
#SBATCH --ntasks=1            # Run on a single CPU
#SBATCH --mem=32gb            # Job memory request
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

python3 files_kdv/FOM_kdv.py False 1000 8000 4 20000 "/work/burela" False 0 1e-3

date

