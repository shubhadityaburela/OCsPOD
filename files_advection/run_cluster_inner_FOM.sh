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
CTC_mask=$1

python3 files_advection/FOM.py 1000 16000 8 20000 "/work/burela" $CTC_mask 0 1e-3

date

