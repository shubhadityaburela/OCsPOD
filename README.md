# "Optimal control with sPOD".

## Prerequisites

Before running the scripts, please ensure that you have the necessary packages installed. You need to install the Conda package manager and use the provided environment files to create a Conda virtual environment.

## Setting up the Environment

After installing Conda:

For Mac users:

```bash
conda env create -f env_with_accel_Mac.yml
conda activate env_with_accel
```

For other OS:

```bash
conda env create -f env_with_accel_others.yml
conda activate env_with_accel
```


## Running the Scripts

This repository includes three script files that need to be executed to reproduce the results. The script files are:

1. `run_FOM.sh`  
   Usage: `./run_FOM.sh arg`  
   - The value of `arg` can be `1`, `2`, or `3`, corresponding to the three examples shown in the paper.  
   - This script runs the tests for the Full-Order Model (FOM) for all example problems.

2. `run_PODG.sh`  
   Usage: `./run_PODG.sh arg1 arg2`  
   - `arg1`: Same as for FOM  
   - `arg2`: Can be either `modes` for mode-based study or `tol` for tolerance-based study.  
   - This script runs the tests for the POD-Galerkin model.

3. `run_sPODG.sh`  
   Usage: `./run_sPODG.sh arg1 arg2`  
   - The arguments are the same as those in the POD-Galerkin case.  
   - This script runs the tests for the sPOD-Galerkin model.
  

Note: This code is under constant development. For the last standard version, please refer to [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14355727.svg)](https://doi.org/10.5281/zenodo.14355726) 
