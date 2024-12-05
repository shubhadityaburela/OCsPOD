#!/bin/bash

# Specify which problem you want to run (1, 2 or 3)
problem=$1
if [[ -z "$problem" ]]; then
    echo "Error: No problem number provided!"
    echo "Usage: $0 <problem> [modes|tol]"
    echo "Example: $0 1 modes   # Problem 1 running with modes"
    echo "Example: $0 1 tol     # Problem 1 running with tolerance"
    exit 1
fi

mkdir -p data/LOG

# Specify which study to run : either "modes" or "tol"
selection=$2
if [[ "$selection" == "modes" ]]; then
    echo "Problem $problem running with modes"

    # Number of modes list for each of the problems
    case "$problem" in
        1) args_list=("4" "5" "8" "10" "12" "15" "17" "20" "25" "30" "35" "40") ;;
        2) args_list=("4" "5" "8" "10" "12" "15" "17" "20" "25" "30" "35" "40" "50" "60" "70" "80" "90" "100" "110" "120" "130" "140") ;;
        3) args_list=("4" "5" "8" "10" "12" "15" "17" "20" "25" "30" "35" "40" "50" "60" "70" "80" "90" "100" "110" "120" "130" "140") ;;
        *) echo "Error: Invalid problem number!"; exit 1 ;;
    esac

    for arg in "${args_list[@]}"; do
        python3 -u PODG_FOTR_FA.py "$problem" --modes="$arg" 2>&1 | tee "data/LOG/P${problem}_PODG_${arg}.txt"
    done

elif [[ "$selection" == "tol" ]]; then
    echo "Problem $problem running with tolerance"

    # Values of tolerances considered for our run
    case "$problem" in
        1) args_list=("1e-2" "1e-3") ;;
        2) args_list=("1e-2" "1e-3") ;;
        3) args_list=("1e-2" "1e-3") ;;
        *) echo "Error: Invalid problem number!"; exit 1 ;;
    esac

    for arg in "${args_list[@]}"; do
        python3 -u PODG_FOTR_FA.py "$problem" --tol="$arg" 2>&1 | tee "data/LOG/P${problem}_PODG_${arg}.txt"
    done

else
    echo "Error: No type selection has been made!!!!!!"
    echo "Usage: $0 <problem> [modes|tol]"
    echo "Example: $0 1 modes   # Problem 1 running with modes"
    echo "Example: $0 1 tol     # Problem 1 running with tolerance"
    exit 1
fi











##!/bin/bash
#
## Select which problem you want to run (Uncomment the problem and the args_list corresponding to that
## and just comment out the rest)
#problem="1"
#args_list=("4" "5" "8" "10" "12" "15" "17" "20" "25" "30" "35" "40")
#
##problem="2"
##args_list=("4" "5" "8" "10" "12" "15" "17" "20" "25" "30" "35" "40" "50" "60" "70" "80" "90" "100" "110" "120" "130" "140")
#
##problem="3"
##args_list=("4" "5" "8" "10" "12" "15" "17" "20" "25" "30" "35" "40" "50" "60" "70" "80" "90" "100" "110" "120" "130" "140")
#
#for arg in "${args_list[@]}"; do
#    python3 -u PODG_FOTR_FA.py "$problem" --modes="$arg" 2>&1 | tee "data/LOG/P${problem}_PODG_${arg}.txt"
#done

