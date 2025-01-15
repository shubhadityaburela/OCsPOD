#!/bin/bash

# Specify which problem you want to run (1, 2 or 3)
problem=$1
if [[ -z "$problem" ]]; then
    echo "Error: No problem number provided!"
    echo "Usage: $0 <problem> [lamda]"
    echo "Example: $0 1 lamda   # Problem 1 running with specified lamda"
    exit 1
fi

mkdir -p data/LOG

# Specify the lamda values
selection=$2
if [[ "$selection" == "lamda" ]]; then
    echo "Problem $problem running with lamda"

    # lamda list for each of the problems
    case "$problem" in
        1) args_list=("1e-5" "5e-5" "1e-4" "5e-4" "1e-3" "5e-3" "1e-2" "5e-2" "1e-1" "5e-1" "1e0" "5e0" "1e1" "5e1" "1e2" "5e2" "1e3" "5e3" "1e4" "5e4" "1e5") ;;
        2) args_list=("1e-5" "5e-5" "1e-4" "5e-4" "1e-3" "5e-3" "1e-2" "5e-2" "1e-1" "5e-1" "1e0" "5e0" "1e1" "5e1" "1e2" "5e2" "1e3" "5e3" "1e4" "5e4" "1e5") ;;
        3) args_list=("1e-5" "5e-5" "1e-4" "5e-4" "1e-3" "5e-3" "1e-2" "5e-2" "1e-1" "5e-1" "1e0" "5e0" "1e1" "5e1" "1e2" "5e2" "1e3" "5e3" "1e4" "5e4" "1e5") ;;
        *) echo "Error: Invalid problem number!"; exit 1 ;;
    esac

    for arg in "${args_list[@]}"; do
        python3 -u sPODG_FRTO_NC_Lagr.py "$problem" --lamda="$arg" 2>&1 | tee "data/LOG/P${problem}_sPODG_${arg}.txt"
    done

else
    echo "Error: No type selection has been made!!!!!!"
    echo "Usage: $0 <problem> [lamda]"
    echo "Example: $0 1 lamda   # Problem 1 running with specified lamda"
    exit 1
fi
