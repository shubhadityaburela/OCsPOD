#!/bin/bash

# Specify which problem you want to run (1, 2 or 3)
problem=$1
if [[ -z "$problem" ]]; then
    echo "Error: No problem number provided!"
    echo "Usage: $0 <problem>"
    echo "Example: $0 1   # Problem 1 running"
    exit 1
fi

mkdir -p data/LOG

python3 -u FOM.py "$problem" 2>&1 | tee "data/LOG/P${problem}_FOM.txt"
