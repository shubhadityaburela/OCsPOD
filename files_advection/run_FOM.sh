#!/bin/bash
set -euo pipefail

# Usage:
#   ./run_FOM_local.sh [type_of_problem]
# If no arg is provided, default_type_of_problem is used.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Accept single optional argument: type_of_problem
DEFAULT_TYPE="Shifting"
type_of_problem="${1:-$DEFAULT_TYPE}"

# Hardcoded values (kept as in your cluster call)
num_controls=20
# Example grid -- change here if you want different defaults
# NOTE: FOM expects 3 grid params; adjust if needed.
g1=3201
g2=2400
g3=1

# Python CLI args that matched your cluster call
Nsteps=20000
out_root="$PROJECT_ROOT"
other_flags=(0 1e-3)  # the two trailing numeric args in your cluster version

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:}${PYTHONPATH:-}"

mkdir -p "$PROJECT_ROOT/data/logs"
timestamp=$(date +"%Y%m%dT%H%M%S")
logfile="$PROJECT_ROOT/data/logs/FOM_p${type_of_problem}_c${num_controls}_${timestamp}.log"

echo "Running local FOM (project root: $PROJECT_ROOT)"
echo "  type_of_problem: $type_of_problem"
echo "  num_controls:    $num_controls"
echo "  grid params:     $g1 $g2 $g3"
echo "  logfile:         $logfile"

cd "$PROJECT_ROOT" || { echo "Failed to cd to $PROJECT_ROOT"; exit 1; }

# Call the same script relative to project root (matches cluster: files_advection/FOM.py ...)
python3 files_advection/FOM.py "$type_of_problem" "$g1" "$g2" "$g3" "$Nsteps" "$out_root" "${other_flags[@]}" "$num_controls" 2>&1 | tee "$logfile"
