#!/bin/bash
set -euo pipefail

# Usage:
#   ./run_PODG_local.sh [type_of_problem]
# If no arg is provided, default_type_of_problem is used.
#
# This script runs the full "outer loop" sequentially (modes + tolerances),
# using the same hardcoded arrays and parameters as your original outer script.
# It will call the appropriate python (adaptive vs fixed) just like the cluster inner script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Accept single optional argument: type_of_problem
DEFAULT_TYPE="Shifting"
type_of_problem="${1:-$DEFAULT_TYPE}"

# Hardcoded (kept from your outer script)
grid_str="3201 2400 1"
script_type="adaptive"
common_basis="False"

mode_sets=(
  "5 5"   "10 10"  "20 20"  "30 30"  "40 40"  "50 50"  "60 60"  "70 70"
  "80 80" "90 90"  "100 100" "200 200" "300 300" "400 400" "500 500"
)

tolerances=(
  "1e-2" "5e-3" "1e-3" "5e-4" "1e-4" "5e-5" "1e-5" "5e-6"
  "1e-6" "5e-7" "1e-7" "5e-8" "1e-8" "5e-9" "1e-9" "5e-10" "1e-10"
)

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:}${PYTHONPATH:-}"

mkdir -p "$PROJECT_ROOT/data/logs"

# helper to run a single job (modes or tol)
run_single_job() {
    local type_of_problem="$1"
    local common_basis="$2"
    local param_type="$3"

    read -r -a grid_params <<< "$grid_str"
    g1="${grid_params[0]:-}"
    g2="${grid_params[1]:-}"
    g3="${grid_params[2]:-}"
    if [[ -z "$g1" || -z "$g2" || -z "$g3" ]]; then
        echo "Error: expected three grid parameters in grid_str: '$grid_str'"
        return 1
    fi

    local py_script
    if [[ "$script_type" == "adaptive" ]]; then
        py_script="files_advection/PODG_FOTR_adaptive.py"
    else
        py_script="files_advection/PODG_FOTR.py"
    fi

    ts=$(date +"%Y%m%dT%H%M%S")
    if [[ "$param_type" == "tol" ]]; then
        local tol_value="$4"
        logfile="$PROJECT_ROOT/data/logs/PODG_${type_of_problem}_${script_type}_common${common_basis}_tol${tol_value}_g${g1}-${g2}-${g3}_${ts}.log"
        echo "RUN (tol) type=$type_of_problem common=$common_basis tol=$tol_value grid=$g1 $g2 $g3 -> $logfile"
        cd "$PROJECT_ROOT" || return 1
        python3 "$py_script" "$type_of_problem" "$common_basis" "$g1" "$g2" "$g3" 20000 "$PROJECT_ROOT" 0 1e-3 --tol "$tol_value" 2>&1 | tee "$logfile"
        return ${PIPESTATUS[0]:-0}
    else
        local mode1="$4"
        local mode2="$5"
        logfile="$PROJECT_ROOT/data/logs/PODG_${type_of_problem}_${script_type}_common${common_basis}_modes${mode1}-${mode2}_g${g1}-${g2}-${g3}_${ts}.log"
        echo "RUN (modes) type=$type_of_problem common=$common_basis modes=$mode1 $mode2 grid=$g1 $g2 $g3 -> $logfile"
        cd "$PROJECT_ROOT" || return 1
        python3 "$py_script" "$type_of_problem" "$common_basis" "$g1" "$g2" "$g3" 20000 "$PROJECT_ROOT" 0 1e-3 --modes "$mode1" "$mode2" 2>&1 | tee "$logfile"
        return ${PIPESTATUS[0]:-0}
    fi
}

echo "Starting local PODG batch (project root: $PROJECT_ROOT), type_of_problem='$type_of_problem'"
for ms in "${mode_sets[@]}"; do
    read -r mode1 mode2 <<< "$ms"
    run_single_job "$type_of_problem" "$common_basis" "modes" "$mode1" "$mode2" || echo "Job failed for modes $mode1 $mode2"
done

for tol in "${tolerances[@]}"; do
    run_single_job "$type_of_problem" "$common_basis" "tol" "$tol" || echo "Job failed for tol $tol"
done

echo "Local PODG batch complete."
