import warnings
from pathlib import Path

import numpy as np
import os
import matplotlib.pyplot as plt
import re
from typing import List, Tuple
import argparse
from scipy.interpolate import interp1d

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"]})

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

impath = "../../../Plots/advection_plot/Constant_shift/ROM/"
os.makedirs(impath, exist_ok=True)

# Regex pattern to extract real numbers from file names
modes_pattern = re.compile(r"modes\s*=\s*[\(\[]\s*(\d+)", re.IGNORECASE)
tol_pattern = re.compile(r"tol=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")

parser = argparse.ArgumentParser(description="Input the variables for running the script.")
parser.add_argument("Framework", type=str, choices=["FOTR", "FRTO"], help="Choose the framework")
args = parser.parse_args()


def get_sorted_mode_dirs(path: str) -> List[str]:
    mode_dirs = []
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if not os.path.isdir(full_path):
            continue
        m = modes_pattern.search(entry)
        if not m:
            continue
        mode_dirs.append((int(m.group(1)), full_path))

    mode_dirs.sort(key=lambda x: x[0])  # sort ascending by the extracted mode
    return [p for _, p in mode_dirs]


def get_sorted_tol_dirs(path: str) -> List[str]:
    """
    Return a list of subdirectory paths that contain "tol=..." sorted by numeric tol value (ascending).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"base path not found: {path}")

    tol_dirs: List[Tuple[float, str]] = []
    for entry in p.iterdir():
        if not entry.is_dir():
            continue
        m = tol_pattern.search(entry.name)
        if not m:
            continue
        try:
            tol_val = float(m.group(1))
        except ValueError:
            # skip directories with unparsable tol
            warnings.warn(f"Could not parse tol from directory name: {entry.name}", RuntimeWarning)
            continue
        tol_dirs.append((tol_val, str(entry.resolve())))

    # numeric sort by tolerance (ascending)
    tol_dirs.sort(key=lambda x: x[0])
    return [path for _, path in tol_dirs]


def extract_data(sorted_paths, file1, file2, file3):
    data1 = []
    data2 = []
    data3 = []

    for root in sorted_paths:
        files = os.listdir(root)

        if file1[0] in files and file2[0] in files and file3[0] in files:
            file1_path = os.path.join(root, file1[0])
            file2_path = os.path.join(root, file2[0])
            file3_path = os.path.join(root, file3[0])

            data1.append(np.load(file1_path, allow_pickle=True))
            data2.append(np.load(file2_path, allow_pickle=True))
            data3.append(np.load(file3_path, allow_pickle=True))
        else:
            file1_path = os.path.join(root, "checkpoint", file1[1])
            file2_path = os.path.join(root, "checkpoint", file2[1])
            file3_path = os.path.join(root, "checkpoint", file3[1])

            data1.append(np.load(file1_path, allow_pickle=True))
            data2.append(np.load(file2_path, allow_pickle=True))
            data3.append(np.load(file3_path, allow_pickle=True))

    return data1, data2, data3


def extract(ROM_framework, type_of_basis, file1=None, file2=None, file3=None):
    base_path = "/Users/shubhadityaburela/Python/Paper4_OCsPOD/OCsPOD/results_advection_new_new/data/Constant_shift/" + ROM_framework + "/" + type_of_basis + "/L1=0.0_L2=0.001/CTC_mask=False/"

    sorted_mode_dirs = get_sorted_mode_dirs(base_path)
    sorted_tol_dirs = get_sorted_tol_dirs(base_path)

    data1_modes, data2_modes, data3_modes = extract_data(sorted_mode_dirs, file1, file2, file3)
    data1_tol, data2_tol, data3_tol = extract_data(sorted_tol_dirs, file1, file2, file3)

    return data1_modes, data2_modes, data3_modes, data1_tol, data2_tol, data3_tol


FOM_J = np.load(
    "/Users/shubhadityaburela/Python/Paper4_OCsPOD/OCsPOD/results_advection_new/data/Constant_shift/FOM/L1=0.0_L2=0.001/CTC_mask=False/J_opt_list_final.npy",
    allow_pickle=True)
FOM_t = np.load(
    "/Users/shubhadityaburela/Python/Paper4_OCsPOD/OCsPOD/results_advection_new/data/Constant_shift/FOM/L1=0.0_L2=0.001/CTC_mask=False/running_time_final.npy",
    allow_pickle=True)

basis_name1 = "primal+adjoint_common_basis"
# case1 = "PODG_FOTR_adaptive"
# case2 = "sPODG_FOTR_adaptive"
case3 = "PODG_FRTO_adaptive"
case4 = "sPODG_FRTO_adaptive"
# ################################### FOTR #########################################
# # PODG
# POD_modes_data_1_fotr, POD_modes_data_2_fotr, POD_modes_data_3_fotr = extract(ROM_framework=case1,
#                                                                               type_of_basis=basis_name1,
#                                                                               file1=["J_opt_FOM_list_final.npy",
#                                                                                      "J_opt_FOM_list.npy"],
#                                                                               file2=["best_details_final.npy",
#                                                                                      "best_details.npy"],
#                                                                               file3=["running_time_final.npy",
#                                                                                      "running_time.npy"]
#                                                                               )
#
# # sPODG
# sPOD_modes_data_1_fotr, sPOD_modes_data_2_fotr, sPOD_modes_data_3_fotr = extract(ROM_framework=case2,
#                                                                                  type_of_basis=basis_name1,
#                                                                                  file1=["J_opt_FOM_list_final.npy",
#                                                                                         "J_opt_FOM_list.npy"],
#                                                                                  file2=["best_details_final.npy",
#                                                                                         "best_details.npy"],
#                                                                                  file3=["running_time_final.npy",
#                                                                                         "running_time.npy"]
#                                                                                  )

################################### FRTO ########################################
# PODG
POD_modes_data_1_frto, POD_modes_data_2_frto, POD_modes_data_3_frto, \
    POD_tol_data_1_frto, POD_tol_data_2_frto, POD_tol_data_3_frto = extract(ROM_framework=case3,
                                                                            type_of_basis=basis_name1,
                                                                            file1=["J_opt_FOM_list_final.npy",
                                                                                   "J_opt_FOM_list.npy"],
                                                                            file2=["best_details_final.npy",
                                                                                   "best_details.npy"],
                                                                            file3=["running_time_final.npy",
                                                                                   "running_time.npy"]
                                                                            )

# sPODG
sPOD_modes_data_1_frto, sPOD_modes_data_2_frto, sPOD_modes_data_3_frto, \
    sPOD_tol_data_1_frto, sPOD_tol_data_2_frto, sPOD_tol_data_3_frto = extract(ROM_framework=case4,
                                                                               type_of_basis=basis_name1,
                                                                               file1=["J_opt_FOM_list_final.npy",
                                                                                      "J_opt_FOM_list.npy"],
                                                                               file2=["best_details_final.npy",
                                                                                      "best_details.npy"],
                                                                               file3=["running_time_final.npy",
                                                                                      "running_time.npy"]
                                                                               )

# Select the FOTR or FRTO
if args.Framework == "FOTR":
    POD_modes_data_3 = POD_modes_data_3_fotr.copy()
    sPOD_modes_data_3 = sPOD_modes_data_3_fotr.copy()
    POD_modes_data_1 = POD_modes_data_1_fotr.copy()
    sPOD_modes_data_1 = sPOD_modes_data_1_fotr.copy()
else:
    POD_modes_data_3 = POD_modes_data_3_frto.copy()
    sPOD_modes_data_3 = sPOD_modes_data_3_frto.copy()
    POD_modes_data_1 = POD_modes_data_1_frto.copy()
    sPOD_modes_data_1 = sPOD_modes_data_1_frto.copy()
    POD_tol_data_3 = POD_tol_data_3_frto.copy()
    sPOD_tol_data_3 = sPOD_tol_data_3_frto.copy()
    POD_tol_data_1 = POD_tol_data_1_frto.copy()
    sPOD_tol_data_1 = sPOD_tol_data_1_frto.copy()

# Interpolate for common grid spec
t_min_1 = min(min([min(t) for t in POD_modes_data_3]), min([min(t) for t in POD_tol_data_3]))
t_max_1 = max(max([max(t) for t in POD_modes_data_3]), max([max(t) for t in POD_tol_data_3]))
t_min_2 = min(min([min(t) for t in sPOD_modes_data_3]), min([min(t) for t in sPOD_tol_data_3]))
t_max_2 = max(max([max(t) for t in sPOD_modes_data_3]), max([max(t) for t in sPOD_tol_data_3]))
t_min_3 = min(FOM_t)
t_max_3 = max(FOM_t)
t_min = min(t_min_1, t_min_2, t_min_3)
t_max = max(t_max_1, t_max_2, t_max_3)
common_t_grid = np.linspace(t_min, t_max, 1000000)

# Interpolate each J array to the common time grid
PODG_J_interpolated = []
for t, J in zip(POD_modes_data_3, POD_modes_data_1):
    len_J = len(J)
    len_t = len(t)
    length = min(len_J, len_t)


    def _const_interp_factory(const_val):
        c = float(np.asarray(const_val).ravel()[0])

        def interp(x):
            xa = np.asarray(x)
            if xa.ndim == 0:
                # scalar input -> scalar output (python float)
                return c
            # preserve input shape for array-like input
            return np.full(xa.shape, c, dtype=float)

        return interp


    if len_t < 2 or len_J < 2:
        # choose value to fill from J's first element (same convention as fill_value=(J[0], J[-1]))
        interp_func = _const_interp_factory(J)
    else:
        interp_func = interp1d(t[:length], J[:length], kind='linear', fill_value=(J[0], J[-1]), bounds_error=False)

    PODG_J_interpolated.append(interp_func(common_t_grid))

for t, J in zip(POD_tol_data_3, POD_tol_data_1):
    len_J = len(J)
    len_t = len(t)
    length = min(len_J, len_t)


    def _const_interp_factory(const_val):
        c = float(np.asarray(const_val).ravel()[0])

        def interp(x):
            xa = np.asarray(x)
            if xa.ndim == 0:
                # scalar input -> scalar output (python float)
                return c
            # preserve input shape for array-like input
            return np.full(xa.shape, c, dtype=float)

        return interp


    if len_t < 2 or len_J < 2:
        # choose value to fill from J's first element (same convention as fill_value=(J[0], J[-1]))
        interp_func = _const_interp_factory(J)
    else:
        interp_func = interp1d(t[:length], J[:length], kind='linear', fill_value=(J[0], J[-1]), bounds_error=False)

    PODG_J_interpolated.append(interp_func(common_t_grid))



# Interpolate each J array to the common time grid
sPODG_J_interpolated = []
for t, J in zip(sPOD_modes_data_3, sPOD_modes_data_1):
    len_J = len(J)
    len_t = len(t)
    length = min(len_J, len_t)
    interp_func = interp1d(t[:length], J[:length], kind='linear', fill_value=(J[0], J[-1]), bounds_error=False)
    sPODG_J_interpolated.append(interp_func(common_t_grid))

for t, J in zip(sPOD_tol_data_3, sPOD_tol_data_1):
    len_J = len(J)
    len_t = len(t)
    length = min(len_J, len_t)
    interp_func = interp1d(t[:length], J[:length], kind='linear', fill_value=(J[0], J[-1]), bounds_error=False)
    sPODG_J_interpolated.append(interp_func(common_t_grid))

# Interpolate each J array to the common time grid
interp_func = interp1d(FOM_t, FOM_J, kind='linear', fill_value=(FOM_J[0], FOM_J[-1]), bounds_error=False)
FOM_interpolated = interp_func(common_t_grid)

PODG_J_min = np.nanmin(np.stack(PODG_J_interpolated), axis=0)
sPODG_J_min = np.nanmin(np.stack(sPODG_J_interpolated), axis=0)

fig = plt.figure(figsize=(9, 6))
ax1 = fig.add_subplot(111)
ax1.loglog(common_t_grid, FOM_interpolated, color='sienna', linestyle='-', label="FOM")
ax1.loglog(common_t_grid, PODG_J_min, color='green', linestyle='--', label="POD-G")
ax1.loglog(common_t_grid, sPODG_J_min, color='red', linestyle='dashdot', label="sPOD-G")
ax1.set_xlabel(r"run time $(\mathrm{s})$")
ax1.set_ylabel(r"$\mathcal{J}$")
ax1.set_title(r"$\mathcal{J}$ vs run time")
ax1.legend()
ax1.grid()

fig.savefig(impath + 'J_vs_runtime_CS_' + args.Framework, dpi=300, transparent=True, format="pdf")
