import numpy as np
import os
import matplotlib.pyplot as plt
import re
from typing import List

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

impath = "../../../Plots/advection_plot/Constant_shift/ROM_in_Armijo/"
os.makedirs(impath, exist_ok=True)

# Regex pattern to extract real numbers from file names
modes_pattern = re.compile(r"modes\s*=\s*[\(\[]\s*(\d+)", re.IGNORECASE)


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
    base_path = "/Users/shubhadityaburela/Python/Paper4_OCsPOD/OCsPOD/results_advection/Constant_shift/ROM_in_Armijo/data/" + ROM_framework + "/" + type_of_basis + "/L1=0.0_L2=0.001/CTC_mask=False/"

    sorted_mode_dirs = get_sorted_mode_dirs(base_path)

    data1_modes, data2_modes, data3_modes = extract_data(sorted_mode_dirs, file1, file2, file3)

    return data1_modes, data2_modes, data3_modes


FOM_J = np.load(
    "/Users/shubhadityaburela/Python/Paper4_OCsPOD/OCsPOD/results_advection/Constant_shift/ROM_in_Armijo/data/FOM/L1=0.0_L2=0.001/CTC_mask=False/J_opt_list_final.npy", allow_pickle=True)
FOM_t = np.load(
    "/Users/shubhadityaburela/Python/Paper4_OCsPOD/OCsPOD/results_advection/Constant_shift/ROM_in_Armijo/data/FOM/L1=0.0_L2=0.001/CTC_mask=False/running_time_final.npy", allow_pickle=True)

modes_array_PODG = np.asarray([10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
modes_array_sPODG = np.asarray([2, 5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50])
basis_name1 = "separate_basis"
basis_name2 = "primal+adjoint_common_basis"
case1 = "PODG_FOTR_FA"
case2 = "sPODG_FOTR_FA"
################################### PODG_FOTR_FA #########################################
# Separate basis / primal basis
POD_modes_data_1s, POD_modes_data_2s, POD_modes_data_3s = extract(ROM_framework=case1,
                                                                  type_of_basis=basis_name1,
                                                                  file1=["J_opt_list_final.npy", "J_opt_list.npy"],
                                                                  file2=["best_details_final.npy", "best_details.npy"],
                                                                  file3=["running_time_final.npy", "running_time.npy"]
                                                                  )
# Common basis
POD_modes_data_1c, POD_modes_data_2c, POD_modes_data_3c = extract(ROM_framework=case1,
                                                                  type_of_basis=basis_name2,
                                                                  file1=["J_opt_list_final.npy", "J_opt_list.npy"],
                                                                  file2=["best_details_final.npy", "best_details.npy"],
                                                                  file3=["running_time_final.npy", "running_time.npy"]
                                                                  )

################################### sPODG_FOTR_RA ########################################
# Separate basis / primal basis
sPOD_modes_data_1s, sPOD_modes_data_2s, sPOD_modes_data_3s = extract(ROM_framework=case2,
                                                                     type_of_basis=basis_name1,
                                                                     file1=["J_opt_list_final.npy", "J_opt_list.npy"],
                                                                     file2=["best_details_final.npy",
                                                                            "best_details.npy"],
                                                                     file3=["running_time_final.npy",
                                                                            "running_time.npy"]
                                                                     )
# Common basis
sPOD_modes_data_1c, sPOD_modes_data_2c, sPOD_modes_data_3c = extract(ROM_framework=case2,
                                                                     type_of_basis=basis_name2,
                                                                     file1=["J_opt_list_final.npy", "J_opt_list.npy"],
                                                                     file2=["best_details_final.npy",
                                                                            "best_details.npy"],
                                                                     file3=["running_time_final.npy",
                                                                            "running_time.npy"]
                                                                     )
# If the timing is not cumulatively summed then do
np.cumsum(FOM_t, out=FOM_t)
for a in POD_modes_data_3s:
    np.cumsum(a, out=a)
for a in sPOD_modes_data_3s:
    np.cumsum(a, out=a)

# Interpolate for common grid spec (ONLY FOR SEPARATE BASIS)
t_min_1 = min([min(t) for t in POD_modes_data_3s])
t_max_1 = max([max(t) for t in POD_modes_data_3s])
t_min_2 = min([min(t) for t in sPOD_modes_data_3s])
t_max_2 = max([max(t) for t in sPOD_modes_data_3s])
t_min_3 = min(FOM_t)
t_max_3 = max(FOM_t)
t_min = min(t_min_1, t_min_2, t_min_3)
t_max = max(t_max_1, t_max_2, t_max_3)
common_t_grid = np.linspace(t_min, t_max, 1000000)

# Interpolate each J array to the common time grid
PODG_J_interpolated = []
for t, J in zip(POD_modes_data_3s, POD_modes_data_1s):
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
for t, J in zip(sPOD_modes_data_3s, sPOD_modes_data_1s):
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

fig.savefig(impath + 'J_vs_runtime_ROMInArmijo', dpi=300, transparent=True)