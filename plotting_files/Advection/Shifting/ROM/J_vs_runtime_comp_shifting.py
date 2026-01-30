import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
import re
from typing import List, Tuple, Optional
import argparse
from scipy.interpolate import interp1d
import warnings

np.set_printoptions(precision=3, suppress=True)
np.set_printoptions(
    threshold=np.inf,   # print ALL elements, no truncation
    linewidth=np.inf    # do not wrap lines
)


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

# Regex pattern to extract real numbers from file names
modes_pattern = re.compile(r"modes\s*=\s*[\(\[]\s*(\d+)", re.IGNORECASE)
tol_pattern = re.compile(r"tol=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")

parser = argparse.ArgumentParser(description="Input the variables for running the script.")
parser.add_argument("Problem", type=str, choices=["Shifting", "Shifting_3"], help="Choose the problem")
parser.add_argument("PODG_modes", type=int, help="Choose the best PODG modes case")
parser.add_argument("PODG_tol", type=float, help="Choose the best PODG tol case")
parser.add_argument("sPODG_modes", type=int, help="Choose the best sPODG modes case")
parser.add_argument("sPODG_tol", type=float, help="Choose the best sPODG tol case")
args = parser.parse_args()

problem = args.Problem

if problem == "Shifting":
    impath = "../../../Plots/results_advection/Shifting/ROM/"
    os.makedirs(impath, exist_ok=True)
else:
    impath = "../../../Plots/results_advection/Shifting_3/ROM/"
    os.makedirs(impath, exist_ok=True)


def save_fig(filepath, figure=None, **kwargs):
    import tikzplotlib
    import os
    import matplotlib.pyplot as plt

    ## split extension
    fpath = os.path.splitext(filepath)[0]
    ## get figure handle
    if figure is None:
        figure = plt.gcf()
    figure.savefig(fpath + ".png", dpi=200, transparent=True)
    tikzplotlib.save(
        figure=figure,
        filepath=fpath + ".tex",
        axis_height='\\figureheight',
        axis_width='\\figurewidth',
        override_externals=True,
        **kwargs
    )


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


def extract_data(sorted_paths: List[str],
                 file1: Tuple[str, str],
                 file2: Tuple[str, str],
                 file3: Tuple[str, str]):
    """
    For each directory in sorted_paths:
      - try to load file1[0] and file2[0] from the directory
      - if not present, try checkpoint/file1[1] and checkpoint/file2[1]
    Returns two lists of loaded arrays (data1, data2), preserving ordering of sorted_paths.
    """
    data1: List[np.ndarray] = []
    data2: List[np.ndarray] = []
    data3: List[np.ndarray] = []

    for root in sorted_paths:
        rootp = Path(root)
        # check presence of top-level files
        top_f1 = rootp / file1[0]
        top_f2 = rootp / file2[0]
        top_f3 = rootp / file3[0]

        if top_f1.exists() and top_f2.exists() and top_f3.exists():
            f1_path = top_f1
            f2_path = top_f2
            f3_path = top_f3
        else:
            chk_dir = rootp / "checkpoint"
            chk_f1 = chk_dir / file1[1]
            chk_f2 = chk_dir / file2[1]
            chk_f3 = chk_dir / file3[1]
            if chk_f1.exists() and chk_f2.exists() and chk_f3.exists():
                f1_path = chk_f1
                f2_path = chk_f2
                f3_path = chk_f3
            else:
                warnings.warn(f"Missing files for directory '{root}': "
                              f"checked {top_f1.name}/{top_f2.name} and {chk_f1 if 'chk_f1' in locals() else 'checkpoint/...'}",
                              RuntimeWarning)
                # skip this directory
                continue

        try:
            arr1 = np.load(f1_path, allow_pickle=True)
        except Exception as e:
            warnings.warn(f"Failed to load {f1_path!s}: {e}", RuntimeWarning)
            continue

        try:
            arr2 = np.load(f2_path, allow_pickle=True)
        except Exception as e:
            warnings.warn(f"Failed to load {f2_path!s}: {e}", RuntimeWarning)
            continue

        try:
            arr3 = np.load(f3_path, allow_pickle=True)
        except Exception as e:
            warnings.warn(f"Failed to load {f3_path!s}: {e}", RuntimeWarning)
            continue

        data1.append(arr1)
        data2.append(arr2)
        data3.append(arr3)

    return data1, data2, data3


def extract(problem: str,
            ROM_framework: str,
            type_of_basis: str,
            file1: Optional[Tuple[str, str]] = None,
            file2: Optional[Tuple[str, str]] = None,
            file3: Optional[Tuple[str, str]] = None):
    """
    Build base path, get sorted tol directories, and extract data.
    file1 and file2 are tuples: (primary_filename, checkpoint_filename)
    """
    if file1 is None or file2 is None or file3 is None:
        raise ValueError("file1, file2 and file3 must be provided as (primary_name, checkpoint_name) tuples")

    base_path = os.path.join(
        "/Users/shubhadityaburela/Python/Paper(4.1 + 4.2)_OCsPOD/OCsPOD/results_advection/data/", problem,
        ROM_framework, type_of_basis, "L1=0.0_L2=0.001"
    )

    sorted_tol_dirs = get_sorted_tol_dirs(base_path)
    sorted_mode_dirs = get_sorted_mode_dirs(base_path)

    data1_modes, data2_modes, data3_modes = extract_data(sorted_mode_dirs, file1, file2, file3)
    data1_tol, data2_tol, data3_tol = extract_data(sorted_tol_dirs, file1, file2, file3)
    return data1_modes, data2_modes, data3_modes, data1_tol, data2_tol, data3_tol


tmp = "results_advection"

if problem == "Shifting":
    FOM_J = np.load(
        "/Users/shubhadityaburela/Python/Paper(4.1 + 4.2)_OCsPOD/OCsPOD/" + tmp + "/data/Shifting/FOM/L1=0.0_L2=0.001/n_c=41/J_opt_list_final.npy",
        allow_pickle=True)
    FOM_t = np.load(
        "/Users/shubhadityaburela/Python/Paper(4.1 + 4.2)_OCsPOD/OCsPOD/" + tmp + "/data/Shifting/FOM/L1=0.0_L2=0.001/n_c=41/running_time_final.npy",
        allow_pickle=True)
else:
    FOM_J = np.load(
        "/Users/shubhadityaburela/Python/Paper(4.1 + 4.2)_OCsPOD/OCsPOD/" + tmp + "/data/Shifting_3/FOM/L1=0.0_L2=0.001/n_c=41/J_opt_list_final.npy",
        allow_pickle=True)
    FOM_t = np.load(
        "/Users/shubhadityaburela/Python/Paper(4.1 + 4.2)_OCsPOD/OCsPOD/" + tmp + "/data/Shifting_3/FOM/L1=0.0_L2=0.001/n_c=41/running_time_final.npy",
        allow_pickle=True)

case3 = "PODG_FRTO_adaptive"
case4 = "sPODG_FRTO_adaptive"

modes_array_PODG = np.asarray([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500])
modes_array_sPODG = np.asarray([2, 5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50])
tol_array_PODG = np.asarray([1e-10, 5e-10, 1e-09, 5e-09, 1e-08, 5e-08, 1e-07, 5e-07, 1e-06, 5e-06, 1e-05, 5e-05, 1e-04, 5e-04, 1e-03, 5e-03, 1e-02])
tol_array_sPODG = np.asarray([1e-10, 5e-10, 1e-09, 5e-09, 1e-08, 5e-08, 1e-07, 5e-07, 1e-06, 5e-06, 1e-05, 5e-05, 1e-04, 5e-04, 1e-03, 5e-03, 1e-02])
################################### FRTO ########################################
# PODG
POD_modes_data_1_frto, POD_modes_data_2_frto, POD_modes_data_3_frto, \
    POD_tol_data_1_frto, POD_tol_data_2_frto, POD_tol_data_3_frto = extract(problem=problem,
                                                                            ROM_framework=case3,
                                                                            type_of_basis="primal_basis",
                                                                            file1=["J_opt_FOM_list_final.npy",
                                                                                   "J_opt_FOM_list.npy"],
                                                                            file2=["best_details_final.npy",
                                                                                   "best_details.npy"],
                                                                            file3=["running_time_final.npy",
                                                                                   "running_time.npy"]
                                                                            )

# sPODG
sPOD_modes_data_1_frto, sPOD_modes_data_2_frto, sPOD_modes_data_3_frto, \
    sPOD_tol_data_1_frto, sPOD_tol_data_2_frto, sPOD_tol_data_3_frto = extract(problem=problem,
                                                                               ROM_framework=case4,
                                                                               type_of_basis="primal_basis",
                                                                               file1=["J_opt_FOM_list_final.npy",
                                                                                      "J_opt_FOM_list.npy"],
                                                                               file2=["best_details_final.npy",
                                                                                      "best_details.npy"],
                                                                               file3=["running_time_final.npy",
                                                                                      "running_time.npy"]
                                                                               )


POD_modes_data_3 = POD_modes_data_3_frto.copy()
sPOD_modes_data_3 = sPOD_modes_data_3_frto.copy()
POD_modes_data_1 = POD_modes_data_1_frto.copy()
sPOD_modes_data_1 = sPOD_modes_data_1_frto.copy()

POD_tol_data_3 = POD_tol_data_3_frto.copy()
sPOD_tol_data_3 = sPOD_tol_data_3_frto.copy()
POD_tol_data_1 = POD_tol_data_1_frto.copy()
sPOD_tol_data_1 = sPOD_tol_data_1_frto.copy()


for coll in (POD_modes_data_3, POD_tol_data_3, sPOD_modes_data_3, sPOD_tol_data_3):
    for i, a in enumerate(coll):
        arr = np.asarray(a, dtype=float)  # convert the n x 7 list -> ndarray (raises if non-numeric/inconsistent)
        np.cumsum(arr, axis=0, out=arr)  # column-wise cumulative sum (in-place into arr)
        pre_factor = np.ones_like(arr)
        pre_factor[:, 0] = -1
        coll[i] = arr * pre_factor


FOM_t = np.asarray(FOM_t, dtype=float)  # convert the n x 6 list -> ndarray (raises if non-numeric/inconsistent)
np.cumsum(FOM_t, axis=0, out=FOM_t)  # column-wise cumulative sum (in-place into arr)
pre_factor = np.ones_like(FOM_t)
pre_factor[:, 0] = -1
FOM_t = FOM_t * pre_factor

# Interpolate for common grid spec
t_min_1 = min(min([min(t[:, 0]) for t in POD_modes_data_3]), min([min(t[:, 0]) for t in POD_tol_data_3]))
t_max_1 = max(max([max(t[:, 0]) for t in POD_modes_data_3]), max([max(t[:, 0]) for t in POD_tol_data_3]))
t_min_2 = min(min([min(t[:, 0]) for t in sPOD_modes_data_3]), min([min(t[:, 0]) for t in sPOD_tol_data_3]))
t_max_2 = max(max([max(t[:, 0]) for t in sPOD_modes_data_3]), max([max(t[:, 0]) for t in sPOD_tol_data_3]))
t_min_3 = min(FOM_t[:, 0])
t_max_3 = max(FOM_t[:, 0])
t_min = min(t_min_1, t_min_2, t_min_3)
t_max = max(t_max_1, t_max_2, t_max_3)
common_t_grid = np.linspace(t_min, t_max, 1000)

# Interpolate each J array to the common time grid
PODG_J_interpolated = []
for t, J in zip(POD_modes_data_3, POD_modes_data_1):
    tt = [row[0] for row in t]
    len_J = len(J)
    len_t = len(tt)
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
        interp_func = interp1d(tt[:length], J[:length], kind='linear', fill_value=(J[0], J[-1]), bounds_error=False)

    PODG_J_interpolated.append(interp_func(common_t_grid))

for t, J in zip(POD_tol_data_3, POD_tol_data_1):
    tt = [row[0] for row in t]
    len_J = len(J)
    len_t = len(tt)
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
        interp_func = interp1d(tt[:length], J[:length], kind='linear', fill_value=(J[0], J[-1]), bounds_error=False)

    PODG_J_interpolated.append(interp_func(common_t_grid))

# Interpolate each J array to the common time grid
sPODG_J_interpolated = []
for t, J in zip(sPOD_modes_data_3, sPOD_modes_data_1):
    tt = [row[0] for row in t]
    len_J = len(J)
    len_t = len(tt)
    length = min(len_J, len_t)
    interp_func = interp1d(tt[:length], J[:length], kind='linear', fill_value=(J[0], J[-1]), bounds_error=False)
    sPODG_J_interpolated.append(interp_func(common_t_grid))

for t, J in zip(sPOD_tol_data_3, sPOD_tol_data_1):
    tt = [row[0] for row in t]
    len_J = len(J)
    len_t = len(tt)
    length = min(len_J, len_t)
    interp_func = interp1d(tt[:length], J[:length], kind='linear', fill_value=(J[0], J[-1]), bounds_error=False)
    sPODG_J_interpolated.append(interp_func(common_t_grid))

# Interpolate each J array to the common time grid
interp_func = interp1d(FOM_t[:, 0], FOM_J, kind='linear', fill_value=(FOM_J[0], FOM_J[-1]), bounds_error=False)
FOM_interpolated = interp_func(common_t_grid)

PODG_J_min = np.nanmin(np.stack(PODG_J_interpolated), axis=0)
sPODG_J_min = np.nanmin(np.stack(sPODG_J_interpolated), axis=0)

fig = plt.figure(figsize=(9, 6))
ax1 = fig.add_subplot(111)
ax1.semilogx(common_t_grid, FOM_interpolated, color='sienna', linestyle='-', label="FOM")
ax1.semilogx(common_t_grid, PODG_J_min, color='green', linestyle='--', label="POD-G")
ax1.semilogx(common_t_grid, sPODG_J_min, color='red', linestyle='dashdot', label="sPOD-G")
ax1.set_xlabel(r"run time $(\mathrm{s})$")
ax1.set_ylabel(r"$\mathcal{J}$")
ax1.set_title(r"$\mathcal{J}$ vs run time")
ax1.legend()
ax1.grid()

fig.savefig(impath + 'J_vs_runtime_' + problem + '.pdf', dpi=300, transparent=True, format="pdf")
save_fig(impath + 'J_vs_runtime_' + problem, fig)


###########################################################################################################
# Select the best result in mode based study for both sPOD and POD
idx_PODG_modes = np.where(modes_array_PODG == args.PODG_modes)[0]
idx_PODG_tol = np.where(tol_array_PODG == args.PODG_tol)[0]
idx_sPODG_modes = np.where(modes_array_sPODG == args.sPODG_modes)[0]
idx_sPODG_tol = np.where(tol_array_sPODG == args.sPODG_tol)[0]


if POD_modes_data_1[idx_PODG_modes[0]][-1] < POD_tol_data_1[idx_PODG_tol[0]][-1]:
    print("Select MODES test for POD-G")
else:
    print("Select TOL test for POD-G")

if sPOD_modes_data_1[idx_sPODG_modes[0]][-1] < sPOD_tol_data_1[idx_sPODG_tol[0]][-1]:
    print("Select MODES test for sPOD-G")
else:
    print("Select TOL test for sPOD-G")

best_PODG_modes = POD_modes_data_3[idx_PODG_modes[0]][-1, :]
best_PODG_tol = POD_tol_data_3[idx_PODG_tol[0]][-1, :]
best_sPODG_modes = sPOD_modes_data_3[idx_sPODG_modes[0]][-1, :]
best_sPODG_tol = sPOD_tol_data_3[idx_sPODG_tol[0]][-1, :]
best_FOM = FOM_t[-1, :]

headers = ["Step", "PODG_modes", "sPODG_modes", "PODG_tol", "sPODG_tol"]
steps = ["Total", "RB construction", "ROM state solve", "Compute J", "ROM adjoint solve", "Compute Gradient", "Update control"]
cols = [best_PODG_modes,  best_sPODG_modes, best_PODG_tol, best_sPODG_tol]
n_iter = [int(POD_modes_data_3[idx_PODG_modes[0]].shape[0]), int(sPOD_modes_data_3[idx_sPODG_modes[0]].shape[0]),
          int(POD_tol_data_3[idx_PODG_tol[0]].shape[0]), int(sPOD_tol_data_3[idx_sPODG_tol[0]].shape[0]),
          int(FOM_t.shape[0])]
headers = headers + ["FOM"]

# Build the FOM column aligned with steps: replace RB construction (index 1) by 0.0
if len(best_FOM) != len(steps) - 1:
    raise ValueError(f"best_FOM must have {len(steps)-1} values (one per step except 'RB construction')")


# Map the 6 provided FOM values into the 7-step table, inserting 0.0 at index 1
FOM_col = [best_FOM[i] if i < 1 else (0.0 if i == 1 else best_FOM[i-1]) for i in range(len(steps))]

# Combine columns (keep numeric columns as lists of floats)
cols = [best_PODG_modes, best_sPODG_modes, best_PODG_tol, best_sPODG_tol, FOM_col]

# Build rows as formatted strings (4 decimal places)
rows_main = [
    [steps[i]] + [f"{col[i]:.4f}" for col in cols]
    for i in range(len(steps))
]

# Build the n_iter final row (format numeric entries to 4 decimals)
n_iter_row = ["n_iter"] + [f"{v}" for v in n_iter]

# Full rows including the final n_iter row
rows = rows_main + [n_iter_row]

# Compute column widths (consider headers and all rows)
widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]

def fmt_row(row):
    return "| " + " | ".join(f"{v:<{w}}" for v, w in zip(row, widths)) + " |"

sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"

# Print table
print(sep)
print(fmt_row(headers))
print(sep)
itr = 0
for r in rows:
    if itr == 1 or itr == 7:
        print(sep)
    print(fmt_row(r))
    itr = itr + 1
print(sep)


# print("\n\n")
# # Combine columns (keep numeric columns as lists of floats)
# cols = [best_PODG_modes / n_iter[0], best_sPODG_modes / n_iter[1], best_PODG_tol / n_iter[2],
#         best_sPODG_tol / n_iter[3], np.asarray(FOM_col) / n_iter[4]]
# # Build rows as formatted strings (4 decimal places)
# rows_main = [
#     [steps[i]] + [f"{col[i]:.4f}" for col in cols]
#     for i in range(len(steps))
# ]
# # Full rows including the final n_iter row
# rows = rows_main + [n_iter_row]
# # Print table
# print(sep)
# print(fmt_row(headers))
# print(sep)
# itr = 0
# for r in rows:
#     if itr == 1 or itr == 7:
#         print(sep)
#     if itr == 7:
#         pass
#     else:
#         print(fmt_row(r))
#     itr = itr + 1