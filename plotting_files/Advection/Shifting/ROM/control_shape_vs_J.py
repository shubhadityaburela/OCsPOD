import numpy as np
import os
import matplotlib.pyplot as plt
import re
from typing import List

import argparse

parser = argparse.ArgumentParser(description="Input the variables for running the script.")
parser.add_argument("Problem", type=str, choices=["Shifting", "Shifting_3"], help="Choose the problem")
parser.add_argument("n_c", type=int, help="Choose the number of controls")
args = parser.parse_args()

problem = args.Problem


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"]})

SMALL_SIZE = 10
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

if problem == "Shifting":
    impath = "../../../Plots/results_advection/Shifting/ROM/"
    os.makedirs(impath, exist_ok=True)
else:
    impath = "../../../Plots/results_advection/Shifting_3/ROM/"
    os.makedirs(impath, exist_ok=True)

# Regex pattern to extract real numbers from file names
nc_pattern = re.compile(r"\bn_c\s*=\s*(?:\(|\[)?\s*(\d+)\s*(?:\)|\])?\b", re.IGNORECASE)


def get_sorted_mode_dirs(path: str) -> List[str]:
    mode_dirs = []
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if not os.path.isdir(full_path):
            continue
        m = nc_pattern.search(entry)
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

            data1.append(np.load(file1_path, allow_pickle=True)[-1])
            data2.append(np.load(file2_path, allow_pickle=True))
            data3.append(np.load(file3_path, allow_pickle=True))
        else:
            file1_path = os.path.join(root, "checkpoint", file1[1])
            file2_path = os.path.join(root, "checkpoint", file2[1])
            file3_path = os.path.join(root, "checkpoint", file3[1])

            data1.append(np.load(file1_path, allow_pickle=True)[-1])
            data2.append(np.load(file2_path, allow_pickle=True))
            data3.append(np.load(file3_path, allow_pickle=True))

    return data1, data2, data3


def extract(problem, ROM_framework, file1=None, file2=None, file3=None):
    base_path = "/Users/shubhadityaburela/Python/Paper(4.1 + 4.2)_OCsPOD/OCsPOD/results_advection/data/" + problem + "/" + ROM_framework + "/L1=0.0_L2=0.001/"

    sorted_mode_dirs = get_sorted_mode_dirs(base_path)

    data1_modes, data2_modes, data3_modes = extract_data(sorted_mode_dirs, file1, file2, file3)

    return data1_modes, data2_modes, data3_modes


controls_array = np.asarray([4, 10, 16, 22, 28, 32, 36, 42, 52, 62])  # n_c + 1

################################### sPODG #########################################
J_FOM, _, T_FOM = extract(problem=problem,
                          ROM_framework="FOM",
                          file1=["J_opt_list_final.npy", "J_opt_list.npy"],
                          file2=["best_details_final.npy", "best_details.npy"],
                          file3=["running_time_final.npy",
                                 "running_time.npy"]
                          )

J_ROM, _, T_ROM = extract(problem=problem,
                          ROM_framework="sPODG_FRTO",
                          file1=["J_opt_FOM_list_final.npy", "J_opt_FOM_list.npy"],
                          file2=["best_details_final.npy", "best_details.npy"],
                          file3=["running_time_final.npy",
                                 "running_time.npy"]
                          )

fig1 = plt.figure(figsize=(5, 5))
ax1 = fig1.add_subplot(111)
ax1.plot(controls_array, J_FOM, marker="*", label="FOM")
ax1.plot(controls_array, J_ROM, marker="*", label="sPODG")
ax1.set_xlabel(r"number of controls")
ax1.set_ylabel(r"$\mathcal{J}$")
ax1.set_yscale('log')
ax1.grid()
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18))

fig1.tight_layout()
fig1.savefig(impath + problem, dpi=300, transparent=True)

print(J_FOM)
print(J_ROM)


#####################################################################################
controls_array = np.asarray([4, 10, 16, 22, 28, 32, 36, 42, 52, 62]) - 1  # n_c
np.set_printoptions(precision=3, suppress=True)
np.set_printoptions(
    threshold=np.inf,   # print ALL elements, no truncation
    linewidth=np.inf    # do not wrap lines
)
for i, a in enumerate(T_ROM):
    arr = np.asarray(a, dtype=float)  # convert the n x 6 list -> ndarray (raises if non-numeric/inconsistent)
    np.cumsum(arr, axis=0, out=arr)  # column-wise cumulative sum (in-place into arr)
    pre_factor = np.ones_like(arr)
    pre_factor[:, 0] = -1
    T_ROM[i] = arr * pre_factor

for i, a in enumerate(T_FOM):
    arr = np.asarray(a, dtype=float)  # convert the n x 6 list -> ndarray (raises if non-numeric/inconsistent)
    np.cumsum(arr, axis=0, out=arr)  # column-wise cumulative sum (in-place into arr)
    pre_factor = np.ones_like(arr)
    pre_factor[:, 0] = -1
    T_FOM[i] = arr * pre_factor

# Select the best result in mode based study for both sPOD and POD
idx_control = np.where(controls_array == args.n_c)[0]
sPODG_control = T_ROM[idx_control[0]][-1, :]
FOM_control = T_FOM[idx_control[0]][-1, :]

headers = ["Step", "sPODG", "FOM"]
steps = ["Total", "ROM/FOM state solve", "Compute J", "ROM/FOM adjoint solve", "Compute Gradient", "Update control"]
cols = [sPODG_control, FOM_control]
n_iter = [int(T_ROM[idx_control[0]].shape[0]), int(T_FOM[idx_control[0]].shape[0])]

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
    if itr == 1 or itr == 6:
        print(sep)
    print(fmt_row(r))
    itr = itr + 1
print(sep)
