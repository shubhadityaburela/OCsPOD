import numpy as np
import os
import matplotlib.pyplot as plt
import re
from typing import List

import argparse

parser = argparse.ArgumentParser(description="Input the variables for running the script.")
parser.add_argument("Problem", type=str, choices=["Shifting", "Shifting_3"], help="Choose the problem")
parser.add_argument("Adaptivity", type=str, choices=["", "adaptive"],
                    help="Choose if the basis refinement should be adaptive or not")
args = parser.parse_args()

problem = args.Problem
adaptivity = args.Adaptivity


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
    impath = "../../Plots/results_advection_FOTR/Shifting/ROM/"
    os.makedirs(impath, exist_ok=True)
else:
    impath = "../../Plots/results_advection_FOTR/Shifting_3/ROM/"
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


def extract_data(sorted_paths, file1, file2):
    data1 = []
    data2 = []

    for root in sorted_paths:
        files = os.listdir(root)

        if file1[0] in files and file2[0] in files:
            file1_path = os.path.join(root, file1[0])
            file2_path = os.path.join(root, file2[0])

            data1.append(np.load(file1_path, allow_pickle=True)[-1])
            data2.append(np.load(file2_path, allow_pickle=True))
        else:
            file1_path = os.path.join(root, "checkpoint", file1[1])
            file2_path = os.path.join(root, "checkpoint", file2[1])

            data1.append(np.load(file1_path, allow_pickle=True)[-1])
            data2.append(np.load(file2_path, allow_pickle=True))

    return data1, data2


def extract(problem, ROM_framework, type_of_basis, file1=None, file2=None):
    base_path = "/Users/shubhadityaburela/Python/Paper(4.1 + 4.2)_OCsPOD/OCsPOD/results_advection_FOTR/data/" + problem + "/" + ROM_framework + "/" + type_of_basis + "/L1=0.0_L2=0.001/"

    sorted_mode_dirs = get_sorted_mode_dirs(base_path)

    data1_modes, data2_modes = extract_data(sorted_mode_dirs, file1, file2)

    return data1_modes, data2_modes


if problem == "Shifting":
    FOM_J = 8.499
else:
    FOM_J = 25.60

modes_array_PODG = np.asarray([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500])
modes_array_sPODG = np.asarray([2, 5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50])
basis_name1 = "primal_basis"
basis_name2 = "primal+adjoint_common_basis"
if adaptivity == "adaptive":
    case1 = "PODG_FOTR_adaptive"
    case2 = "sPODG_FOTR_adaptive"
    basis_refine = "adaptive"
else:
    case1 = "PODG_FOTR"
    case2 = "sPODG_FOTR"
    basis_refine = "fixed"


################################### PODG #########################################
# Separate basis
POD_modes_data_1s, POD_modes_data_2s = extract(problem=problem,
                                               ROM_framework=case1,
                                               type_of_basis="primal_basis",
                                               file1=["J_opt_FOM_list_final.npy", "J_opt_FOM_list.npy"],
                                               file2=["best_details_final.npy", "best_details.npy"]
                                               )

# Common basis
POD_modes_data_1c, POD_modes_data_2c = extract(problem=problem,
                                               ROM_framework=case1,
                                               type_of_basis="primal+adjoint_common_basis",
                                               file1=["J_opt_FOM_list_final.npy", "J_opt_FOM_list.npy"],
                                               file2=["best_details_final.npy", "best_details.npy"]
                                               )

POD_best_J_s = []
POD_best_J_c = []
for idx, val in enumerate(modes_array_PODG):
    POD_best_J_s.append(POD_modes_data_2s[idx].item()["J"])
    POD_best_J_c.append(POD_modes_data_2c[idx].item()["J"])

################################### sPODG ########################################
# Separate basis / primal basis
sPOD_modes_data_1s, sPOD_modes_data_2s = extract(problem=problem,
                                                 ROM_framework=case2,
                                                 type_of_basis="primal_basis",
                                                 file1=["J_opt_FOM_list_final.npy", "J_opt_FOM_list.npy"],
                                                 file2=["best_details_final.npy", "best_details.npy"]
                                                 )

# Common basis
sPOD_modes_data_1c, sPOD_modes_data_2c = extract(problem=problem,
                                                 ROM_framework=case2,
                                                 type_of_basis="primal+adjoint_common_basis",
                                                 file1=["J_opt_FOM_list_final.npy", "J_opt_FOM_list.npy"],
                                                 file2=["best_details_final.npy", "best_details.npy"]
                                                 )


sPOD_best_J_s = []
sPOD_best_J_c = []
for idx, val in enumerate(modes_array_sPODG):
    sPOD_best_J_s.append(sPOD_modes_data_2s[idx].item()["J"])
    sPOD_best_J_c.append(sPOD_modes_data_2c[idx].item()["J"])



fig1 = plt.figure(figsize=(9, 8))
ax1 = fig1.add_subplot(111)
ax1.axhline(y=FOM_J, color='sienna', linestyle='-', label="FOM")
ax1.plot(modes_array_PODG, POD_modes_data_1s, marker="*", label="PODG" + "_" + basis_name1)
ax1.plot(modes_array_PODG, POD_modes_data_1c, marker="*", label="PODG" + "_" + basis_name2)
ax1.plot(modes_array_sPODG, sPOD_modes_data_1s, marker="*", label="sPODG" + "_" + basis_name1)
ax1.plot(modes_array_sPODG, sPOD_modes_data_1c, marker="*", label="sPODG" + "_" + basis_name2)
ax1.set_xlabel(r"modes")
ax1.set_ylabel(r"$\mathcal{J}$")
ax1.set_yscale('log')
ax1.grid()
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18))


fig1.tight_layout()
fig1.savefig(impath + "Modes_vs_J_" + basis_refine, dpi=300, transparent=True)

print(POD_modes_data_1s)
print(sPOD_modes_data_1s)