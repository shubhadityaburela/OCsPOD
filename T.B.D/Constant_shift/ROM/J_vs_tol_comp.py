from pathlib import Path

import numpy as np
import os
import matplotlib.pyplot as plt
import re
from typing import List, Tuple, Optional
import warnings
import argparse

parser = argparse.ArgumentParser(description="Input the variables for running the script.")
parser.add_argument("Framework", type=str, choices=["FOTR", "FRTO"], help="Choose the framework")
parser.add_argument("Adaptivity", type=str, choices=["", "adaptive"],
                    help="Choose if the basis refinement should be adaptive or not")
args = parser.parse_args()

framework = args.Framework
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

impath = "../../../Plots/advection_plot/Constant_shift/ROM/"
os.makedirs(impath, exist_ok=True)

# Regex pattern to extract real numbers from file names
tol_pattern = re.compile(r"tol=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")


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


def extract_data(sorted_paths: List[str],
                 file1: Tuple[str, str],
                 file2: Tuple[str, str],
                 file3: Tuple[Tuple[str, str], Tuple[str, str]]):
    """
    For each directory in sorted_paths:
      - try to load file1[0] and file2[0] from the directory
      - if not present, try checkpoint/file1[1] and checkpoint/file2[1]
    Returns two lists of loaded arrays (data1, data2), preserving ordering of sorted_paths.
    """
    data1: List[np.ndarray] = []
    data2: List[np.ndarray] = []
    data3: List[List[np.ndarray, np.ndarray]] = []

    for root in sorted_paths:
        rootp = Path(root)
        # check presence of top-level files
        top_f1 = rootp / file1[0]
        top_f2 = rootp / file2[0]
        top_f31 = rootp / file3[0][0]
        if file3[0][1] is not None: top_f32 = rootp / file3[0][1]

        if top_f1.exists() and top_f2.exists() and top_f31.exists():
            f1_path = top_f1
            f2_path = top_f2
            f31_path = top_f31
            if file3[0][1] is not None: f32_path = top_f32
        else:
            chk_dir = rootp / "checkpoint"
            chk_f1 = chk_dir / file1[1]
            chk_f2 = chk_dir / file2[1]
            chk_f31 = chk_dir / file3[1][0]
            if file3[1][1] is not None: chk_f32 = chk_dir / file3[1][1]
            if chk_f1.exists() and chk_f2.exists() and chk_f31.exists():
                f1_path = chk_f1
                f2_path = chk_f2
                f31_path = chk_f31
                if file3[1][1] is not None: f32_path = chk_f32
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
            arr31 = np.load(f31_path, allow_pickle=True)
        except Exception as e:
            warnings.warn(f"Failed to load {f31_path!s}: {e}", RuntimeWarning)
            continue

        if file3[0][1] is not None or file3[1][1] is not None:
            arr32 = np.load(f32_path, allow_pickle=True)
        else:
            arr32 = None

        # preserve your original behaviour: take last element of arr1 if possible
        try:
            # if arr1 is array-like and has indexable last element
            data1.append(arr1[-1])
        except Exception:
            # fallback: append entire thing
            data1.append(arr1)

        data2.append(arr2)
        data3.append([arr31, arr32])

    return data1, data2, data3


def extract(ROM_framework: str,
            type_of_basis: str,
            file1: Optional[Tuple[str, str]] = None,
            file2: Optional[Tuple[str, str]] = None,
            file3: Optional[Tuple[Tuple[str, str], Tuple[str, str]]] = None):
    """
    Build base path, get sorted tol directories, and extract data.
    file1 and file2 are tuples: (primary_filename, checkpoint_filename)
    """
    if file1 is None or file2 is None or file3 is None:
        raise ValueError("file1, file2 and file3 must be provided as (primary_name, checkpoint_name) tuples")

    base_path = os.path.join(
        "/Users/shubhadityaburela/Python/Paper4_OCsPOD/OCsPOD/results_advection_new_new/data/Constant_shift",
        ROM_framework, type_of_basis, "L1=0.0_L2=0.001", "CTC_mask=False"
    )

    sorted_tol_dirs = get_sorted_tol_dirs(base_path)

    data1_tol, data2_tol, data3_tol = extract_data(sorted_tol_dirs, file1, file2, file3)
    return data1_tol, data2_tol, data3_tol


FOM_J = 2.629e-02
tol_array_PODG = np.asarray([1e-06, 5e-06, 1e-05, 5e-05, 1e-04, 5e-04, 1e-03, 5e-03, 1e-02])
tol_array_sPODG = np.asarray([1e-06, 5e-06, 1e-05, 5e-05, 1e-04, 5e-04, 1e-03, 5e-03, 1e-02])
if framework == "FOTR":
    basis_name1 = "separate_basis"
    basis_name2 = "primal+adjoint_common_basis"

    modes_name1 = "trunc_modes_list_p_final.npy"
    modes_name2 = "trunc_modes_list_a_final.npy"
    modes_name3 = "trunc_modes_p.npy"
    modes_name4 = "trunc_modes_a.npy"
    if adaptivity == "adaptive":
        case1 = "PODG_FOTR_adaptive"
        case2 = "sPODG_FOTR_adaptive"
        basis_refine = "adaptive"
    else:
        case1 = "PODG_FOTR"
        case2 = "sPODG_FOTR"
        basis_refine = "fixed"
elif framework == "FRTO":
    basis_name1 = "primal_basis"
    basis_name2 = "primal+adjoint_common_basis"

    modes_name1 = "trunc_modes_list_final.npy"
    modes_name2 = None
    modes_name3 = "trunc_modes.npy"
    modes_name4 = None
    if adaptivity == "adaptive":
        case1 = "PODG_FRTO_adaptive"
        case2 = "sPODG_FRTO_adaptive"
        basis_refine = "adaptive"
    else:
        case1 = "PODG_FRTO"
        case2 = "sPODG_FRTO"
        basis_refine = "fixed"

################################### PODG_FOTR#########################################
# Separate basis / primal basis
POD_tol_data_1s, POD_tol_data_2s, POD_tol_data_3s = extract(ROM_framework=case1,
                                                            type_of_basis=basis_name1,
                                                            file1=["J_opt_FOM_list_final.npy", "J_opt_FOM_list.npy"],
                                                            file2=["best_details_final.npy", "best_details.npy"],
                                                            file3=[[modes_name1, modes_name2],
                                                                   [modes_name3, modes_name4]]
                                                            )
# Common basis
POD_tol_data_1c, POD_tol_data_2c, POD_tol_data_3c = extract(ROM_framework=case1,
                                                            type_of_basis=basis_name2,
                                                            file1=["J_opt_FOM_list_final.npy", "J_opt_FOM_list.npy"],
                                                            file2=["best_details_final.npy", "best_details.npy"],
                                                            file3=[[modes_name1, modes_name2],
                                                                   [modes_name3, modes_name4]]
                                                            )
POD_best_J_s = []
POD_best_J_c = []
for idx, val in enumerate(tol_array_PODG):
    POD_best_J_s.append(POD_tol_data_2s[idx].item()["J"])
    POD_best_J_c.append(POD_tol_data_2c[idx].item()["J"])

################################### sPODG_FOTR ########################################
# Separate basis / primal basis
sPOD_tol_data_1s, sPOD_tol_data_2s, sPOD_tol_data_3s = extract(ROM_framework=case2,
                                                               type_of_basis=basis_name1,
                                                               file1=["J_opt_FOM_list_final.npy", "J_opt_FOM_list.npy"],
                                                               file2=["best_details_final.npy", "best_details.npy"],
                                                               file3=[[modes_name1, modes_name2],
                                                                      [modes_name3, modes_name4]]
                                                               )
# Common basis
sPOD_tol_data_1c, sPOD_tol_data_2c, sPOD_tol_data_3c = extract(ROM_framework=case2,
                                                               type_of_basis=basis_name2,
                                                               file1=["J_opt_FOM_list_final.npy", "J_opt_FOM_list.npy"],
                                                               file2=["best_details_final.npy", "best_details.npy"],
                                                               file3=[[modes_name1, modes_name2],
                                                                      [modes_name3, modes_name4]]
                                                               )
sPOD_best_J_s = []
sPOD_best_J_c = []
for idx, val in enumerate(tol_array_sPODG):
    sPOD_best_J_s.append(sPOD_tol_data_2s[idx].item()["J"])
    sPOD_best_J_c.append(sPOD_tol_data_2c[idx].item()["J"])

fig1 = plt.figure(figsize=(15, 5))
ax1 = fig1.add_subplot(121)
ax1.axhline(y=FOM_J, color='sienna', linestyle='-', label="FOM")
ax1.plot(tol_array_PODG, POD_tol_data_1s, marker="*", label=case1 + "_" + basis_name1)
ax1.plot(tol_array_PODG, POD_tol_data_1c, marker="*", label=case1 + "_" + basis_name2)
ax1.plot(tol_array_sPODG, sPOD_tol_data_1s, marker="*", label=case2 + "_" + basis_name1)
ax1.plot(tol_array_sPODG, sPOD_tol_data_1c, marker="*", label=case2 + "_" + basis_name2)
ax1.set_xlabel(r"modes")
ax1.set_ylabel(r"$\mathcal{J}$")
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.grid()
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18))

ax2 = fig1.add_subplot(122)
ax2.axhline(y=FOM_J, color='sienna', linestyle='-', label="FOM")
ax2.plot(tol_array_PODG, POD_best_J_s, marker="*", label=case1 + "_" + basis_name1 + "(best)")
ax2.plot(tol_array_PODG, POD_best_J_c, marker="*", label=case1 + "_" + basis_name2 + "(best)")
ax2.plot(tol_array_sPODG, sPOD_best_J_s, marker="*", label=case2 + "_" + basis_name1 + "(best)")
ax2.plot(tol_array_sPODG, sPOD_best_J_c, marker="*", label=case2 + "_" + basis_name2 + "(best)")
ax2.set_xlabel(r"modes")
ax2.set_ylabel(r"$\mathcal{J}$")
ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.grid()
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18))

fig1.tight_layout()
fig1.savefig(impath + "Comparing_" + framework + "_" + basis_refine + "_tol", dpi=300, transparent=True)


print(POD_tol_data_1s)
print(POD_tol_data_1c)
print(sPOD_tol_data_1s)
print(sPOD_tol_data_1c)


if framework == "FOTR":
    print("-----------------------PODG---------------------------")
    for idx, val in enumerate(tol_array_PODG):
        print(
            f"Separate basis, tolerance = {val}, with "
            f"Nm_avg_p: {int(sum(POD_tol_data_3s[idx][0]) / len(POD_tol_data_3s[idx][0]))}, "
            f"Nm_avg_a: {int(sum(POD_tol_data_3s[idx][1]) / len(POD_tol_data_3s[idx][1]))}")
    print("-------------------------------------")
    # Tolerance values for table
    for idx, val in enumerate(tol_array_PODG):
        print(
            f"Common basis, tolerance = {val}, with "
            f"Nm_avg_p: {int(sum(POD_tol_data_3c[idx][0]) / len(POD_tol_data_3c[idx][0]))}, "
            f"Nm_avg_a: {int(sum(POD_tol_data_3c[idx][1]) / len(POD_tol_data_3c[idx][1]))}")

    print("-----------------------sPODG---------------------------")
    for idx, val in enumerate(tol_array_sPODG):
        print(
            f"Separate basis, tolerance = {val}, with "
            f"Nm_avg_p: {int(sum(sPOD_tol_data_3s[idx][0]) / len(sPOD_tol_data_3s[idx][0]))}, "
            f"Nm_avg_a: {int(sum(sPOD_tol_data_3s[idx][1]) / len(sPOD_tol_data_3s[idx][1]))}")
    print("-------------------------------------")
    # Tolerance values for table
    for idx, val in enumerate(tol_array_sPODG):
        print(
            f"Common basis, tolerance = {val}, with "
            f"Nm_avg_p: {int(sum(sPOD_tol_data_3c[idx][0]) / len(sPOD_tol_data_3c[idx][0]))}, "
            f"Nm_avg_a: {int(sum(sPOD_tol_data_3c[idx][1]) / len(sPOD_tol_data_3c[idx][1]))}")
elif framework == "FRTO":
    print("-----------------------PODG---------------------------")
    for idx, val in enumerate(tol_array_PODG):
        print(
            f"Separate basis, tolerance = {val}, with "
            f"Nm_avg_p: {int(sum(POD_tol_data_3s[idx][0]) / len(POD_tol_data_3s[idx][0]))}")
    print("-------------------------------------")
    # Tolerance values for table
    for idx, val in enumerate(tol_array_PODG):
        print(
            f"Common basis, tolerance = {val}, with "
            f"Nm_avg_p: {int(sum(POD_tol_data_3c[idx][0]) / len(POD_tol_data_3c[idx][0]))}")

    print("-----------------------sPODG---------------------------")
    for idx, val in enumerate(tol_array_sPODG):
        print(
            f"Separate basis, tolerance = {val}, with "
            f"Nm_avg_p: {int(sum(sPOD_tol_data_3s[idx][0]) / len(sPOD_tol_data_3s[idx][0]))}")
    print("-------------------------------------")
    # Tolerance values for table
    for idx, val in enumerate(tol_array_sPODG):
        print(
            f"Common basis, tolerance = {val}, with "
            f"Nm_avg_p: {int(sum(sPOD_tol_data_3c[idx][0]) / len(sPOD_tol_data_3c[idx][0]))}")