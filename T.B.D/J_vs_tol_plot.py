import numpy as np
import os
import matplotlib.pyplot as plt
import re

import argparse

parser = argparse.ArgumentParser(description="Input the variables for running the script.")
parser.add_argument("problem", type=int, choices=[1, 2, 3], help="Specify the problem number (1, 2, or 3)")
parser.add_argument("regularize", type=str, choices=["L1", "L2"], help="Choose between L1 and L2")
args = parser.parse_args()
problem = args.problem
regularize = args.regularize

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


if regularize == "L2":
    impath = "./L2_results_plot/"
    os.makedirs(impath, exist_ok=True)
elif regularize == "L1":
    impath = "./L1_results_plot/"
    os.makedirs(impath, exist_ok=True)

# Regex pattern to extract real numbers from file names
modes_pattern = re.compile(r"modes=\[(\d+)\]")
tol_pattern = re.compile(r"tol=([0-9.eE+-]+)")


def save_fig(filepath, figure=None, **kwargs):
    import tikzplotlib
    import os
    import matplotlib.pyplot as plt

    ## split extension
    fpath = os.path.splitext(filepath)[0]
    ## get figure handle
    if figure is None:
        figure = plt.gcf()
    figure.savefig(fpath + ".png", dpi=300, transparent=True)
    tikzplotlib.save(
        figure=figure,
        filepath=fpath + ".tex",
        axis_height='\\figureheight',
        axis_width='\\figurewidth',
        override_externals=True,
        **kwargs
    )


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


def get_sorted_mode_and_tol_dirs(path):
    mode_dirs = []
    tol_dirs = []

    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if not os.path.isdir(full_path):
            continue

        mode_match = modes_pattern.match(entry)
        tol_match = tol_pattern.match(entry)

        if mode_match:
            mode_dirs.append((int(mode_match.group(1)), full_path))
        elif tol_match:
            tol_dirs.append((float(tol_match.group(1)), full_path))

    mode_dirs.sort(key=lambda x: x[0])
    tol_dirs.sort(key=lambda x: x[0])

    sorted_mode_paths = [d[1] for d in mode_dirs]
    sorted_tol_paths = [d[1] for d in tol_dirs]

    return sorted_mode_paths, sorted_tol_paths


def extract_data(sorted_paths, file1, file2, file3, file4):
    data1 = []
    data2 = []
    data3 = []
    data4 = []

    for root in sorted_paths:
        files = os.listdir(root)

        if file1[0] in files and file2[0] in files and file3[0] in files and file4[0] in files:
            file1_path = os.path.join(root, file1[0])
            file2_path = os.path.join(root, file2[0])
            file3_path = os.path.join(root, file3[0])
            file4_path = os.path.join(root, file4[0])

            data1.append(np.load(file1_path, allow_pickle=True)[-1])
            data2.append(np.load(file2_path, allow_pickle=True))
            data3.append(np.load(file3_path, allow_pickle=True))
            data4.append(np.load(file4_path, allow_pickle=True))
        else:
            file1_path = os.path.join(root, "checkpoint", file1[1])
            file2_path = os.path.join(root, "checkpoint", file2[1])
            file3_path = os.path.join(root, "checkpoint", file3[1])
            file4_path = os.path.join(root, "checkpoint", file4[1])

            data1.append(np.load(file1_path, allow_pickle=True)[-1])
            data2.append(np.load(file2_path, allow_pickle=True))
            data3.append(np.load(file3_path, allow_pickle=True))
            data4.append(np.load(file4_path, allow_pickle=True))

    return data1, data2, data3, data4


# Function to extract the number from a string (like a file name)
def extract_number(dir_name: str) -> float:
    """
    Given a directory name like "tol=1e-4", return 1e-4 as a float.
    If no match, return +inf so it sorts to the end.
    """
    m = tol_pattern.search(dir_name)
    return float(m.group("val")) if m else float('inf')


def extract(Lx_results, case, type_of_basis, L1L2_val, CTC_mask, problem, interp_scheme=None,
            file1=None, file2=None, file3=None, file4=None):
    if interp_scheme is None:
        base_path = "/Users/shubhadityaburela/Python/Paper4_OCsPOD/OCsPOD/" + Lx_results + "/data/" + case + "/" + type_of_basis + "/" + L1L2_val + "/" + CTC_mask + "/problem=" + str(
            problem) + "/"
    else:
        base_path = "/Users/shubhadityaburela/Python/Paper4_OCsPOD/OCsPOD/" + Lx_results + "/data/" + case + "/" + type_of_basis + "/" + L1L2_val + "/" + CTC_mask + "/" + interp_scheme + "/problem=" + str(
            problem) + "/"

    sorted_mode_dirs, sorted_tol_dirs = get_sorted_mode_and_tol_dirs(base_path)

    data1_modes, data2_modes, data3_modes, data4_modes = extract_data(sorted_mode_dirs, file1, file2, file3, file4)
    data1_tol, data2_tol, data3_tol, data4_tol = extract_data(sorted_tol_dirs, file1, file2, file3, file4)

    return (data1_modes, data2_modes, data3_modes, data4_modes), (data1_tol, data2_tol, data3_tol, data4_tol)


# Number of modes for each problem
if regularize == "L2":
    if problem == 1:
        FOM_J = 1.2290
    elif problem == 2:
        FOM_J = 2.9831
    else:
        FOM_J = 1.8977
    tol_array = np.asarray([1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
    Lx_results = "L2_results"
    L1L2_val = "L1=0.0_L2=0.001"
    CTC_mask = "CTC_mask=False"
elif regularize == "L1":
    if problem == 1:
        FOM_J = 0.04541
    elif problem == 2:
        FOM_J = 0.04592
    else:
        FOM_J = 0.02950
    tol_array = np.asarray([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    Lx_results = "L1_results"
    L1L2_val = "L1=0.001_L2=0.0"
    CTC_mask = "CTC_mask=True"
#################################################################################################################
# Plotting for the POD-Galerkin cases
case = [["PODG_FOTR_RA", "PODG_FRTO"],
        ["PODG_FOTR_RA_adaptive", "PODG_FRTO_adaptive"]]

for idx, cs in enumerate(case):
    best_J_FOTR_s = []
    best_J_FOTR_c = []
    best_J_FRTO_p = []
    best_J_FRTO_c = []
    if idx == 0:
        basis = "Fixed"
    else:
        basis = "Adaptive"

    # FOTR
    modes_FOTR_s, tol_FOTR_s = extract(Lx_results=Lx_results,
                                       case=cs[0],
                                       type_of_basis="separate_basis",
                                       L1L2_val=L1L2_val,
                                       CTC_mask=CTC_mask,
                                       problem=problem,
                                       interp_scheme=None,
                                       file1=["J_opt_FOM_list_final.npy", "J_opt_FOM_list.npy"],
                                       file2=["trunc_modes_list_p_final.npy", "trunc_modes_p.npy"],
                                       file3=["trunc_modes_list_a_final.npy", "trunc_modes_a.npy"],
                                       file4=["best_details_final.npy", "best_details.npy"]
                                       )
    modes_FOTR_c, tol_FOTR_c = extract(Lx_results=Lx_results,
                                       case=cs[0],
                                       type_of_basis="primal+adjoint_common_basis",
                                       L1L2_val=L1L2_val,
                                       CTC_mask=CTC_mask,
                                       problem=problem,
                                       interp_scheme=None,
                                       file1=["J_opt_FOM_list_final.npy", "J_opt_FOM_list.npy"],
                                       file2=["trunc_modes_list_p_final.npy", "trunc_modes_p.npy"],
                                       file3=["trunc_modes_list_a_final.npy", "trunc_modes_a.npy"],
                                       file4=["best_details_final.npy", "best_details.npy"]
                                       )
    print("-------------------------------------")
    # Tolerance values for table
    for idx, val in enumerate(tol_array):
        print(
            f"{cs[0]}, {basis} separate basis, tolerance = {val} best cost: {tol_FOTR_s[3][idx]}, last valid cost: {tol_FOTR_s[0][idx]:.8}, with "
            f"Nm_avg_p: {int(sum(tol_FOTR_s[1][idx]) / len(tol_FOTR_s[1][idx]))}, "
            f"Nm_avg_a: {int(sum(tol_FOTR_s[2][idx]) / len(tol_FOTR_s[2][idx]))}")
        best_J_FOTR_s.append(tol_FOTR_s[3][idx].item()["J"])
    print("-------------------------------------")
    # Tolerance values for table
    for idx, val in enumerate(tol_array):
        print(
            f"{cs[0]}, {basis} common basis, tolerance = {val} best cost: {tol_FOTR_c[3][idx]}, last valid cost: {tol_FOTR_c[0][idx]:.8}, with "
            f"Nm_avg_p: {int(sum(tol_FOTR_c[1][idx]) / len(tol_FOTR_c[1][idx]))}, "
            f"Nm_avg_a: {int(sum(tol_FOTR_c[2][idx]) / len(tol_FOTR_c[2][idx]))}")
        best_J_FOTR_c.append(tol_FOTR_c[3][idx].item()["J"])

    # FRTO
    modes_FRTO_p, tol_FRTO_p = extract(Lx_results=Lx_results,
                                       case=cs[1],
                                       type_of_basis="primal_basis",
                                       L1L2_val=L1L2_val,
                                       CTC_mask=CTC_mask,
                                       problem=problem,
                                       interp_scheme=None,
                                       file1=["J_opt_FOM_list_final.npy", "J_opt_FOM_list.npy"],
                                       file2=["trunc_modes_list_final.npy", "trunc_modes.npy"],
                                       file3=["trunc_modes_list_final.npy", "trunc_modes.npy"],
                                       file4=["best_details_final.npy", "best_details.npy"]
                                       )
    modes_FRTO_c, tol_FRTO_c = extract(Lx_results=Lx_results,
                                       case=cs[1],
                                       type_of_basis="primal+adjoint_common_basis",
                                       L1L2_val=L1L2_val,
                                       CTC_mask=CTC_mask,
                                       problem=problem,
                                       interp_scheme=None,
                                       file1=["J_opt_FOM_list_final.npy", "J_opt_FOM_list.npy"],
                                       file2=["trunc_modes_list_final.npy", "trunc_modes.npy"],
                                       file3=["trunc_modes_list_final.npy", "trunc_modes.npy"],
                                       file4=["best_details_final.npy", "best_details.npy"]
                                       )
    print("-------------------------------------")
    # Tolerance values for table
    for idx, val in enumerate(tol_array):
        print(
            f"{cs[1]}, {basis} primal basis, tolerance = {val} best cost: {tol_FRTO_p[3][idx]}, last valid cost: {tol_FRTO_p[0][idx]:.8}, with "
            f"Nm_avg: {int(sum(tol_FRTO_p[2][idx]) / len(tol_FRTO_p[2][idx]))}")
        best_J_FRTO_p.append(tol_FRTO_p[3][idx].item()["J"])
    print("-------------------------------------")
    # Tolerance values for table
    for idx, val in enumerate(tol_array):
        print(
            f"{cs[1]}, {basis} common basis, tolerance = {val} best cost: {tol_FRTO_c[3][idx]}, last valid cost: {tol_FRTO_c[0][idx]:.8}, with "
            f"Nm_avg: {int(sum(tol_FRTO_c[2][idx]) / len(tol_FRTO_c[2][idx]))}")
        best_J_FRTO_c.append(tol_FRTO_c[3][idx].item()["J"])

    fig1 = plt.figure(figsize=(15, 5))
    ax1 = fig1.add_subplot(111)
    ax1.axhline(y=FOM_J, color='sienna', linestyle='-', label="FOM")
    ax1.plot(tol_array, tol_FOTR_s[0], marker="1", label="FOTR separate basis")
    # ax1.plot(tol_array, best_J_FOTR_s, marker="2", label="FOTR separate basis (best)")
    ax1.plot(tol_array, tol_FOTR_c[0], marker="1", label="FOTR common basis")
    # ax1.plot(tol_array, best_J_FOTR_c, marker="2", label="FOTR common basis (best)")
    ax1.plot(tol_array, tol_FRTO_p[0], marker="1", label="FRTO primal basis")
    # ax1.plot(tol_array, best_J_FRTO_p, marker="2", label="FRTO primal basis (best)")
    ax1.plot(tol_array, tol_FRTO_c[0], marker="1", label="FRTO common basis")
    # ax1.plot(tol_array, best_J_FRTO_c, marker="2", label="FRTO common basis (best)")
    ax1.set_xlabel(r"tol")
    ax1.set_ylabel(r"$\mathcal{J}$")
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.grid()
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig1.tight_layout()
    fig1.savefig(impath + 'PODG_tolVsCost_' + basis + '_P' + str(problem), dpi=300, transparent=True)
    save_fig(impath + 'PODG_tolVsCost_' + basis + '_P' + str(problem), fig1)

# ===============================================================================================================#
# Plotting for the sPOD-Galerkin cases
case = [["sPODG_FOTR_RA", "sPODG_FRTO"],
        ["sPODG_FOTR_RA_adaptive", "sPODG_FRTO_adaptive"]]

for idx, cs in enumerate(case):
    best_J_FOTR_s = []
    best_J_FOTR_c = []
    best_J_FRTO_p = []
    best_J_FRTO_c = []
    if idx == 0:
        basis = "Fixed"
    else:
        basis = "Adaptive"

    # FOTR
    modes_FOTR_s, tol_FOTR_s = extract(Lx_results=Lx_results,
                                       case=cs[0],
                                       type_of_basis="separate_basis",
                                       L1L2_val=L1L2_val,
                                       CTC_mask=CTC_mask,
                                       problem=problem,
                                       interp_scheme="Lagr",
                                       file1=["J_opt_FOM_list_final.npy", "J_opt_FOM_list.npy"],
                                       file2=["trunc_modes_list_p_final.npy", "trunc_modes_p.npy"],
                                       file3=["trunc_modes_list_a_final.npy", "trunc_modes_a.npy"],
                                       file4=["best_details_final.npy", "best_details.npy"]
                                       )
    modes_FOTR_c, tol_FOTR_c = extract(Lx_results=Lx_results,
                                       case=cs[0],
                                       type_of_basis="primal+adjoint_common_basis",
                                       L1L2_val=L1L2_val,
                                       CTC_mask=CTC_mask,
                                       problem=problem,
                                       interp_scheme="Lagr",
                                       file1=["J_opt_FOM_list_final.npy", "J_opt_FOM_list.npy"],
                                       file2=["trunc_modes_list_p_final.npy", "trunc_modes_p.npy"],
                                       file3=["trunc_modes_list_a_final.npy", "trunc_modes_a.npy"],
                                       file4=["best_details_final.npy", "best_details.npy"]
                                       )
    print("-------------------------------------")
    # Tolerance values for table
    for idx, val in enumerate(tol_array):
        print(
            f"{cs[0]}, {basis} separate basis, tolerance = {val} best cost: {tol_FOTR_s[3][idx]}, last valid cost: {tol_FOTR_s[0][idx]:.8}, with "
            f"Nm_avg_p: {int(sum(tol_FOTR_s[1][idx]) / len(tol_FOTR_s[1][idx]))}, "
            f"Nm_avg_a: {int(sum(tol_FOTR_s[2][idx]) / len(tol_FOTR_s[2][idx]))}")
        best_J_FOTR_s.append(tol_FOTR_s[3][idx].item()["J"])
    print("-------------------------------------")
    # Tolerance values for table
    for idx, val in enumerate(tol_array):
        print(
            f"{cs[0]}, {basis} common basis, tolerance = {val} best cost: {tol_FOTR_c[3][idx]}, last valid cost: {tol_FOTR_c[0][idx]:.8}, with "
            f"Nm_avg_p: {int(sum(tol_FOTR_c[1][idx]) / len(tol_FOTR_c[1][idx]))}, "
            f"Nm_avg_a: {int(sum(tol_FOTR_c[2][idx]) / len(tol_FOTR_c[2][idx]))}")
        best_J_FOTR_c.append(tol_FOTR_c[3][idx].item()["J"])

    # FRTO
    modes_FRTO_p, tol_FRTO_p = extract(Lx_results=Lx_results,
                                       case=cs[1],
                                       type_of_basis="primal_basis",
                                       L1L2_val=L1L2_val,
                                       CTC_mask=CTC_mask,
                                       problem=problem,
                                       interp_scheme="Lagr",
                                       file1=["J_opt_FOM_list_final.npy", "J_opt_FOM_list.npy"],
                                       file2=["trunc_modes_list_final.npy", "trunc_modes.npy"],
                                       file3=["trunc_modes_list_final.npy", "trunc_modes.npy"],
                                       file4=["best_details_final.npy", "best_details.npy"]
                                       )
    modes_FRTO_c, tol_FRTO_c = extract(Lx_results=Lx_results,
                                       case=cs[1],
                                       type_of_basis="primal+adjoint_common_basis",
                                       L1L2_val=L1L2_val,
                                       CTC_mask=CTC_mask,
                                       problem=problem,
                                       interp_scheme="Lagr",
                                       file1=["J_opt_FOM_list_final.npy", "J_opt_FOM_list.npy"],
                                       file2=["trunc_modes_list_final.npy", "trunc_modes.npy"],
                                       file3=["trunc_modes_list_final.npy", "trunc_modes.npy"],
                                       file4=["best_details_final.npy", "best_details.npy"]
                                       )
    print("-------------------------------------")
    # Tolerance values for table
    for idx, val in enumerate(tol_array):
        print(
            f"{cs[1]}, {basis} primal basis, tolerance = {val} best cost: {tol_FRTO_p[3][idx]}, last valid cost: {tol_FRTO_p[0][idx]:.8}, with "
            f"Nm_avg: {int(sum(tol_FRTO_p[2][idx]) / len(tol_FRTO_p[2][idx]))}")
        best_J_FRTO_p.append(tol_FRTO_p[3][idx].item()["J"])
    print("-------------------------------------")
    # Tolerance values for table
    for idx, val in enumerate(tol_array):
        print(
            f"{cs[1]}, {basis} common basis, tolerance = {val} best cost: {tol_FRTO_c[3][idx]}, last valid cost: {tol_FRTO_c[0][idx]:.8}, with "
            f"Nm_avg: {int(sum(tol_FRTO_c[2][idx]) / len(tol_FRTO_c[2][idx]))}")
        best_J_FRTO_c.append(tol_FRTO_c[3][idx].item()["J"])

    fig1 = plt.figure(figsize=(15, 5))
    ax1 = fig1.add_subplot(111)
    ax1.axhline(y=FOM_J, color='sienna', linestyle='-', label="FOM")
    ax1.plot(tol_array, tol_FOTR_s[0], marker="1", label="FOTR separate basis")
    # ax1.plot(tol_array, best_J_FOTR_s, marker="2", label="FOTR separate basis (best)")
    ax1.plot(tol_array, tol_FOTR_c[0], marker="1", label="FOTR common basis")
    # ax1.plot(tol_array, best_J_FOTR_c, marker="2", label="FOTR common basis (best)")
    ax1.plot(tol_array, tol_FRTO_p[0], marker="1", label="FRTO primal basis")
    # ax1.plot(tol_array, best_J_FRTO_p, marker="2", label="FRTO primal basis (best)")
    ax1.plot(tol_array, tol_FRTO_c[0], marker="1", label="FRTO common basis")
    # ax1.plot(tol_array, best_J_FRTO_c, marker="2", label="FRTO common basis (best)")
    ax1.set_xlabel(r"tol")
    ax1.set_ylabel(r"$\mathcal{J}$")
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.grid()
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig1.tight_layout()
    fig1.savefig(impath + 'sPODG_tolVsCost_' + basis + '_P' + str(problem), dpi=300, transparent=True)
    save_fig(impath + 'sPODG_tolVsCost_' + basis + '_P' + str(problem), fig1)
