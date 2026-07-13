import argparse
import warnings
from pathlib import Path

import numpy as np
import os
import matplotlib.pyplot as plt
import re
from typing import List, Tuple, Optional

import scipy
from mpl_toolkits.axes_grid1 import make_axes_locatable

from Coefficient_Matrix import CoefficientMatrix
from FOM_solver import IC_primal, TI_primal, IC_adjoint, TI_adjoint, TI_primal_target
from Helper import ControlSelectionMatrix
from grid_params import advection

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

impath = "../../../Plots/advection_plot/Shifting/ROM/"
os.makedirs(impath, exist_ok=True)

# Regex pattern to extract real numbers from file names
tol_pattern = re.compile(r"tol=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")

parser = argparse.ArgumentParser(description="Input the variables for running the script.")
parser.add_argument("Framework", type=str, choices=["FOTR", "FRTO"], help="Choose the framework")
args = parser.parse_args()


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


def setup_advection(Nx, Nt, cfl_fac, type):
    if type == "Shifting":
        wf = advection(Lx=100, Nx=Nx, timesteps=Nt,
                       cfl=(8 / 6) / cfl_fac, tilt_from=3 * Nt // 4,
                       v_x=0.5, v_x_t=1.0,
                       variance=7, offset=12)
    elif type == "Constant_shift":
        wf = advection(Lx=80, Nx=Nx, timesteps=Nt,
                       cfl=0.0425 / cfl_fac, tilt_from=0,
                       v_x=8 / 3, v_x_t=8 / 3,
                       variance=7, offset=20)
    else:
        print("Please choose the correct problem type!!")
        exit()

    wf.Grid()
    return wf


def C_matrix(Nx, CTC_end_index, apply_CTC_mask=False):
    C = np.ones(Nx)
    if apply_CTC_mask:
        C[:CTC_end_index] = 0
        return C == 1
    else:
        return C == 1


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
                 file2: Tuple[str, str]):
    """
    For each directory in sorted_paths:
      - try to load file1[0] and file2[0] from the directory
      - if not present, try checkpoint/file1[1] and checkpoint/file2[1]
    Returns two lists of loaded arrays (data1, data2), preserving ordering of sorted_paths.
    """
    data1: List[np.ndarray] = []
    data2: List[np.ndarray] = []

    for root in sorted_paths:
        rootp = Path(root)
        # check presence of top-level files
        top_f1 = rootp / file1[0]
        top_f2 = rootp / file2[0]

        if top_f1.exists() and top_f2.exists():
            f1_path = top_f1
            f2_path = top_f2
        else:
            chk_dir = rootp / "checkpoint"
            chk_f1 = chk_dir / file1[1]
            chk_f2 = chk_dir / file2[1]
            if chk_f1.exists() and chk_f2.exists():
                f1_path = chk_f1
                f2_path = chk_f2
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

        # preserve your original behaviour: take last element of arr1 if possible
        try:
            # if arr1 is array-like and has indexable last element
            data1.append(arr1[-1])
        except Exception:
            # fallback: append entire thing
            data1.append(arr1)

        data2.append(arr2)

    return data1, data2


def extract(ROM_framework: str,
            type_of_basis: str,
            file1: Optional[Tuple[str, str]] = None,
            file2: Optional[Tuple[str, str]] = None):
    """
    Build base path, get sorted tol directories, and extract data.
    file1 and file2 are tuples: (primary_filename, checkpoint_filename)
    """
    if file1 is None or file2 is None:
        raise ValueError("file1, file2 and file3 must be provided as (primary_name, checkpoint_name) tuples")

    base_path = os.path.join(
        "/Users/shubhadityaburela/Python/Paper4_OCsPOD/OCsPOD/results_advection/Shifting/ROM/data",
        ROM_framework, type_of_basis, "L1=0.0_L2=0.001", "CTC_mask=False"
    )

    sorted_tol_dirs = get_sorted_tol_dirs(base_path)

    data1_tol, data2_tol = extract_data(sorted_tol_dirs, file1, file2)
    return data1_tol, data2_tol


FOM_J = 1.230e+00
tol_array_PODG = np.asarray([1e-07, 5e-07, 1e-06, 5e-06, 1e-05, 5e-05, 1e-04, 5e-04, 1e-03, 5e-03, 1e-02])
tol_array_sPODG = np.asarray([1e-07, 5e-07, 1e-06, 5e-06, 1e-05, 5e-05, 1e-04, 5e-04, 1e-03, 5e-03, 1e-02])

basis_name1 = "primal+adjoint_common_basis"
if args.Framework == "FOTR":
    case1 = "PODG_FOTR_adaptive"
    case2 = "sPODG_FOTR_adaptive"
else:
    case1 = "PODG_FRTO_adaptive"
    case2 = "sPODG_FRTO_adaptive"

################################### FOM #########################################
FOM_control = np.load(
    "/Users/shubhadityaburela/Python/Paper4_OCsPOD/OCsPOD/results_advection/Shifting/ROM/data/FOM/L1=0.0_L2=0.001/CTC_mask=False/best_control_final.npy",
    allow_pickle=True)

################################### PODG #########################################
POD_tol_data_1, POD_tol_data_2 = extract(ROM_framework=case1,
                                         type_of_basis=basis_name1,
                                         file1=["J_opt_FOM_list_final.npy", "J_opt_FOM_list.npy"],
                                         file2=["best_control_final.npy", "best_control.npy"]
                                         )

################################### sPODG ########################################
sPOD_tol_data_1, sPOD_tol_data_2 = extract(ROM_framework=case2,
                                           type_of_basis=basis_name1,
                                           file1=["J_opt_FOM_list_final.npy", "J_opt_FOM_list.npy"],
                                           file2=["best_control_final.npy", "best_control.npy"]
                                           )

tol_idx_with_min_cost_PODG = np.argmin(POD_tol_data_1)
tol_idx_with_min_cost_sPODG = np.argmin(sPOD_tol_data_1)
tol_with_min_cost_PODG = tol_array_PODG[tol_idx_with_min_cost_PODG]
tol_with_min_cost_sPODG = tol_array_sPODG[tol_idx_with_min_cost_sPODG]

###########################################################################################
# Unpack regularization parameters
Nx, Nt, cfl_fac = 3200, 3360, 1
type_of_problem = "Shifting"
# Set up WF and control matrix
wf = setup_advection(Nx, Nt, cfl_fac, type_of_problem)
n_c_init = 40
psi = ControlSelectionMatrix(wf, n_c_init, Gaussian=True, gaussian_mask_sigma=0.5)
adjust = wf.dx
n_c = psi.shape[1]
psi = scipy.sparse.csc_matrix(psi)
# Build coefficient matrices
Mat = CoefficientMatrix(orderDerivative=wf.firstderivativeOrder,
                        Nxi=wf.Nx, Neta=1,
                        periodicity='Periodic',
                        dx=wf.dx, dy=0)
A_p = - wf.v_x[0] * Mat.Grad_Xi_kron
A_a = A_p.transpose()
qs0 = IC_primal(wf.X, wf.Lx, wf.offset, wf.variance, type_of_problem=type_of_problem)
q0_adj = np.ascontiguousarray(IC_adjoint(wf.X))
qs_target = TI_primal_target(qs0, Mat.Grad_Xi_kron, wf.v_x_target, wf.Nx, wf.Nt,
                             wf.dt, nu=0.1 if type_of_problem == "Constant_shift" else 0.0)
C = C_matrix(wf.Nx, wf.CTC_end_index, apply_CTC_mask=False)

###########################################################################################
# FOM
f = (FOM_control.copy()).astype(np.float64)
qs_org = TI_primal(qs0, f, A_p, psi, wf.Nx, wf.Nt, wf.dt)
qs_adj = TI_adjoint(q0_adj, qs_org, qs_target, None, A_a, None, C, wf.Nx, wf.dx, wf.Nt, wf.dt,
                    scheme="RK4")
f_FOM = psi @ f

###########################################################################################
# PODG
f = POD_tol_data_2[tol_idx_with_min_cost_PODG].astype(np.float64)
qs_org_PODG = TI_primal(qs0, f, A_p, psi, wf.Nx, wf.Nt, wf.dt)
qs_adj_PODG = TI_adjoint(q0_adj, qs_org_PODG, qs_target, None, A_a, None, C, wf.Nx, wf.dx, wf.Nt, wf.dt,
                         scheme="RK4")
f_PODG = psi @ f

###########################################################################################
# PODG (Wrong)
f = POD_tol_data_2[0].astype(np.float64)
qs_org_PODG_wrong = TI_primal(qs0, f, A_p, psi, wf.Nx, wf.Nt, wf.dt)
qs_adj_PODG_wrong = TI_adjoint(q0_adj, qs_org_PODG_wrong, qs_target, None, A_a, None, C, wf.Nx, wf.dx, wf.Nt, wf.dt,
                               scheme="RK4")
f_PODG_wrong = psi @ f

###########################################################################################
# sPODG
f = sPOD_tol_data_2[tol_idx_with_min_cost_sPODG].astype(np.float64)
qs_org_sPODG = TI_primal(qs0, f, A_p, psi, wf.Nx, wf.Nt, wf.dt)
qs_adj_sPODG = TI_adjoint(q0_adj, qs_org_sPODG, qs_target, None, A_a, None, C, wf.Nx, wf.dx, wf.Nt, wf.dt,
                          scheme="RK4")
f_sPODG = psi @ f

f_min = np.min(f_FOM)
f_max = np.max(f_FOM)
adj_min = np.min(qs_adj)
adj_max = np.max(qs_adj)
state_min = np.min(qs_org)
state_max = np.max(qs_org)
##############################################################################################################
fig = plt.figure(figsize=(20, 15))
ax1 = fig.add_subplot(3, 4, 1)
im1 = ax1.pcolormesh(f_FOM.T, cmap='YlOrRd', vmin=f_min, vmax=f_max)
ax1.axis('off')
ax1.axis('auto')
ax1.set_title(r"$FOM$")
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='10%', pad=0.08)
fig.colorbar(im1, cax=cax, orientation='vertical')

ax2 = fig.add_subplot(3, 4, 2)
im2 = ax2.pcolormesh(f_sPODG.T, cmap='YlOrRd', vmin=f_min, vmax=f_max)
ax2.axis('off')
ax2.axis('auto')
ax2.set_title(r"$sPOD-G, \: modes = $" + str(modes_with_min_cost_sPODG))
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='10%', pad=0.08)
fig.colorbar(im2, cax=cax, orientation='vertical')

ax3 = fig.add_subplot(3, 4, 3)
im3 = ax3.pcolormesh(f_PODG.T, cmap='YlOrRd', vmin=f_min, vmax=f_max)
ax3.axis('off')
ax3.axis('auto')
ax3.set_title(r"$POD-G, \: modes = $" + str(modes_with_min_cost_PODG))
divider = make_axes_locatable(ax3)
cax = divider.append_axes('right', size='10%', pad=0.08)
fig.colorbar(im3, cax=cax, orientation='vertical')

ax4 = fig.add_subplot(3, 4, 4)
im4 = ax4.pcolormesh(f_PODG_wrong.T, cmap='YlOrRd', vmin=f_min, vmax=f_max)
ax4.axis('off')
ax4.axis('auto')
ax4.set_title(r"$POD-G, \: modes = $" + str(modes_with_min_cost_sPODG))
divider = make_axes_locatable(ax4)
cax = divider.append_axes('right', size='10%', pad=0.08)
fig.colorbar(im4, cax=cax, orientation='vertical')

##############################################################################################################
ax5 = fig.add_subplot(3, 4, 5)
im5 = ax5.pcolormesh(qs_adj.T, cmap='YlOrRd', vmin=adj_min, vmax=adj_max)
ax5.axis('off')
ax5.axis('auto')
divider = make_axes_locatable(ax5)
cax = divider.append_axes('right', size='10%', pad=0.08)
fig.colorbar(im5, cax=cax, orientation='vertical')

ax6 = fig.add_subplot(3, 4, 6)
im6 = ax6.pcolormesh(qs_adj_sPODG.T, cmap='YlOrRd', vmin=adj_min, vmax=adj_max)
ax6.axis('off')
ax6.axis('auto')
divider = make_axes_locatable(ax6)
cax = divider.append_axes('right', size='10%', pad=0.08)
fig.colorbar(im6, cax=cax, orientation='vertical')

ax7 = fig.add_subplot(3, 4, 7)
im7 = ax7.pcolormesh(qs_adj_PODG.T, cmap='YlOrRd', vmin=adj_min, vmax=adj_max)
ax7.axis('off')
ax7.axis('auto')
divider = make_axes_locatable(ax7)
cax = divider.append_axes('right', size='10%', pad=0.08)
fig.colorbar(im7, cax=cax, orientation='vertical')

ax8 = fig.add_subplot(3, 4, 8)
im8 = ax8.pcolormesh(qs_adj_PODG_wrong.T, cmap='YlOrRd', vmin=adj_min, vmax=adj_max)
ax8.axis('off')
ax8.axis('auto')
divider = make_axes_locatable(ax8)
cax = divider.append_axes('right', size='10%', pad=0.08)
fig.colorbar(im8, cax=cax, orientation='vertical')

##############################################################################################################
ax9 = fig.add_subplot(3, 4, 9)
im9 = ax9.pcolormesh(qs_org.T, cmap='YlOrRd', vmin=state_min, vmax=state_max)
ax9.axis('off')
ax9.axis('auto')
divider = make_axes_locatable(ax9)
cax = divider.append_axes('right', size='10%', pad=0.08)
fig.colorbar(im9, cax=cax, orientation='vertical')

ax10 = fig.add_subplot(3, 4, 10)
im10 = ax10.pcolormesh(qs_org_sPODG.T, cmap='YlOrRd', vmin=state_min, vmax=state_max)
ax10.axis('off')
ax10.axis('auto')
divider = make_axes_locatable(ax10)
cax = divider.append_axes('right', size='10%', pad=0.08)
fig.colorbar(im10, cax=cax, orientation='vertical')

ax11 = fig.add_subplot(3, 4, 11)
im11 = ax11.pcolormesh(qs_org_PODG.T, cmap='YlOrRd', vmin=state_min, vmax=state_max)
ax11.axis('off')
ax11.axis('auto')
divider = make_axes_locatable(ax11)
cax = divider.append_axes('right', size='10%', pad=0.08)
fig.colorbar(im11, cax=cax, orientation='vertical')

ax12 = fig.add_subplot(3, 4, 12)
im12 = ax12.pcolormesh(qs_org_PODG_wrong.T, cmap='YlOrRd', vmin=state_min, vmax=state_max)
ax12.axis('off')
ax12.axis('auto')
divider = make_axes_locatable(ax12)
cax = divider.append_axes('right', size='10%', pad=0.08)
fig.colorbar(im12, cax=cax, orientation='vertical')

fig.supylabel(r"time $t$")
fig.supxlabel(r"space $x$")

# fig.savefig(impath + 'Results', dpi=300, transparent=True)
save_fig(impath + 'Results', fig)
