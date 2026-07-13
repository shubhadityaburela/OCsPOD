import numpy as np
import os
import matplotlib.pyplot as plt
import re
from typing import List

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

impath = "../../../Plots/advection_plot/Constant_shift/ROM_in_Armijo/"
os.makedirs(impath, exist_ok=True)

# Regex pattern to extract real numbers from file names
modes_pattern = re.compile(r"modes\s*=\s*[\(\[]\s*(\d+)", re.IGNORECASE)


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


def setup_advection(Nx, Nt, cfl_fac, type, number):
    if type == "Shifting":
        if number == 1:
            wf = advection(Lx=100, Nx=Nx, timesteps=Nt,
                           cfl=(8 / 6) / cfl_fac, tilt_from=3 * Nt // 4,
                           v_x=0.5, v_x_t=1.0,
                           variance=7, offset=12)
        elif number == 2:
            wf = advection(Lx=100, Nx=Nx, timesteps=Nt,
                           cfl=(8 / 6) / cfl_fac, tilt_from=3 * Nt // 4,
                           v_x=0.55, v_x_t=1.0,
                           variance=0.5, offset=30)
        elif number == 3:
            wf = advection(Lx=100, Nx=Nx, timesteps=Nt,
                           cfl=(8 / 6) / cfl_fac, tilt_from=9 * Nt // 10,
                           v_x=0.6, v_x_t=1.3,
                           variance=0.5, offset=30)
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


def extract(ROM_framework, type_of_basis, file1=None, file2=None):
    base_path = "/Users/shubhadityaburela/Python/Paper4_OCsPOD/OCsPOD/results_advection/Constant_shift/ROM_in_Armijo/data/" + ROM_framework + "/" + type_of_basis + "/L1=0.0_L2=0.001/CTC_mask=False/"

    sorted_mode_dirs = get_sorted_mode_dirs(base_path)

    data1_modes, data2_modes = extract_data(sorted_mode_dirs, file1, file2)

    return data1_modes, data2_modes


modes_array_PODG = np.asarray([10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
modes_array_sPODG = np.asarray([2, 5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50])
basis_name1 = "separate_basis"
basis_name2 = "primal+adjoint_common_basis"
case1 = "PODG_FOTR_FA"
case2 = "sPODG_FOTR_FA"

################################### FOM #########################################
FOM_control = np.load(
    "/Users/shubhadityaburela/Python/Paper4_OCsPOD/OCsPOD/results_advection/Constant_shift/ROM_in_Armijo/data/FOM/L1=0.0_L2=0.001/CTC_mask=False/best_control_final.npy",
    allow_pickle=True)

################################### PODG_FOTR_FA #########################################
# Separate basis / primal basis
POD_modes_data_1s, POD_modes_data_2s = extract(ROM_framework=case1,
                                               type_of_basis=basis_name1,
                                               file1=["J_opt_list_final.npy", "J_opt_list.npy"],
                                               file2=["best_control_final.npy", "best_control.npy"]
                                               )

################################### sPODG_FOTR_RA ########################################
# Separate basis / primal basis
sPOD_modes_data_1s, sPOD_modes_data_2s = extract(ROM_framework=case2,
                                                 type_of_basis=basis_name1,
                                                 file1=["J_opt_list_final.npy", "J_opt_list.npy"],
                                                 file2=["best_control_final.npy", "best_control.npy"]
                                                 )

modes_idx_with_min_cost_PODG = np.argmin(POD_modes_data_1s)
modes_idx_with_min_cost_sPODG = np.argmin(sPOD_modes_data_1s)
modes_with_min_cost_PODG = modes_array_PODG[modes_idx_with_min_cost_PODG]
modes_with_min_cost_sPODG = modes_array_sPODG[modes_idx_with_min_cost_sPODG]

###########################################################################################
# Unpack regularization parameters
Nx, Nt, cfl_fac = 1000, 8000, 1
type_of_problem = "Constant_shift"
# Set up WF and control matrix
wf = setup_advection(Nx, Nt, cfl_fac, type_of_problem, None)
n_c_init = 100
psi = ControlSelectionMatrix(wf, n_c_init, Gaussian=False, gaussian_mask_sigma=0.5)
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
f = POD_modes_data_2s[modes_idx_with_min_cost_PODG].astype(np.float64)
qs_org_PODG = TI_primal(qs0, f, A_p, psi, wf.Nx, wf.Nt, wf.dt)
qs_adj_PODG = TI_adjoint(q0_adj, qs_org_PODG, qs_target, None, A_a, None, C, wf.Nx, wf.dx, wf.Nt, wf.dt,
                         scheme="RK4")
f_PODG = psi @ f

###########################################################################################
# PODG (Wrong)
f = POD_modes_data_2s[0].astype(np.float64)
qs_org_PODG_wrong = TI_primal(qs0, f, A_p, psi, wf.Nx, wf.Nt, wf.dt)
qs_adj_PODG_wrong = TI_adjoint(q0_adj, qs_org_PODG_wrong, qs_target, None, A_a, None, C, wf.Nx, wf.dx, wf.Nt, wf.dt,
                               scheme="RK4")
f_PODG_wrong = psi @ f

###########################################################################################
# sPODG
f = sPOD_modes_data_2s[modes_idx_with_min_cost_sPODG].astype(np.float64)
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