"""
This file is the version with FOM adjoint (adaptive basis update for primal) with Barzilai Borwein. It can handle both the scenarios.
1. Fixed tolerance
2. Fixed modes
"""
import matplotlib.pyplot as plt

from Coefficient_Matrix import CoefficientMatrix
from Update import Update_Control_sPODG_FOTR_FA_TWBT, Update_Control_BB, Update_Control_sPODG_FOTR_FA_TWBT_dot
from advection import advection
from Plots import PlotFlow
from Helper import ControlSelectionMatrix_advection, compute_red_basis, calc_shift, is_contiguous
from Helper_sPODG import subsample, findIntervals, get_T, central_FDMatrix, get_approx_state_sPODG, compute_residual
from Costs import Calc_Cost_sPODG, Calc_Cost
import os
from time import perf_counter
import numpy as np
import time
import scipy.sparse as sp
import argparse

import matplotlib

matplotlib.use('TkAgg')

import sys

sys.path.append('../sPOD/lib/')
from sPOD_algo import give_interpolation_error

parser = argparse.ArgumentParser(description="Input the variables for running the script.")
parser.add_argument("problem", type=int, choices=[1, 2, 3], help="Specify the problem number (1, 2, or 3)")
parser.add_argument("--modes", type=int, help="Enter the number of modes for modes test")
parser.add_argument("--tol", type=float, help="Enter the tolerance level for tolerance test")
args = parser.parse_args()

problem = args.problem
print("\n")
print(f"Solving problem: {args.problem}")

# Check which argument was provided and act accordingly
if args.modes and args.tol:
    print(f"Modes test takes precedence.....")
    print(f"Mode number provided: {args.modes}")
    TYPE = "modes"
    modes = args.modes
    threshold = False
    tol = None
    VAL = modes
elif args.modes:
    print(f"Modes test.....")
    print(f"Mode number provided: {args.modes}")
    TYPE = "modes"
    modes = args.modes
    threshold = False
    tol = None
    VAL = modes
elif args.tol is not None:
    print(f"Tolerance test.....")
    print(f"Tolerance provided: {args.tol}")
    TYPE = "tol"
    tol = args.tol
    threshold = True
    modes = None
    VAL = tol
else:
    print("No 'modes' or 'tol' argument provided. Please specify one.")
    exit()

impath = "./data/sPODG/problem=" + str(problem) + "/" + TYPE + "=" + str(VAL) + "/"  # for data
immpath = "./plots/sPODG/problem=" + str(problem) + "/" + TYPE + "=" + str(VAL) + "/"  # for plots
os.makedirs(impath, exist_ok=True)

Nxi = 3200
Neta = 1
Nt = 3360
# Wildfire solver initialization along with grid initialization
# Thick wave params:                  # Sharp wave params (earlier kink):             # Sharp wave params (later kink):
# cfl = 8 / 6                         # cfl = 8 / 6                                   # cfl = 8 / 6
# tilt_from = 3 * Nt // 4             # tilt_from = 3 * Nt // 4                       # tilt_from = 9 * Nt / 10
# v_x = 0.5                           # v_x = 0.55                                    # v_x = 0.6
# v_x_t = 1.0                         # v_x_t = 1.0                                   # v_x_t = 1.3
# variance = 7                        # variance = 0.5                                # variance = 0.5
# offset = 12                         # offset = 30                                   # offset = 30


if problem == 1:  # Thick wave params
    wf = advection(Nxi=Nxi, Neta=Neta, timesteps=Nt, cfl=8 / 6,
                   tilt_from=3 * Nt // 4, v_x=0.5, v_x_t=1.0, variance=7, offset=12)
    tilt_from = 3 * Nt // 4
elif problem == 2:  # Sharp wave params (earlier kink):
    wf = advection(Nxi=Nxi, Neta=Neta, timesteps=Nt, cfl=8 / 6,
                   tilt_from=3 * Nt // 4, v_x=0.55, v_x_t=1.0, variance=0.5, offset=30)
    tilt_from = 3 * Nt // 4
elif problem == 3:  # Sharp wave params (later kink):
    wf = advection(Nxi=Nxi, Neta=Neta, timesteps=Nt, cfl=8 / 6,
                   tilt_from=9 * Nt // 10, v_x=0.6, v_x_t=1.3, variance=0.5, offset=30)
    tilt_from = 9 * Nt // 10
else:  # Default is problem 2
    wf = advection(Nxi=Nxi, Neta=Neta, timesteps=Nt, cfl=8 / 6,
                   tilt_from=3 * Nt // 4, v_x=0.55, v_x_t=1.0, variance=0.5, offset=30)
    tilt_from = 3 * Nt // 4

wf.Grid()

# %%
n_c_init = 40  # Number of initial controls

# Selection matrix for the control input
psi = ControlSelectionMatrix_advection(wf, n_c_init, Gaussian=True, gaussian_mask_sigma=0.5)  # Changing the value of
# trim_first_n should basically make the psi matrix and the number of controls to be user defined.
n_c = psi.shape[1]
f = np.zeros((n_c, wf.Nt), order="F")  # Initial guess for the control

# %% Assemble the linear operators
Mat = CoefficientMatrix(orderDerivative=wf.firstderivativeOrder, Nxi=wf.Nxi,
                        Neta=wf.Neta, periodicity='Periodic', dx=wf.dx, dy=wf.dy)
# Convection matrix (Needs to be changed if the velocity is time dependent)
A_p = - (wf.v_x[0] * Mat.Grad_Xi_kron + wf.v_y[0] * Mat.Grad_Eta_kron)
A_a = A_p.transpose()

# %% Solve the uncontrolled system
qs_org = wf.TI_primal(wf.IC_primal(), f, A_p, psi)

qs_target = wf.TI_primal_target(wf.IC_primal(), Mat, np.zeros((wf.Nxi * wf.Neta, wf.Nt)))

# Initial conditions for both primal and adjoint are defined here as they only need to defined once.
q0 = wf.IC_primal()
q0_adj = wf.IC_adjoint()

# %%
dL_du_norm_list = []  # Collecting the gradient over the optimization steps
J_opt_FOM_list = []  # Collecting the FOM cost over the optimization steps
running_time = []  # Time calculated for each iteration in a running manner
J_opt_list = []  # Collecting the optimal cost functional for plotting
dL_du_norm_ratio_list = []  # Collecting the ratio of gradients for plotting
err_list = []  # Offline error reached according to the tolerance
trunc_modes_list = []  # Number of modes needed to reach the offline error
shift_refine_cntr_list = []  # Collects the iteration number at which the shifts are refined/updated

# List of problem constants
kwargs = {
    'dx': wf.dx,
    'dy': wf.dy,
    'dt': wf.dt,
    'Nx': wf.Nxi,
    'Ny': wf.Neta,
    'Nt': wf.Nt,
    'n_c': n_c,
    'lamda': 1e-3,  # Regularization parameter
    'omega': 1,  # initial step size for gradient update
    'delta_conv': 1e-4,  # Convergence criteria
    'delta': 1e-2,  # Armijo constant
    'opt_iter': 500,  # Total iterations
    'shift_sample': 800,  # Number of samples for shift interpolation
    'beta': 1 / 2,  # Beta factor for two-way backtracking line search
    'verbose': True,  # Print options
    'base_tol': tol,  # Base tolerance for selecting number of modes (main variable for truncation)
    'omega_cutoff': 1e-10,  # Below this cutoff the Armijo and Backtracking should exit the update loop
    'threshold': threshold,
    # Variable for selecting threshold based truncation or mode based. "TRUE" for threshold based
    # "FALSE" for mode based.
    'Nm': modes,  # Number of modes for truncation if threshold selected to False.
    'trafo_interp_order': 5,  # Order of the polynomial interpolation for the transformation operators
    'epsilon': 1e-1,  # Error measure for basis refinement
}

#%%
# prepare the time points at which the residual has to be monitored
t_monitor = list(range(0, tilt_from, 20)) + list(range(tilt_from, Nt, 5))

# %% ROM Variables
D = central_FDMatrix(order=6, Nx=wf.Nxi, dx=wf.dx)

# Generate the shift samples
delta_s = subsample(wf.X, num_sample=kwargs['shift_sample'])

# Extract transformation operators based on sub-sampled delta
T_delta, _ = get_T(delta_s, wf.X, wf.t, interp_order=kwargs['trafo_interp_order'])

delta_init = calc_shift(qs_org, q0, wf.X, wf.t)
_, T = get_T(delta_init, wf.X, wf.t, interp_order=kwargs['trafo_interp_order'])

print(2 * give_interpolation_error(qs_org, T))

# For two-way backtracking line search
omega = 1

stag = False
stag_cntr = 0
qq = np.copy(qs_org)
residual = np.zeros_like(qq)

start = time.time()
time_odeint_s = perf_counter()  # save running time
# %%
for opt_step in range(kwargs['opt_iter']):

    print("\n==============================")
    print("Optimization step: %d" % opt_step)

    '''
    Forward calculation with primal FOM for basis update
    '''
    epsilon = np.linalg.norm(residual) / np.linalg.norm(psi @ f)  # Norm error of the residual
    print(f"The residual error measure for basis refinement: {epsilon}")

    if opt_step == 0 or epsilon > kwargs['epsilon']:
        qs = wf.TI_primal(q0, f, A_p, psi)
        print(f"The error: {np.linalg.norm(qs - qq) / np.linalg.norm(qs)}")
        qs_s = T.reverse(qs)

        V_p, qs_s_POD = compute_red_basis(qs_s, **kwargs)
        Nm = V_p.shape[1]
        err = np.linalg.norm(qs_s - qs_s_POD) / np.linalg.norm(qs_s)
        print(f"Relative error for shifted primal: {err}, with Nm: {Nm}")

        err_list.append(err)
        trunc_modes_list.append(Nm)

        # Initial condition for dynamical simulation
        a_p = wf.IC_primal_sPODG_FOTR(q0, V_p)

        # Construct the primal system matrices for the sPOD-Galerkin approach
        Vd_p, Wd_p, lhs_p, rhs_p, c_p = wf.mat_primal_sPODG_FOTR(T_delta, V_p, A_p, psi, D,
                                                                 samples=kwargs['shift_sample'],
                                                                 modes=Nm)
    else:
        qs = np.copy(qq)  # If the error is pretty small then chose the reduced order approximation for adjoint
    '''
    Forward calculation with reduced model
    '''
    as_, intIds, weights = wf.TI_primal_sPODG_FOTR(lhs_p, rhs_p, c_p, a_p, f, delta_s, modes=Nm)

    '''
    Objective and costs for control
    '''
    # Compute the interpolation weight and the interval in which the shift lies corresponding to which we compute the
    # V_delta and W_delta matrices
    J, _ = Calc_Cost_sPODG(Vd_p, as_[:-1], qs_target, f, intIds, weights,
                           kwargs['dx'], kwargs['dt'], kwargs['lamda'])
    J_opt_list.append(J)

    '''
    Backward calculation with adjoint FOM for basis update
    '''
    # Backward FOM solve for basis generation
    qs_adj = wf.TI_adjoint(q0_adj, qs, qs_target, A_a)

    '''
     Update Control
    '''
    if opt_step == 0:
        # First step with a line search
        fNew, J_opt, dL_du_Old, dL_du_norm, omega, qq, as_res, intIds_res, weights_res, as_dot_res, stag = \
            Update_Control_sPODG_FOTR_FA_TWBT_dot(f, lhs_p, rhs_p, c_p, a_p,
                                                  qs_adj,
                                                  qs_target, delta_s,
                                                  Vd_p,
                                                  psi, J, omega, Nm, wf=wf,
                                                  **kwargs)
    else:
        # Rest of all the steps with Barzilai Borwein
        fNew, dL_du_Old, dL_du_norm = Update_Control_BB(fOld, fNew, dL_du_Old, qs_adj, psi, opt_step, **kwargs)
        as_res, intIds_res, weights_res, as_dot_res = wf.TI_primal_sPODG_FOTR_dot(lhs_p, rhs_p, c_p, a_p,
                                                                                  fNew, delta_s, modes)
        qq = get_approx_state_sPODG(Vd_p, fNew, as_res[:-1], intIds_res, weights_res, kwargs['Nx'], kwargs['Nt'])

    # Compute the residual
    residual = compute_residual(D, A_p, Vd_p, psi, as_res[:-1], as_dot_res, fNew, intIds_res,
                                weights_res, t_monitor, kwargs['Nx'], kwargs['Nt'])

    running_time.append(perf_counter() - time_odeint_s)

    # Saving previous controls for Barzilai Borwein step
    fOld = np.copy(f)
    f = np.copy(fNew)

    qs_opt_full = wf.TI_primal(q0, f, A_p, psi)
    JJ = Calc_Cost(qs_opt_full, qs_target, f,
                   kwargs['dx'], kwargs['dt'], kwargs['lamda'])

    J_opt_FOM_list.append(JJ)
    dL_du_norm_list.append(dL_du_norm)
    dL_du_norm_ratio_list.append(dL_du_norm / dL_du_norm_list[0])

    print(
        f"J_opt : {J}, ||dL_du|| = {dL_du_norm}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du_norm / dL_du_norm_list[0]}"
    )

    # Convergence criteria
    if opt_step == kwargs['opt_iter'] - 1:
        print("\n\n-------------------------------")
        print(
            f"WARNING... maximal number of steps reached, "
            f"J_opt : {J}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du_norm / dL_du_norm_list[0]}"
        )
        f_last_valid = np.copy(f)
        break
    elif dL_du_norm / dL_du_norm_list[0] < kwargs['delta_conv']:
        print("\n\n-------------------------------")
        print(
            f"Optimization converged with, "
            f"J_opt : {J}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du_norm / dL_du_norm_list[0]}"
        )
        f_last_valid = np.copy(f)
        break
    else:
        if opt_step != 0:
            dJ = (J_opt_list[-1] - J_opt_list[-2]) / J_opt_list[0]
            if abs(dJ) == 0:
                print(f"WARNING: dJ has turned close to 0...")
                f_last_valid = np.copy(f)
                break

# Compute the final state
qs_opt_full = wf.TI_primal(q0, f_last_valid, A_p, psi)

f_opt = psi @ f_last_valid

# Compute the cost with the optimal control
J = Calc_Cost(qs_opt_full, qs_target, f_last_valid, kwargs['dx'], kwargs['dt'], kwargs['lamda'])
print("\n")
print(f"J with respect to the optimal control for FOM: {J}")

end = time.time()
print("\n")
print("Total time elapsed = %1.3f" % (end - start))
# %%

# Save the convergence lists
np.save(impath + 'J_opt_FOM_list.npy', J_opt_FOM_list)
np.save(impath + 'J_opt_list.npy', J_opt_list)
np.save(impath + 'running_time.npy', running_time)

# Save the optimized solution
np.save(impath + 'qs_opt.npy', qs_opt_full)
np.save(impath + 'qs_adj_opt.npy', qs_adj)
np.save(impath + 'f_opt.npy', f_opt)

# %%
# Plot the results
pf = PlotFlow(wf.X, wf.Y, wf.t)
pf.plot1D(qs_opt_full, name="qs_opt", immpath=immpath)
pf.plot1D(qs_adj, name="qs_adj_opt", immpath=immpath)
pf.plot1D(f_opt, name="f_opt", immpath=immpath)
pf.plot1D_ROM_converg(J_opt_list,
                      err_list,
                      trunc_modes_list,
                      immpath=immpath)
