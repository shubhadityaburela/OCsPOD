"""
This file can handle both the scenarios. (This file is specifically for the cubic spline interpolation version)
1. Fixed tolerance
2. Fixed modes
"""
from time import perf_counter
import time

import matplotlib

from Update import Update_Control_sPODG_FRTO_TWBT
from matplotlib import pyplot as plt

from Coefficient_Matrix import CoefficientMatrix
from Costs import Calc_Cost, Calc_Cost_sPODG
from Cubic_spline import *
from Plots import PlotFlow
from advection import advection
from Helper import ControlSelectionMatrix_advection, calc_shift, compute_red_basis
from Helper_sPODG import subsample, central_FDMatrix
import os
import numpy as np
import scipy.sparse as sp
import argparse

import sys

sys.path.append('./sPOD/lib/')

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

impath = "./data/sPODG/FRTO/CubSpl/problem=" + str(problem) + "/" + TYPE + "=" + str(VAL) + "/"  # for data
immpath = "./plots/sPODG/FRTO/CubSpl/problem=" + str(problem) + "/" + TYPE + "=" + str(VAL) + "/"  # for plots
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
elif problem == 2:  # Sharp wave params (earlier kink):
    wf = advection(Nxi=Nxi, Neta=Neta, timesteps=Nt, cfl=8 / 6,
                   tilt_from=3 * Nt // 4, v_x=0.55, v_x_t=1.0, variance=0.5, offset=30)
elif problem == 3:  # Sharp wave params (later kink):
    wf = advection(Nxi=Nxi, Neta=Neta, timesteps=Nt, cfl=8 / 6,
                   tilt_from=9 * Nt // 10, v_x=0.6, v_x_t=1.3, variance=0.5, offset=30)
else:  # Default is problem 2
    wf = advection(Nxi=Nxi, Neta=Neta, timesteps=Nt, cfl=8 / 6,
                   tilt_from=3 * Nt // 4, v_x=0.55, v_x_t=1.0, variance=0.5, offset=30)
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
    'opt_iter': 100,  # Total iterations
    'shift_sample': 800,  # Number of samples for shift interpolation
    'beta': 1 / 2,  # Beta factor for two-way backtracking line search
    'verbose': True,  # Print options
    'base_tol': tol,  # Base tolerance for selecting number of modes (main variable for truncation)
    'omega_cutoff': 1e-10,  # Below this cutoff the Armijo and Backtracking should exit the update loop
    'threshold': threshold,
    # Variable for selecting threshold based truncation or mode based. "TRUE" for threshold based
    # "FALSE" for mode based.
    'Nm': modes,  # Number of modes for truncation if threshold selected to False.
}

# %% ROM Variables
D = central_FDMatrix(order=6, Nx=wf.Nxi, dx=wf.dx)

# Generate the shift samples
delta_s = subsample(wf.X, num_sample=kwargs['shift_sample'])

# Calculate the initial shift
delta = calc_shift(qs_org, q0, wf.X, wf.t)

# Calculate the constant spline coefficient matrices (only needed once)
A1, D1, D2, R = give_spline_coefficient_matrices(kwargs['Nx'])

# For two-way backtracking line search
omega = 1

stag = False
stag_cntr = 0

start = time.time()
time_odeint_s = perf_counter()  # save running time
# %%
for opt_step in range(kwargs['opt_iter']):

    print("\n==============================")
    print("Optimization step: %d" % opt_step)

    '''
    Forward calculation with primal FOM at intermediate steps for basis computation
    '''
    qs = wf.TI_primal(q0, f, A_p, psi)

    if stag:
        delta = calc_shift(qs, q0, wf.X, wf.t)
        shift_refine_cntr_list.append(opt_step)

    # Construct the spline coefficients for the qs
    b, c, d = construct_spline_coeffs_multiple(qs, A1, D1, D2, R, kwargs['dx'])
    qs_s = shift_matrix_precomputed_coeffs_multiple(qs, delta[0], b, c, d, kwargs['Nx'], kwargs['dx'])

    V_p, qs_s_POD = compute_red_basis(qs_s, **kwargs)
    Nm = V_p.shape[1]
    err = np.linalg.norm(qs_s - qs_s_POD) / np.linalg.norm(qs_s)
    print(f"Relative error for shifted primal: {err}, with Nm: {Nm}")

    # Initial condition for dynamical simulation
    a_p = wf.IC_primal_sPODG_FRTO(q0, V_p)
    a_a = wf.IC_adjoint_sPODG_FRTO(Nm)

    # Construct the primal system matrices for the sPOD-Galerkin approach
    Vd_p, Wd_p, Ud_p, lhs_p, rhs_p, c_p = wf.mat_primal_sPODG_FRTO_CubSpl(V_p, A1, D1, D2, R, A_p, psi,
                                                                          delta_s,
                                                                          samples=kwargs['shift_sample'],
                                                                          modes=Nm)
    # plt.plot(Vd_p[100][:, 0], label="1")
    # plt.plot(Vd_p[100][:, 1], label="2")
    # plt.plot(Vd_p[100][:, 2], label="3")
    # plt.plot(Vd_p[100][:, 3], label="4")
    # plt.plot(Vd_p[100][:, 4], label="5")
    # # plt.plot(Vd_p[100][:, 5], label="6")
    # # plt.plot(Vd_p[100][:, 6], label="7")
    # # plt.plot(Vd_p[100][:, 7], label="8")
    # plt.legend()
    # plt.savefig('V.png', bbox_inches='tight')
    # exit()

    '''
    Forward calculation
    '''
    as_, as_dot, intIds, weights = wf.TI_primal_sPODG_FRTO(lhs_p, rhs_p, c_p, a_p, f, delta_s, modes=Nm)

    '''
    Objective and costs for control
    '''
    # Compute the interpolation weight and the interval in which the shift lies corresponding to which we compute the
    # V_delta and W_delta matrices
    J, _ = Calc_Cost_sPODG(Vd_p, as_[:-1], qs_target, f, intIds, weights,
                           kwargs['dx'], kwargs['dt'], kwargs['lamda'])

    '''
    Backward calculation with reduced adjoint system
    '''
    as_adj = wf.TI_adjoint_sPODG_FRTO(a_a, f, as_, qs_target, as_dot, lhs_p, rhs_p, c_p, Vd_p, Wd_p,
                                      Nm, intIds, weights)

    '''
     Update Control
    '''
    f, J_opt, _, dL_du_norm, omega, stag = Update_Control_sPODG_FRTO_TWBT(f, lhs_p, rhs_p, c_p, Vd_p, a_p, as_adj,
                                                                          as_, qs_target, delta_s, J, omega,
                                                                          Nm, intIds, weights, wf=wf,
                                                                          **kwargs)

    running_time.append(perf_counter() - time_odeint_s)
    qs_opt_full = wf.TI_primal(q0, f, A_p, psi)
    JJ = Calc_Cost(qs_opt_full, qs_target, f,
                   kwargs['dx'], kwargs['dt'], kwargs['lamda'])

    J_opt_FOM_list.append(JJ)
    J_opt_list.append(J_opt)
    dL_du_norm_list.append(dL_du_norm)
    dL_du_norm_ratio_list.append(dL_du_norm / dL_du_norm_list[0])

    print(
        f"J_opt : {J_opt}, ||dL_du|| = {dL_du_norm}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du_norm / dL_du_norm_list[0]}"
    )

    # Convergence criteria
    if opt_step == kwargs['opt_iter'] - 1:
        print("\n\n-------------------------------")
        print(
            f"WARNING... maximal number of steps reached, "
            f"J_opt : {J_opt}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du_norm / dL_du_norm_list[0]}"
        )
        f_last_valid = np.copy(f)
        break
    elif dL_du_norm / dL_du_norm_list[0] < kwargs['delta_conv']:
        print("\n\n-------------------------------")
        print(
            f"Optimization converged with, "
            f"J_opt : {J_opt}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du_norm / dL_du_norm_list[0]}"
        )
        f_last_valid = np.copy(f)
        break
    else:
        if opt_step == 0:
            if stag:
                print("\n-------------------------------")
                print(
                    f"Armijo Stagnated !!!!!! due to the step length being too low thus updating the shifts at itr: {opt_step}")
                stag_cntr = stag_cntr + 1
                f_last_valid = np.copy(f)
            else:
                stag_cntr = 0
        else:
            dJ = (J_opt_list[-1] - J_opt_list[-2]) / J_opt_list[0]
            if abs(dJ) == 0:
                print(f"WARNING: dJ has turned close to 0...")
                f_last_valid = np.copy(f)
                break
            if stag:
                stag_cntr = stag_cntr + 1
                if stag_cntr >= 2:
                    print(
                        f"Armijo Stagnated !!!!!! even after 2 consecutive shift updates thus exiting at itr: {opt_step} with "
                        f"J_opt : {J_opt}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du_norm / dL_du_norm_list[0]}")
                    break
                print("\n-------------------------------")
                print(
                    f"Armijo Stagnated !!!!!! due to the step length being too low thus updating the shifts at itr: {opt_step}")
                f_last_valid = np.copy(f)
            else:
                stag_cntr = 0

# Compute the final state and adjoint state
qs_opt_full = wf.TI_primal(q0, f_last_valid, A_p, psi)
qs_adj = wf.TI_adjoint(q0_adj, qs_opt_full, qs_target, A_a)

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
np.save(impath + 'err_list.npy', err_list)
np.save(impath + 'trunc_modes_list.npy', trunc_modes_list)
np.save(impath + 'running_time.npy', running_time)
np.save(impath + 'shift_refine_cntr_list.npy', shift_refine_cntr_list)

# Save the optimized solution
np.save(impath + 'qs_opt.npy', qs_opt_full)
np.save(impath + 'qs_adj_opt.npy', qs_adj)
np.save(impath + 'f_opt.npy', f_opt)
np.save(impath + 'f_opt_low.npy', f_last_valid)

# %%
# Plot the results
pf = PlotFlow(wf.X, wf.Y, wf.t)
pf.plot1D(qs_org, name="qs_org", immpath=immpath)
pf.plot1D(qs_target, name="qs_target", immpath=immpath)
pf.plot1D(qs_opt_full, name="qs_opt", immpath=immpath)
pf.plot1D(qs_adj, name="qs_adj_opt", immpath=immpath)
pf.plot1D(f_opt, name="f_opt", immpath=immpath)
pf.plot1D_ROM_converg(J_opt_list, immpath=immpath)
