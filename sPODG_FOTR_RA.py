"""
This file is the version with FOM adjoint. It can handle both the scenarios.
1. Fixed tolerance
2. Fixed modes
"""
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import splu

from Coefficient_Matrix import CoefficientMatrix
from FOM_solver import IC_primal, TI_primal, TI_primal_target, IC_adjoint, TI_adjoint
from TI_schemes import DF_start_FOM
from Update import Update_Control_sPODG_FOTR_FA_TWBT, Update_Control_sPODG_FOTR_RA_TWBT, Update_Control_sPODG_FOTR_RA_BB
from grid_params import advection
from Plots import PlotFlow
from Helper import ControlSelectionMatrix_advection, compute_red_basis, calc_shift
from Helper_sPODG import subsample, get_T, central_FDMatrix
from Costs import Calc_Cost_sPODG, Calc_Cost
import os
from time import perf_counter
import numpy as np
import time
import scipy.sparse as sp
import argparse
from scipy.sparse import csc_matrix

import sys

from sPODG_solver import IC_primal_sPODG_FOTR, mat_primal_sPODG_FOTR, TI_primal_sPODG_FOTR, IC_adjoint_sPODG_FOTR, \
    mat_adjoint_sPODG_FOTR, TI_adjoint_sPODG_FOTR

sys.path.append('./sPOD/lib/')
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

Nxi = 3200
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
    wf = advection(Nxi=Nxi, timesteps=Nt, cfl=8 / 6,
                   tilt_from=3 * Nt // 4, v_x=0.5, v_x_t=1.0, variance=7, offset=12)
elif problem == 2:  # Sharp wave params (earlier kink):
    wf = advection(Nxi=Nxi, timesteps=Nt, cfl=8 / 6,
                   tilt_from=3 * Nt // 4, v_x=0.55, v_x_t=1.0, variance=0.5, offset=30)
elif problem == 3:  # Sharp wave params (later kink):
    wf = advection(Nxi=Nxi, timesteps=Nt, cfl=8 / 6,
                   tilt_from=9 * Nt // 10, v_x=0.6, v_x_t=1.3, variance=0.5, offset=30)
else:  # Default is problem 2
    wf = advection(Nxi=Nxi, timesteps=Nt, cfl=8 / 6,
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
                        Neta=1, periodicity='Periodic', dx=wf.dx, dy=0)
# Convection matrix (Needs to be changed if the velocity is time dependent)
A_p = - wf.v_x[0] * Mat.Grad_Xi_kron
A_a = A_p.transpose()

# %% Solve the uncontrolled system
qs0 = IC_primal(wf.X, wf.Lxi, wf.offset, wf.variance)
qs_org = TI_primal(qs0, f, A_p, psi, wf.Nxi, wf.Nt, wf.dt)

qs_target = TI_primal_target(qs0, Mat.Grad_Xi_kron, wf.v_x_target, wf.Nxi, wf.Nt, wf.dt)

# Initial conditions for both primal and adjoint are defined here as they only need to defined once.
q0 = np.ascontiguousarray(IC_primal(wf.X, wf.Lxi, wf.offset, wf.variance))
q0_adj = np.ascontiguousarray(IC_adjoint(wf.X))

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
    'dt': wf.dt,
    'Nx': wf.Nxi,
    'Nt': wf.Nt,
    'n_c': n_c,
    'lamda': 1e-3,  # Regularization parameter
    'omega': 1,  # initial step size for gradient update
    'delta_conv': 1e-4,  # Convergence criteria
    'delta': 1e-2,  # Armijo constant
    'opt_iter': 200,  # Total iterations
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
    'adjoint_scheme': "RK4",  # Time integration scheme for adjoint equation
    'include_target_for_basis': False,  # True if we want to include the target state in the basis computation of
    # primal and adjoint
    'use_TWBT': True,  # This is true for TWBT and should be set False for using BB
}

# %% Prepare the directory for storing results
if kwargs['use_TWBT']:
    conv_crit = "TWBT"
else:
    conv_crit = "BB"
if kwargs['include_target_for_basis']:
    tar_for_bas = "include_target"
else:
    tar_for_bas = "no_target"

impath = "./data/sPODG_RA/" + conv_crit + "/" + tar_for_bas + "/" + "problem=" + str(
    problem) + "/" + TYPE + "=" + str(VAL) + "/"  # for data
immpath = "./plots/sPODG_RA/" + conv_crit + "/" + tar_for_bas + "/" + "problem=" + str(
    problem) + "/" + TYPE + "=" + str(VAL) + "/"  # for plots
os.makedirs(impath, exist_ok=True)

# %% ROM Variables
D = central_FDMatrix(order=6, Nx=wf.Nxi, dx=wf.dx)

# Generate the shift samples
delta_s = subsample(wf.X, num_sample=kwargs['shift_sample'])

# Extract transformation operators based on sub-sampled delta
T_delta, _ = get_T(delta_s, wf.X, wf.t, interp_order=kwargs['trafo_interp_order'])

delta_init = calc_shift(qs_org, q0, wf.X, wf.t)
_, T = get_T(delta_init, wf.X, wf.t, interp_order=kwargs['trafo_interp_order'])

print(2 * give_interpolation_error(qs_org, T))

# %% Basis computation and fixing upfront
'''
Forward calculation with FOM
'''
qs = TI_primal(q0, f, A_p, psi, wf.Nxi, wf.Nt, wf.dt)
z = calc_shift(qs, q0, wf.X, wf.t)
_, T = get_T(z, wf.X, wf.t, interp_order=kwargs['trafo_interp_order'])
if kwargs['include_target_for_basis']:
    qs_s = T.reverse(qs)
    qs_target_s = T.reverse(qs_target)
    qs_con = np.concatenate([qs_s, qs_target_s], axis=1)  # CHOOSE IF TO INCLUDE qs_target
else:
    qs_s = T.reverse(qs)
    qs_con = qs_s.copy()
V_p, qs_s_POD = compute_red_basis(qs_con, **kwargs)
Nm_p = V_p.shape[1]
err = np.linalg.norm(qs_con - qs_s_POD) / np.linalg.norm(qs_con)
print(f"Relative error for shifted primal: {err}, with Nm: {Nm_p}")

'''
Backward calculation with FOM
'''
qs_adj = TI_adjoint(q0_adj, qs, qs_target, None, A_a, None, wf.Nxi, wf.dx, wf.Nt, wf.dt, scheme="RK4")
if kwargs['include_target_for_basis']:
    qs_adj_s = T.reverse(qs_adj)
    qs_target_s = T.reverse(qs_target)
    qs_adj_con = np.concatenate([qs_adj_s, qs_target_s], axis=1)  # CHOOSE IF TO INCLUDE qs_target
else:
    qs_adj_s = T.reverse(qs_adj)
    qs_adj_con = qs_adj_s.copy()
V_a, qs_s_POD = compute_red_basis(qs_adj_con, **kwargs)
Nm_a = V_a.shape[1]
err = np.linalg.norm(qs_adj_con - qs_s_POD) / np.linalg.norm(qs_adj_con)
print(f"Relative error for shifted adjoint: {err}, with Nm: {Nm_a}")

# Initial condition for dynamical simulation
a_p = IC_primal_sPODG_FOTR(q0, V_p)
a_a = IC_adjoint_sPODG_FOTR(Nm_a, z[0, -1])

# Construct the primal system matrices for the sPOD-Galerkin approach
Vd_p, Wd_p, lhs_p, rhs_p, c_p = mat_primal_sPODG_FOTR(T_delta, V_p, A_p, psi, D, samples=kwargs['shift_sample'],
                                                      modes=Nm_p, Nx=kwargs['Nx'])
Vd_a, Wd_a, lhs_a, rhs_a, t_a = mat_adjoint_sPODG_FOTR(T_delta, V_a, A_a, D, Vd_p, samples=kwargs['shift_sample'],
                                                       modes_a=Nm_a, modes_p=Nm_p, Nx=kwargs['Nx'])

# %% For two-way backtracking line search
omega = 1

stag = False

start = time.time()
time_odeint_s = perf_counter()  # save running time
# %%
for opt_step in range(kwargs['opt_iter']):

    print("\n==============================")
    print("Optimization step: %d" % opt_step)

    '''
    Forward calculation
    '''
    as_, intIds, weights = TI_primal_sPODG_FOTR(lhs_p, rhs_p, c_p, a_p, f, delta_s, modes=Nm_p,
                                                Nt=kwargs['Nt'], dt=kwargs['dt'])

    '''
    Objective and costs for control
    '''
    # Compute the interpolation weight and the interval in which the shift lies corresponding to which we compute the
    # V_delta and W_delta matrices
    J, _ = Calc_Cost_sPODG(Vd_p, as_[:-1], qs_target, f, intIds, weights,
                           kwargs['dx'], kwargs['dt'], kwargs['lamda'])

    '''
    Backward calculation with reduced system
    '''
    as_adj = TI_adjoint_sPODG_FOTR(lhs_a, rhs_a, t_a, Vd_a, Wd_a, a_a, as_, qs_target, Nm_a, Nm_p, delta_s,
                                   kwargs['dx'], kwargs['Nt'], kwargs['dt'], kwargs['adjoint_scheme'])

    '''
     Update Control
    '''
    if opt_step == 0 or kwargs['use_TWBT']:
        fNew, J_opt, dL_du_Old, dL_du_norm, omega, _, stag = Update_Control_sPODG_FOTR_RA_TWBT(f, lhs_p, rhs_p, c_p,
                                                                                               a_p, Vd_a,
                                                                                               as_adj, qs_target,
                                                                                               delta_s, Vd_p, psi,
                                                                                               J, omega, Nm_p, intIds,
                                                                                               weights,
                                                                                               **kwargs)
    else:
        fNew, dL_du_Old, dL_du_norm = Update_Control_sPODG_FOTR_RA_BB(fOld, fNew, dL_du_Old, Vd_a,
                                                                      as_adj, psi, opt_step, intIds, weights,
                                                                      **kwargs)

    running_time.append(perf_counter() - time_odeint_s)

    # Saving previous controls for Barzilai Borwein step
    fOld = np.copy(f)
    f = np.copy(fNew)

    qs_opt_full = TI_primal(q0, f, A_p, psi, wf.Nxi, wf.Nt, wf.dt)
    JJ = Calc_Cost(qs_opt_full, qs_target, f,
                   kwargs['dx'], kwargs['dt'], kwargs['lamda'])

    J_opt_FOM_list.append(JJ)
    J_opt_list.append(J)
    dL_du_norm_list.append(dL_du_norm)
    dL_du_norm_ratio_list.append(dL_du_norm / dL_du_norm_list[0])

    print(
        f"J_ROM : {J}, J_FOM: {JJ}, ||dL_du|| = {dL_du_norm}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du_norm / dL_du_norm_list[0]}"
    )

    # Convergence criteria
    if opt_step == kwargs['opt_iter'] - 1:
        print("\n\n-------------------------------")
        print(
            f"WARNING... maximal number of steps reached, "
            f"J_ROM : {J}, J_FOM: {JJ}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du_norm / dL_du_norm_list[0]}"
        )
        f_last_valid = np.copy(f)
        break
    elif dL_du_norm / dL_du_norm_list[0] < kwargs['delta_conv']:
        print("\n\n-------------------------------")
        print(
            f"Optimization converged with, "
            f"J_ROM : {J}, J_FOM: {JJ}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du_norm / dL_du_norm_list[0]}"
        )
        f_last_valid = np.copy(f)
        break
    else:
        if opt_step == 0:
            if stag:
                print("\n-------------------------------")
                print(
                    f"Armijo Stagnated !!!!!! due to the step length being too low thus exiting at itr: {opt_step}")
                f_last_valid = np.copy(f)
                break
        else:
            dJ = (J_opt_list[-1] - J_opt_list[-2]) / J_opt_list[0]
            if abs(dJ) == 0:
                print(f"WARNING: dJ has turned close to 0...")
                f_last_valid = np.copy(f)
                break
            # Convergence criteria for BB
            if not kwargs['use_TWBT']:
                if len(J_opt_FOM_list) >= 5:
                    # Check for strict monotonic increase over the last 5 steps.
                    if all(J_opt_FOM_list[-5 + i] < J_opt_FOM_list[-5 + i + 1] for i in range(4)):
                        print("\n\n-------------------------------")
                        print(
                            f"Barzilai Borwein Diverged!!!!!! J_FOM increased for 5 consecutive optimization steps thus exiting at itr: {opt_step} with "
                            f"J_ROM : {J}, J_FOM: {JJ}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du_norm / dL_du_norm_list[0]}")
                        f_last_valid = np.copy(f)
                        break
                    if JJ > 1e6:
                        print("\n\n-------------------------------")
                        print(
                            f"Barzilai Borwein Failed!!!!!! J_FOM increased to unrealistic values, thus exiting "
                            f"at itr: {opt_step} with "
                            f"J_ROM : {J}, J_FOM: {JJ}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du_norm / dL_du_norm_list[0]}")
                        f_last_valid = np.copy(f)
                        break
            else:
                if stag:
                    print("\n-------------------------------")
                    print(
                        f"Armijo Stagnated !!!!!! due to the step length being too low thus exiting at itr: {opt_step}")
                    f_last_valid = np.copy(f)
                    break

# Compute the final state
qs_opt_full = TI_primal(q0, f_last_valid, A_p, psi, wf.Nxi, wf.Nt, wf.dt)
qs_adj = TI_adjoint(q0_adj, qs_opt_full, qs_target, None, A_a, None, wf.Nxi, wf.dx, wf.Nt, wf.dt,
                    scheme="RK4", opt_poly_jacobian=None)
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
pf = PlotFlow(wf.X, wf.t)
pf.plot1D(qs_org, name="qs_org", immpath=immpath)
pf.plot1D(qs_target, name="qs_target", immpath=immpath)
pf.plot1D(qs_opt_full, name="qs_opt", immpath=immpath)
pf.plot1D(qs_adj, name="qs_adj_opt", immpath=immpath)
pf.plot1D(f_opt, name="f_opt", immpath=immpath)
pf.plot1D_ROM_converg(J_opt_list, immpath=immpath)
