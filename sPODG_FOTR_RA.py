"""
This file is the version with FOM adjoint. It can handle both the scenarios.
1. Fixed tolerance
2. Fixed modes
"""

from Coefficient_Matrix import CoefficientMatrix
from Cubic_spline import give_spline_coefficient_matrices, construct_spline_coeffs_multiple, \
    shift_matrix_precomputed_coeffs_multiple
from FOM_solver import IC_primal, TI_primal, TI_primal_target, IC_adjoint, TI_adjoint
from Grads import Calc_Grad_sPODG
from Update import Update_Control_sPODG_FOTR_RA_TWBT, Update_Control_sPODG_FOTR_RA_BB
from grid_params import advection
from Plots import PlotFlow
from Helper import ControlSelectionMatrix_advection, compute_red_basis, calc_shift, L2norm_ROM
from Helper_sPODG import subsample, get_T, central_FDMatrix, make_V_W_delta, make_V_W_delta_CubSpl
from Costs import Calc_Cost_sPODG, Calc_Cost
import os
from time import perf_counter
import numpy as np
import time
import scipy.sparse as sp
import argparse
from scipy.sparse import csc_matrix
from ast import literal_eval

import sys

from sPODG_solver import IC_primal_sPODG_FOTR, mat_primal_sPODG_FOTR, TI_primal_sPODG_FOTR, IC_adjoint_sPODG_FOTR, \
    mat_adjoint_sPODG_FOTR, TI_adjoint_sPODG_FOTR

sys.path.append('./sPOD/lib/')
from sPOD_algo import give_interpolation_error

parser = argparse.ArgumentParser(description="Input the variables for running the script.")
parser.add_argument("problem", type=int, choices=[1, 2, 3], help="Specify the problem number (1, 2, or 3)")
parser.add_argument("conv_accel", type=literal_eval, choices=[True, False],
                    help="Specify if to use BB as acceleration for the already running TWBT("
                         "True or False)")
parser.add_argument("target_for_basis", type=literal_eval, choices=[True, False], help="Specify if you want to "
                                                                                       "include the"
                                                                                       "target state for computing "
                                                                                       "the basis ("
                                                                                       "True or False)")
parser.add_argument("interp_scheme", type=str, choices=["Lagr", "CubSpl"],
                    help="Specify the Interpolation scheme to use ("
                         "Lagr or CubSpl)")
parser.add_argument("N_iter", type=int, help="Enter the number of optimization iterations")
parser.add_argument("dir_prefix", type=str, choices=[".", "/work/burela"],
                    help="Specify the directory prefix for proper storage of the files")
parser.add_argument("--modes", type=int, nargs=2,
                    help="Enter the modes for both the primal and adjoint systems e.g., --modes 3 5")
parser.add_argument("--tol", type=float, help="Enter the tolerance level for tolerance test")
args = parser.parse_args()

print("\n")
print(f"Solving problem: {args.problem}")
print(f"Choosing BB accelerated convergence: {args.conv_accel}")
print(f"Using target state for basis computation: {args.target_for_basis}")
print(f"Interpolation scheme to be used for shift matrix construction: {args.interp_scheme}")
print(f"Type of basis computation: fixed")

if args.conv_accel is False:
    conv_crit = "TWBT"
elif args.conv_accel is True:
    conv_crit = "TWBT+BB"
    print("\n---------------------")
    print(f"BB acceleration is only activated once the relative normed gradient has reached low enough value with the TWBT")
    print("\n---------------------")
else:
    conv_crit = "TWBT"  # Default is just TWBT with no acceleration
problem = args.problem
target_for_basis = args.target_for_basis
interp_scheme = args.interp_scheme

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
    modes = (None, None)
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
trunc_modes_list_p = []  # Number of modes needed to reach the offline error
trunc_modes_list_a = []  # Number of modes needed to reach the offline error
running_online_error_p = []  # Online error for tracking primal approximation
running_online_error_a = []  # Online error for tracking adjoint approximation

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
    'opt_iter': args.N_iter,  # Total iterations
    'shift_sample': 800,  # Number of samples for shift interpolation
    'beta': 1 / 2,  # Beta factor for two-way backtracking line search
    'verbose': True,  # Print options
    'base_tol': tol,  # Base tolerance for selecting number of modes (main variable for truncation)
    'omega_cutoff': 1e-10,  # Below this cutoff the Armijo and Backtracking should exit the update loop
    'threshold': threshold,
    # Variable for selecting threshold based truncation or mode based. "TRUE" for threshold based
    # "FALSE" for mode based.
    'Nm_p': modes[0],  # Number of modes for truncation if threshold selected to False.
    'Nm_a': modes[1],  # Number of modes for truncation if threshold selected to False.
    'interp_scheme': interp_scheme,  # Either Lagrange interpolation or Cubic spline
    'trafo_interp_order': 5,  # Order of the polynomial interpolation for the transformation operators
    'adjoint_scheme': "RK4",  # Time integration scheme for adjoint equation
    'include_target_for_basis': target_for_basis,
    # True if we want to include the target state in the basis computation of
    # primal and adjoint
}

# %% Prepare the directory for storing results
if kwargs['include_target_for_basis']:
    tar_for_bas = "include_target"
else:
    tar_for_bas = "no_target"

impath = args.dir_prefix + "/data/sPODG_FOTR_RA/" + conv_crit + "/" + tar_for_bas + "/" + interp_scheme + "/" + "problem=" + str(
    problem) + "/" + TYPE + "=" + str(VAL) + "/"  # for data
immpath = args.dir_prefix + "/plots/sPODG_FOTR_RA/" + conv_crit + "/" + tar_for_bas + "/" + interp_scheme + "/" + "problem=" + str(
    problem) + "/" + TYPE + "=" + str(VAL) + "/"  # for plots
os.makedirs(impath, exist_ok=True)

# %% ROM Variables
D = central_FDMatrix(order=6, Nx=wf.Nxi, dx=wf.dx)

# Generate the shift samples
delta_s = subsample(wf.X, num_sample=kwargs['shift_sample'])

if kwargs['interp_scheme'] == "Lagr":
    # Extract transformation operators based on sub-sampled delta
    T_delta, _ = get_T(delta_s, wf.X, wf.t, interp_order=kwargs['trafo_interp_order'])
else:
    # Calculate the constant spline coefficient matrices (only needed once)
    A1, D1, D2, R = give_spline_coefficient_matrices(kwargs['Nx'])

# %% Basis computation and fixing upfront
'''
Forward calculation with FOM
'''
qs = TI_primal(q0, f, A_p, psi, wf.Nxi, wf.Nt, wf.dt)
z = calc_shift(qs, q0, wf.X, wf.t)

if kwargs['interp_scheme'] == "Lagr":
    _, T = get_T(z, wf.X, wf.t, interp_order=kwargs['trafo_interp_order'])
    qs_s = T.reverse(qs)
    if kwargs['include_target_for_basis']:
        qs_target_s = T.reverse(qs_target)
        qs_con = np.concatenate([qs_s, qs_target_s], axis=1)  # CHOOSE IF TO INCLUDE qs_target
    else:
        qs_con = qs_s.copy()
else:
    # Construct the spline coefficients for the qs
    b, c, d = construct_spline_coeffs_multiple(qs, A1, D1, D2, R, kwargs['dx'])
    qs_s = shift_matrix_precomputed_coeffs_multiple(qs, z[0], b, c, d, kwargs['Nx'], kwargs['dx'])
    if kwargs['include_target_for_basis']:
        b, c, d = construct_spline_coeffs_multiple(qs_target, A1, D1, D2, R, kwargs['dx'])
        qs_target_s = shift_matrix_precomputed_coeffs_multiple(qs_target, z[0], b, c, d, kwargs['Nx'], kwargs['dx'])
        qs_con = np.concatenate([qs_s, qs_target_s], axis=1)  # CHOOSE IF TO INCLUDE qs_target
    else:
        qs_con = qs_s.copy()

V_p, qs_s_POD = compute_red_basis(qs_con, equation="primal", **kwargs)
Nm_p = V_p.shape[1]
err = np.linalg.norm(qs_con - qs_s_POD) / np.linalg.norm(qs_con)
print(f"Relative error for shifted primal: {err}, with Nm: {Nm_p}")

'''
Backward calculation with FOM
'''
qs_adj = TI_adjoint(q0_adj, qs, qs_target, None, A_a, None, wf.Nxi, wf.dx, wf.Nt, wf.dt, scheme="RK4")
if kwargs['interp_scheme'] == "Lagr":
    qs_adj_s = T.reverse(qs_adj)
    if kwargs['include_target_for_basis']:
        qs_adj_con = np.concatenate([qs_adj_s, qs_target_s], axis=1)  # CHOOSE IF TO INCLUDE qs_target
    else:
        qs_adj_con = qs_adj_s.copy()
else:
    # Construct the spline coefficients for the qs
    b, c, d = construct_spline_coeffs_multiple(qs_adj, A1, D1, D2, R, kwargs['dx'])
    qs_adj_s = shift_matrix_precomputed_coeffs_multiple(qs_adj, z[0], b, c, d, kwargs['Nx'], kwargs['dx'])
    if kwargs['include_target_for_basis']:
        qs_adj_con = np.concatenate([qs_adj_s, qs_target_s], axis=1)  # CHOOSE IF TO INCLUDE qs_target
    else:
        qs_adj_con = qs_adj_s.copy()

V_a, qs_s_POD = compute_red_basis(qs_adj_con, equation="adjoint", **kwargs)
Nm_a = V_a.shape[1]
err = np.linalg.norm(qs_adj_con - qs_s_POD) / np.linalg.norm(qs_adj_con)
print(f"Relative error for shifted adjoint: {err}, with Nm: {Nm_a}")

# Initial condition for dynamical simulation
a_p = IC_primal_sPODG_FOTR(q0, V_p)

# Construct the primal system matrices for the sPOD-Galerkin approach
if kwargs['interp_scheme'] == "Lagr":
    Vd_p, Wd_p = make_V_W_delta(V_p, T_delta, D, kwargs['shift_sample'], kwargs['Nx'], Nm_p)
    Vd_a, Wd_a = make_V_W_delta(V_a, T_delta, D, kwargs['shift_sample'], kwargs['Nx'], Nm_a)
else:
    Vd_p, Wd_p = make_V_W_delta_CubSpl(V_p, delta_s, A1, D1, D2, R, kwargs['shift_sample'], kwargs['Nx'], kwargs['dx'],
                                       Nm_p)
    Vd_a, Wd_a = make_V_W_delta_CubSpl(V_a, delta_s, A1, D1, D2, R, kwargs['shift_sample'], kwargs['Nx'], kwargs['dx'],
                                       Nm_a)

lhs_p, rhs_p, c_p = mat_primal_sPODG_FOTR(Vd_p, Wd_p, A_p, psi, samples=kwargs['shift_sample'], modes=Nm_p)
lhs_a, rhs_a, t_a = mat_adjoint_sPODG_FOTR(Vd_a, Wd_a, A_a, Vd_p, samples=kwargs['shift_sample'],
                                           modes_a=Nm_a, modes_p=Nm_p)

# %% For two-way backtracking line search
omega_twbt = 1
omega_bb = 1

stag = False

start = time.time()
time_odeint_s = perf_counter()  # save running time
# %%
for opt_step in range(kwargs['opt_iter']):

    print("\n==============================")
    print("Optimization step: %d" % opt_step)

    trunc_modes_list_p.append(Nm_p)
    trunc_modes_list_a.append(Nm_a)

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
    J, qs_approx = Calc_Cost_sPODG(Vd_p, as_[:-1], qs_target, f, intIds, weights,
                           kwargs['dx'], kwargs['dt'], kwargs['lamda'])


    qs_opt_full = TI_primal(q0, f, A_p, psi, wf.Nxi, wf.Nt, wf.dt)
    running_online_error_p.append(np.linalg.norm(qs_opt_full - qs_approx) / np.linalg.norm(qs_opt_full))


    JJ = Calc_Cost(qs_opt_full, qs_target, f,
                   kwargs['dx'], kwargs['dt'], kwargs['lamda'])
    J_opt_FOM_list.append(JJ)
    J_opt_list.append(J)

    '''
    Backward calculation with reduced system
    '''
    a_a = IC_adjoint_sPODG_FOTR(Nm_a, as_[-1, -1])
    as_adj = TI_adjoint_sPODG_FOTR(lhs_a, rhs_a, t_a, Vd_a, Wd_a, a_a, as_, qs_target, Nm_a, Nm_p, delta_s,
                                   kwargs['dx'], kwargs['Nt'], kwargs['dt'], kwargs['adjoint_scheme'])

    '''
     Update Control
    '''
    dL_du, qs_adj_approx = Calc_Grad_sPODG(psi, f, Vd_a, as_adj[:-1], intIds, weights, kwargs['lamda'])
    dL_du_norm_square = L2norm_ROM(dL_du, kwargs['dt'])
    dL_du_norm = np.sqrt(dL_du_norm_square)

    dL_du_norm_list.append(dL_du_norm)
    dL_du_norm_ratio_list.append(dL_du_norm / dL_du_norm_list[0])



    qs_adj_full = TI_adjoint(q0_adj, qs_opt_full, qs_target, None, A_a, None, wf.Nxi, wf.dx, wf.Nt, wf.dt, scheme="RK4")
    running_online_error_a.append(np.linalg.norm(qs_adj_full - qs_adj_approx) / np.linalg.norm(qs_adj_full))


    if opt_step == 0:
        print(f"TWBT acting.....")
        fNew, J_opt, omega_twbt, _, stag = Update_Control_sPODG_FOTR_RA_TWBT(f, lhs_p, rhs_p, c_p,
                                                                             a_p, qs_target,
                                                                             delta_s, Vd_p,
                                                                             J, omega_twbt, Nm_p,
                                                                             dL_du,
                                                                             dL_du_norm_square,
                                                                             **kwargs)
    else:
        if conv_crit == "TWBT+BB" and dL_du_norm / dL_du_norm_list[0] < 5e-3:
            print(f"BB acting.....")
            fNew, omega_bb = Update_Control_sPODG_FOTR_RA_BB(fOld, fNew, dL_du_Old, dL_du, opt_step,
                                                             **kwargs)
        else:
            print(f"TWBT acting.....")
            fNew, J_opt, omega_twbt, _, stag = Update_Control_sPODG_FOTR_RA_TWBT(f, lhs_p, rhs_p, c_p,
                                                                                 a_p, qs_target,
                                                                                 delta_s, Vd_p,
                                                                                 J, omega_twbt, Nm_p,
                                                                                 dL_du,
                                                                                 dL_du_norm_square,
                                                                                 **kwargs)

    running_time.append(perf_counter() - time_odeint_s)

    # Saving previous controls for Barzilai Borwein step
    fOld = np.copy(f)
    f = np.copy(fNew)
    dL_du_Old = np.copy(dL_du)

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
                f_last_valid = np.copy(fOld)
                f = np.copy(fOld)
                break
        else:
            dJ = (J_opt_list[-1] - J_opt_list[-2]) / J_opt_list[0]
            if abs(dJ) == 0:
                print(f"WARNING: dJ has turned close to 0...")
                f_last_valid = np.copy(f)
                break
            if stag:
                print("\n-------------------------------")
                print(
                    f"TWBT Armijo Stagnated !!!!!! due to the step length being too low thus exiting at itr: {opt_step} with "
                    f"J_ROM: {J}, J_FOM: {JJ}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du_norm / dL_du_norm_list[0]}")
                f_last_valid = np.copy(fOld)
                break
            # Convergence criteria for BB
            if conv_crit == "TWBT+BB":
                if JJ > 1e6 or omega_bb < kwargs['omega_cutoff']:
                    print("\n\n-------------------------------")
                    print(
                        f"Barzilai Borwein acceleration failed!!!!!! J_FOM increased to unrealistic values or the omega went below cutoff or even negative, thus exiting "
                        f"at itr: {opt_step} with "
                        f"J_ROM: {J}, J_FOM: {JJ}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du_norm / dL_du_norm_list[0]}")
                    f_last_valid = np.copy(fOld)
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
np.save(impath + 'trunc_modes_list_p.npy', trunc_modes_list_p)
np.save(impath + 'trunc_modes_list_a.npy', trunc_modes_list_a)
np.save(impath + 'running_time.npy', running_time)
np.save(impath + 'running_online_error_p.npy', running_online_error_p)
np.save(impath + 'running_online_error_a.npy', running_online_error_a)

# Save the optimized solution
# np.save(impath + 'qs_opt.npy', qs_opt_full)
# np.save(impath + 'qs_adj_opt.npy', qs_adj)
# np.save(impath + 'f_opt.npy', f_opt)
# np.save(impath + 'f_opt_low.npy', f_last_valid)

# %%
# Plot the results
pf = PlotFlow(wf.X, wf.t)
pf.plot1D(qs_org, name="qs_org", immpath=immpath)
pf.plot1D(qs_target, name="qs_target", immpath=immpath)
pf.plot1D(qs_opt_full, name="qs_opt", immpath=immpath)
pf.plot1D(qs_adj, name="qs_adj_opt", immpath=immpath)
pf.plot1D(f_opt, name="f_opt", immpath=immpath)
pf.plot1D_ROM_converg(J_opt_list, immpath=immpath)
