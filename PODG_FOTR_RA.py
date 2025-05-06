"""
This file is the version with ROM adjoint. It can handle both the scenarios.
1. Fixed tolerance
2. Fixed modes
"""
import scipy

from Coefficient_Matrix import CoefficientMatrix
from Costs import Calc_Cost_PODG, Calc_Cost
from FOM_solver import IC_primal, TI_primal, TI_primal_target, IC_adjoint, TI_adjoint
from Grads import Calc_Grad_PODG
from Helper import ControlSelectionMatrix_advection, compute_red_basis, L2norm_ROM
from PODG_solver import IC_primal_PODG_FOTR, mat_primal_PODG_FOTR, TI_primal_PODG_FOTR, IC_adjoint_PODG_FOTR, \
    mat_adjoint_PODG_FOTR, TI_adjoint_PODG_FOTR
from Update import Update_Control_PODG_FOTR_RA_TWBT, Update_Control_PODG_FOTR_RA_BB
from grid_params import advection
from Plots import PlotFlow
import numpy as np
import os
from time import perf_counter
import time
import argparse
from ast import literal_eval

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
f = np.zeros((n_c, wf.Nt))  # Initial guess for the control

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

# %% Optimal control
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
    'beta': 1 / 2,  # Beta factor for two-way backtracking line search
    'verbose': True,  # Print options
    'base_tol': tol,  # Base tolerance for selecting number of modes (main variable for truncation)
    'omega_cutoff': 1e-10,  # Below this cutoff the Armijo and Backtracking should exit the update loop
    'threshold': threshold,
    # Variable for selecting threshold based truncation or mode based. "TRUE" for threshold based
    # "FALSE" for mode based.
    'Nm_p': modes[0],  # Number of modes for truncation if threshold selected to False.
    'Nm_a': modes[1],  # Number of modes for truncation if threshold selected to False.
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

impath = args.dir_prefix + "/data/PODG_FOTR_RA/" + conv_crit + "/" + tar_for_bas + "/" + "problem=" + str(
    problem) + "/" + TYPE + "=" + str(
    VAL) + "/"  # for data
immpath = args.dir_prefix + "/plots/PODG_FOTR_RA/" + conv_crit + "/" + tar_for_bas + "/" + "problem=" + str(
    problem) + "/" + TYPE + "=" + str(
    VAL) + "/"  # for plots
os.makedirs(impath, exist_ok=True)


# %% Basis computation and fixing upfront
'''
Forward calculation with FOM
'''
qs = TI_primal(q0, f, A_p, psi, wf.Nxi, wf.Nt, wf.dt)
if kwargs['include_target_for_basis']:
    qs_con = np.concatenate([qs, qs_target], axis=1)  # CHOOSE IF TO INCLUDE qs_target
else:
    qs_con = qs.copy()
V_p, qs_POD = compute_red_basis(qs_con, equation="primal", **kwargs)
Nm_p = V_p.shape[1]
err = np.linalg.norm(qs_con - qs_POD) / np.linalg.norm(qs_con)
print(f"Relative error for primal: {err}, with Nm: {Nm_p}")

'''
Backward calculation with FOM
'''
qs_adj = TI_adjoint(q0_adj, qs, qs_target, None, A_a, None, wf.Nxi, wf.dx, wf.Nt, wf.dt, scheme="RK4")
if kwargs['include_target_for_basis']:
    qs_adj_con = np.concatenate([qs_adj, qs_target], axis=1)  # CHOOSE IF TO INCLUDE qs_target
else:
    qs_adj_con = qs_adj.copy()
V_a, qs_POD_adj = compute_red_basis(qs_adj_con, equation="adjoint", **kwargs)
Nm_a = V_a.shape[1]
err = np.linalg.norm(qs_adj_con - qs_POD_adj) / np.linalg.norm(qs_adj_con)
print(f"Relative error for adjoint: {err}, with Nm: {Nm_a}")

# Initial condition for dynamical simulation
a_p = IC_primal_PODG_FOTR(V_p, q0)
a_a = IC_adjoint_PODG_FOTR(Nm_a)

# Construct the primal and adjoint system matrices for the POD-Galerkin approach
Ar_p, psir_p = mat_primal_PODG_FOTR(A_p, V_p, psi)
Ar_a, V_aTV_p, Tarr_a, psir_a = mat_adjoint_PODG_FOTR(A_a, V_a, V_p, qs_target, psi)

# %% Select the LU pre-factors for the inverse of mass matrix for linear solve of adjoint equation
if kwargs['adjoint_scheme'] == "RK4":
    M_f = None
    A_f = Ar_a.copy()
    LU_M_f = None
    Df = None
elif kwargs['adjoint_scheme'] == "implicit_midpoint":
    M_f = np.eye(Nm_a) + (- kwargs['dt']) / 2 * Ar_a
    A_f = np.eye(Nm_a) - (- kwargs['dt']) / 2 * Ar_a
    LU_M_f = scipy.linalg.lu_factor(M_f)
elif kwargs['adjoint_scheme'] == "DIRK":
    M_f = np.eye(Nm_a) + (- kwargs['dt']) / 4 * Ar_a
    A_f = Ar_a.copy()
    LU_M_f = scipy.linalg.lu_factor(M_f)
elif kwargs['adjoint_scheme'] == "BDF2":
    M_f = 3.0 * np.eye(Nm_a) + 2.0 * (- kwargs['dt']) * Ar_a
    A_f = Ar_a.copy()
    LU_M_f = scipy.linalg.lu_factor(M_f)
else:
    kwargs['adjoint_scheme'] = "RK4"
    M_f = None
    A_f = Ar_a.copy()
    LU_M_f = None

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
    Forward calculation with reduced system
    '''
    as_ = TI_primal_PODG_FOTR(a_p, f, Ar_p, psir_p, wf.Nt, wf.dt)

    '''
    Objective and costs for control
    '''
    J = Calc_Cost_PODG(V_p, as_, qs_target, f, kwargs['dx'], kwargs['dt'], kwargs['lamda'])

    qs_approx = V_p @ as_
    qs_opt_full = TI_primal(q0, f, A_p, psi, wf.Nxi, wf.Nt, wf.dt)
    running_online_error_p.append(np.linalg.norm(qs_opt_full - qs_approx) / np.linalg.norm(qs_opt_full))

    JJ = Calc_Cost(qs_opt_full, qs_target, f, kwargs['dx'], kwargs['dt'], kwargs['lamda'])
    J_opt_FOM_list.append(JJ)
    J_opt_list.append(J)

    '''
    Backward calculation with reduced system
    '''
    as_adj = TI_adjoint_PODG_FOTR(a_a, as_, M_f, A_f, LU_M_f, V_aTV_p, Tarr_a, kwargs['Nt'], kwargs['dt'], kwargs['dx'],
                                  kwargs['adjoint_scheme'])

    qs_adj_approx = V_a @ as_adj
    qs_adj_full = TI_adjoint(q0_adj, qs_opt_full, qs_target, None, A_a, None, wf.Nxi, wf.dx, wf.Nt, wf.dt, scheme="RK4")
    running_online_error_a.append(np.linalg.norm(qs_adj_full - qs_adj_approx) / np.linalg.norm(qs_adj_full))

    '''
    Update Control
    '''
    # Compute the gradient
    dL_du = Calc_Grad_PODG(psir_a, f, as_adj, kwargs['lamda'])
    dL_du_norm_square = L2norm_ROM(dL_du, kwargs['dt'])
    dL_du_norm = np.sqrt(dL_du_norm_square)

    dL_du_norm_list.append(dL_du_norm)
    dL_du_norm_ratio_list.append(dL_du_norm / dL_du_norm_list[0])

    if opt_step == 0:
        print(f"TWBT acting.....")
        fNew, J_opt, omega_twbt, stag = Update_Control_PODG_FOTR_RA_TWBT(f, a_p, qs_target,
                                                                         V_p, Ar_p,
                                                                         psir_p, J, omega_twbt, dL_du,
                                                                         dL_du_norm_square,
                                                                         **kwargs)
    else:
        if conv_crit == "TWBT+BB" and dL_du_norm / dL_du_norm_list[0] < 5e-3:
            print(f"BB acting.....")
            fNew, omega_bb = Update_Control_PODG_FOTR_RA_BB(fOld, fNew, dL_du_Old, dL_du, opt_step, **kwargs)
        else:
            print(f"TWBT acting.....")
            fNew, J_opt, omega_twbt, stag = Update_Control_PODG_FOTR_RA_TWBT(f, a_p, qs_target,
                                                                             V_p, Ar_p,
                                                                             psir_p, J, omega_twbt, dL_du,
                                                                             dL_du_norm_square,
                                                                             **kwargs)

    running_time.append(perf_counter() - time_odeint_s)

    # Saving previous controls for Barzilai Borwein step
    fOld = np.copy(f)
    f = np.copy(fNew)
    dL_du_Old = np.copy(dL_du)

    print(
        f"J_ROM: {J}, J_FOM: {JJ}, ||dL_du|| = {dL_du_norm}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du_norm / dL_du_norm_list[0]}"
    )

    # Convergence criteria
    if opt_step == kwargs['opt_iter'] - 1:
        print("\n\n-------------------------------")
        print(
            f"WARNING... maximal number of steps reached, "
            f"J_ROM: {J}, J_FOM: {JJ}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du_norm / dL_du_norm_list[0]}"
        )
        break
    elif dL_du_norm / dL_du_norm_list[0] < kwargs['delta_conv']:
        print("\n\n-------------------------------")
        print(
            f"Optimization converged with, "
            f"J_ROM: {J}, J_FOM: {JJ}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du_norm / dL_du_norm_list[0]}"
        )
        break
    else:
        if opt_step == 0:
            if stag:
                print("\n\n-------------------------------")
                print(
                    f"Armijo Stagnated !!!!!! due to the step length being too low thus exiting at itr: {opt_step} with "
                    f"J_ROM: {J}, J_FOM: {JJ}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du_norm / dL_du_norm_list[0]}")
                f = np.copy(fOld)
                break
        else:
            dJ = (J_opt_list[-1] - J_opt_list[-2]) / J_opt_list[0]
            if abs(dJ) == 0:
                print(f"WARNING: dJ has turned close to 0...")
                break
            if stag:
                print("\n-------------------------------")
                print(
                    f"TWBT Armijo Stagnated !!!!!! due to the step length being too low thus exiting at itr: {opt_step} with "
                    f"J_ROM: {J}, J_FOM: {JJ}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du_norm / dL_du_norm_list[0]}")
                f = np.copy(fOld)
                break
            # Convergence criteria for BB
            if conv_crit == "TWBT+BB":
                if JJ > 1e6 or omega_bb < kwargs['omega_cutoff']:
                    print("\n\n-------------------------------")
                    print(
                        f"Barzilai Borwein acceleration failed!!!!!! J_FOM increased to unrealistic values or the omega went below cutoff or even negative, thus exiting "
                        f"at itr: {opt_step} with "
                        f"J_ROM: {J}, J_FOM: {JJ}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du_norm / dL_du_norm_list[0]}")
                    f = np.copy(fOld)
                    break

# Compute the final state
qs_opt_full = TI_primal(q0, f, A_p, psi, wf.Nxi, wf.Nt, wf.dt)
qs_adj = TI_adjoint(q0_adj, qs_opt_full, qs_target, None, A_a, None, wf.Nxi, wf.dx, wf.Nt, wf.dt,
                    scheme="RK4", opt_poly_jacobian=None)
f_opt = psi @ f

# Compute the cost with the optimal control
J = Calc_Cost(qs_opt_full, qs_target, f, kwargs['dx'], kwargs['dt'], kwargs['lamda'])
print("\n")
print(f"J with respect to the optimal control for FOM: {J}")

end = time.time()
print("\n")
print("Total time elapsed = %1.3f" % (end - start))

# %%
# Save the convergence lists
np.save(impath + 'J_opt_list.npy', J_opt_list)
np.save(impath + 'J_opt_FOM_list.npy', J_opt_FOM_list)
np.save(impath + 'trunc_modes_list_p.npy', trunc_modes_list_p)
np.save(impath + 'trunc_modes_list_a.npy', trunc_modes_list_a)
np.save(impath + 'running_time.npy', running_time)
np.save(impath + 'running_online_error_p.npy', running_online_error_p)
np.save(impath + 'running_online_error_a.npy', running_online_error_a)

# Save the optimized solution
# np.save(impath + 'qs_opt.npy', qs_opt_full)
# np.save(impath + 'qs_adj_opt.npy', qs_adj)
# np.save(impath + 'f_opt.npy', f_opt)
# np.save(impath + 'f_opt_low.npy', f)

# %%
# Plot the results
pf = PlotFlow(wf.X, wf.t)
pf.plot1D(qs_org, name="qs_org", immpath=immpath)
pf.plot1D(qs_target, name="qs_target", immpath=immpath)
pf.plot1D(qs_opt_full, name="qs_opt", immpath=immpath)
pf.plot1D(qs_adj, name="qs_adj_opt", immpath=immpath)
pf.plot1D(f_opt, name="f_opt", immpath=immpath)
pf.plot1D_ROM_converg(J_opt_list, immpath=immpath)
