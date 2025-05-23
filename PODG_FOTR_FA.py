"""
This file is the version with FOM adjoint. It can handle both the scenarios.
1. Fixed tolerance
2. Fixed modes
"""
from scipy import sparse

from Coefficient_Matrix import CoefficientMatrix
from Costs import Calc_Cost_PODG, Calc_Cost
from FOM_solver import IC_primal, TI_primal, TI_primal_target, IC_adjoint, TI_adjoint
from Helper import ControlSelectionMatrix_advection, compute_red_basis
from PODG_solver import IC_primal_PODG_FOTR, mat_primal_PODG_FOTR, TI_primal_PODG_FOTR
from TI_schemes import DF_start_FOM
from Update import Update_Control_PODG_FOTR_FA_TWBT
from grid_params import advection
from Plots import PlotFlow
import numpy as np
import os
from time import perf_counter
import time
import scipy.sparse as sp
import argparse
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import inv
from scipy.sparse import csc_matrix

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

impath = "./data/PODG/problem=" + str(problem) + "/" + TYPE + "=" + str(VAL) + "/"  # for data
immpath = "./plots/PODG/problem=" + str(problem) + "/" + TYPE + "=" + str(VAL) + "/"  # for plots
os.makedirs(impath, exist_ok=True)

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


if problem == 1:    # Thick wave params
    wf = advection(Nxi=Nxi, timesteps=Nt, cfl=8 / 6,
                   tilt_from=3 * Nt // 4, v_x=0.5, v_x_t=1.0, variance=7, offset=12)
elif problem == 2:    # Sharp wave params (earlier kink):
    wf = advection(Nxi=Nxi, timesteps=Nt, cfl=8 / 6,
                   tilt_from=3 * Nt // 4, v_x=0.55, v_x_t=1.0, variance=0.5, offset=30)
elif problem == 3:    # Sharp wave params (later kink):
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

#%% Assemble the linear operators
Mat = CoefficientMatrix(orderDerivative=wf.firstderivativeOrder, Nxi=wf.Nxi,
                        Neta=1, periodicity='Periodic', dx=wf.dx, dy=0)
# Convection matrix (Needs to be changed if the velocity is time dependent)
A_p = - wf.v_x[0] * Mat.Grad_Xi_kron
A_a = A_p.transpose()

#%% Solve the uncontrolled system
qs0 = IC_primal(wf.X, wf.Lxi, wf.offset, wf.variance)
qs_org = TI_primal(qs0, f, A_p, psi, wf.Nxi, wf.Nt, wf.dt)
np.save(impath + 'qs_org.npy', qs_org)

qs_target = TI_primal_target(qs0, Mat.Grad_Xi_kron, wf.v_x_target, wf.Nxi, wf.Nt, wf.dt)
np.save(impath + 'qs_target.npy', qs_target)

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
trunc_modes_list = []  # Number of modes needed to reach the offline error

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
    'opt_iter': 10,  # Total iterations
    'beta': 1 / 2,  # Beta factor for two-way backtracking line search
    'verbose': True,  # Print options
    'base_tol': tol,  # Base tolerance for selecting number of modes (main variable for truncation)
    'omega_cutoff': 1e-10,  # Below this cutoff the Armijo and Backtracking should exit the update loop
    'threshold': threshold,
    # Variable for selecting threshold based truncation or mode based. "TRUE" for threshold based
    # "FALSE" for mode based.
    'Nm_p': modes,  # Number of modes for truncation if threshold selected to False.
    'adjoint_scheme': "implicit_midpoint"  # Time integration scheme for adjoint equation
}

#%% Select the LU pre-factors for the inverse of mass matrix for linear solve of adjoint equation
if kwargs['adjoint_scheme'] == "RK4":
    M_f = None
    A_f = A_a.copy()
    LU_M_f = None
    Df = None
elif kwargs['adjoint_scheme'] == "implicit_midpoint":
    M_f = sparse.eye(kwargs['Nx'], format="csc") + (- kwargs['dt']) / 2 * A_a
    A_f = sparse.eye(kwargs['Nx'], format="csc") - (- kwargs['dt']) / 2 * A_a
    LU_M_f = splu(M_f)
    Df = None
elif kwargs['adjoint_scheme'] == "DIRK":
    M_f = sparse.eye(kwargs['Nx'], format="csc") + (- kwargs['dt']) / 4 * A_a
    A_f = A_a.copy()
    LU_M_f = splu(M_f)
    Df = None
elif kwargs['adjoint_scheme'] == "BDF2":
    M_f = 3.0 * sparse.eye(kwargs['Nx'], format="csc") + 2.0 * (- kwargs['dt']) * A_a
    A_f = A_a.copy()
    LU_M_f = splu(M_f)
    Df = None
elif kwargs['adjoint_scheme'] == "BDF3":
    M_f = 11.0 * sparse.eye(kwargs['Nx'], format="csc") + 6.0 * (- kwargs['dt']) * A_a
    A_f = A_a.copy()
    LU_M_f = splu(M_f)
    Df = csc_matrix(DF_start_FOM(A_a.todense(), kwargs['Nx'], - kwargs['dt']))
elif kwargs['adjoint_scheme'] == "BDF4":
    M_f = 25.0 * sparse.eye(kwargs['Nx'], format="csc") + 12.0 * (- kwargs['dt']) * A_a
    A_f = A_a.copy()
    LU_M_f = splu(M_f)
    Df = csc_matrix(DF_start_FOM(A_a.todense(), kwargs['Nx'], - kwargs['dt']).tocsc())
else:
    kwargs['adjoint_scheme'] = "RK4"
    M_f = None
    A_f = A_a.copy()
    LU_M_f = None
    Df = None

#%% For two-way backtracking line search
omega = 1

stag = False

start = time.time()
time_odeint_s = perf_counter()  # save running time
# %%
for opt_step in range(kwargs['opt_iter']):

    print("\n==============================")
    print("Optimization step: %d" % opt_step)

    '''
    Forward calculation with primal for basis update
    '''
    qs = TI_primal(q0, f, A_p, psi, wf.Nxi, wf.Nt, wf.dt)

    V_p, qs_POD = compute_red_basis(qs, **kwargs)
    Nm = V_p.shape[1]
    err = np.linalg.norm(qs - qs_POD) / np.linalg.norm(qs)
    print(f"Relative error for primal: {err}, with Nm: {Nm}")

    err_list.append(err)
    trunc_modes_list.append(Nm)

    # Initial condition for dynamical simulation
    a_p = IC_primal_PODG_FOTR(V_p, q0)

    # Construct the primal system matrices for the POD-Galerkin approach
    Ar_p, psir_p = mat_primal_PODG_FOTR(A_p, V_p, psi)

    '''
    Forward calculation with reduced system
    '''
    as_ = TI_primal_PODG_FOTR(a_p, f, Ar_p, psir_p, wf.Nt, wf.dt)

    '''
    Objective and costs for control
    '''
    J = Calc_Cost_PODG(V_p, as_, qs_target, f,
                       kwargs['dx'], kwargs['dt'], kwargs['lamda'])

    '''
    Backward calculation with FOM system
    '''
    qs_adj = TI_adjoint(q0_adj, qs, qs_target, M_f, A_f, LU_M_f, wf.Nxi, wf.dx, wf.Nt, wf.dt,
                        scheme=kwargs['adjoint_scheme'], opt_poly_jacobian=Df)

    '''
     Update Control
    '''
    f, J_opt, _, dL_du_norm, omega, stag = Update_Control_PODG_FOTR_FA_TWBT(f, a_p, qs_adj, qs_target, V_p, Ar_p,
                                                                            psir_p, psi, J, omega,
                                                                            **kwargs)


    running_time.append(perf_counter() - time_odeint_s)

    qs_opt_full = TI_primal(q0, f, A_p, psi, wf.Nxi, wf.Nt, wf.dt)
    JJ = Calc_Cost(qs_opt_full, qs_target, f, kwargs['dx'], kwargs['dt'], kwargs['lamda'])

    J_opt_FOM_list.append(JJ)
    J_opt_list.append(J_opt)
    dL_du_norm_list.append(dL_du_norm)
    dL_du_norm_ratio_list.append(dL_du_norm / dL_du_norm_list[0])

    print(
        f"J_opt : {J_opt}, J_FOM: {JJ}, ||dL_du|| = {dL_du_norm}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du_norm / dL_du_norm_list[0]}"
    )

    # Convergence criteria
    if opt_step == kwargs['opt_iter'] - 1:
        print("\n\n-------------------------------")
        print(
            f"WARNING... maximal number of steps reached, "
            f"J_opt : {J_opt}, J_FOM: {JJ}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du_norm / dL_du_norm_list[0]}"
        )
        break
    elif dL_du_norm / dL_du_norm_list[0] < kwargs['delta_conv']:
        print("\n\n-------------------------------")
        print(
            f"Optimization converged with, "
            f"J_opt : {J_opt}, J_FOM: {JJ}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du_norm / dL_du_norm_list[0]}"
        )
        break
    else:
        if opt_step == 0:
            if stag:
                print("\n\n-------------------------------")
                print(
                    f"Armijo Stagnated !!!!!! due to the step length being too low thus exiting at itr: {opt_step} with "
                    f"J_opt : {J_opt}, J_FOM: {JJ}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du_norm / dL_du_norm_list[0]}")
                break
        else:
            dJ = (J_opt_list[-1] - J_opt_list[-2]) / J_opt_list[0]
            if abs(dJ) == 0:
                print(f"WARNING: dJ has turned close to 0...")
                break
            if stag:
                print("\n-------------------------------")
                print(
                    f"Armijo Stagnated !!!!!! due to the step length being too low thus exiting at itr: {opt_step} with "
                    f"J_opt : {J_opt}, J_FOM: {JJ}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du_norm / dL_du_norm_list[0]}")
                break

# Compute the final state
qs_opt_full = TI_primal(q0, f, A_p, psi, wf.Nxi, wf.Nt, wf.dt)
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
np.save(impath + 'err_list.npy', err_list)
np.save(impath + 'trunc_modes_list.npy', trunc_modes_list)
np.save(impath + 'running_time.npy', running_time)

# Save the optimized solution
np.save(impath + 'qs_opt.npy', qs_opt_full)
np.save(impath + 'qs_adj_opt.npy', qs_adj)
np.save(impath + 'f_opt.npy', f_opt)
np.save(impath + 'f_opt_low.npy', f)

# %%
# Plot the results
pf = PlotFlow(wf.X, wf.t)
pf.plot1D(qs_org, name="qs_org", immpath=immpath)
pf.plot1D(qs_target, name="qs_target", immpath=immpath)
pf.plot1D(qs_opt_full, name="qs_opt", immpath=immpath)
pf.plot1D(qs_adj, name="qs_adj_opt", immpath=immpath)
pf.plot1D(f_opt, name="f_opt", immpath=immpath)
pf.plot1D_ROM_converg(J_opt_list, immpath=immpath)
