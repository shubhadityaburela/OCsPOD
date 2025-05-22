import matplotlib.pyplot as plt

from Coefficient_Matrix import CoefficientMatrix
from Costs import Calc_Cost
from FOM_solver import IC_primal, TI_primal, TI_primal_target, IC_adjoint, TI_adjoint
from Grads import Calc_Grad
from Helper import ControlSelectionMatrix_advection, L2norm_ROM
from TI_schemes import DF_start_FOM
from Update import Update_Control_TWBT, get_BB_step, Update_Control_BB
from grid_params import advection
from Plots import PlotFlow
import numpy as np
import os
from time import perf_counter
import time
import scipy.sparse as sp
import argparse
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import inv
from scipy.sparse import csc_matrix

parser = argparse.ArgumentParser(description="Input the variables for running the script.")
parser.add_argument("problem", type=int, choices=[1, 2, 3], help="Specify the problem number (1, 2, or 3)")
parser.add_argument("N_iter", type=int, help="Enter the number of optimization iterations")
parser.add_argument("dir_prefix", type=str, choices=[".", "/work/burela"],
                    help="Specify the directory prefix for proper storage of the files")
args = parser.parse_args()
problem = args.problem

print("\n")
print(f"Solving problem: {args.problem}")

Nxi = 3200 // 4
Nt = 3360 // 4

impath = args.dir_prefix + "/data/FOM/problem=" + str(problem) + "/"  # for data
immpath = args.dir_prefix + "/plots/FOM/problem=" + str(problem) + "/"  # for plots
os.makedirs(impath, exist_ok=True)

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
np.save(impath + 'qs_org.npy', qs_org)

qs_target = TI_primal_target(qs0, Mat.Grad_Xi_kron, wf.v_x_target, wf.Nxi, wf.Nt, wf.dt)
np.save(impath + 'qs_target.npy', qs_target)

# Initial conditions for both primal and adjoint are defined here as they only need to defined once.
q0 = IC_primal(wf.X, wf.Lxi, wf.offset, wf.variance)
q0_adj = IC_adjoint(wf.X)

# %% Optimal control
dL_du_norm_list = []  # Collecting the gradient over the optimization steps
J_opt_list = []  # Collecting the optimal cost functional for plotting
J_opt_FOM_list = []  # Collecting the FOM cost over the optimization steps
running_time = []  # Time calculated for each iteration in a running manner
dL_du_norm_ratio_list = []  # Collecting the ratio of gradients for plotting

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
    'omega_cutoff': 1e-10,  # Below this cutoff the Armijo and Backtracking should exit the update loop
    'adjoint_scheme': "RK4"  # Time integration scheme for adjoint equation
}

# %% Select the LU pre-factors for the inverse of mass matrix for linear solve of adjoint equation
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

# %%
# For two-way backtracking line search
omega_twbt = 1
omega_bb = 1
stag = False

start = time.time()
time_odeint_s = perf_counter()  # save running time
# %%
for opt_step in range(kwargs['opt_iter']):
    '''
    Forward calculation
    '''
    print("\n-------------------------------")
    print("Optimization step: %d" % opt_step)

    qs = TI_primal(q0, f, A_p, psi, wf.Nxi, wf.Nt, wf.dt)

    '''
    Objective and costs for control
    '''
    J = Calc_Cost(qs, qs_target, f, kwargs['dx'], kwargs['dt'], kwargs['lamda'])

    qs_opt_full = TI_primal(q0, f, A_p, psi, wf.Nxi, wf.Nt, wf.dt)
    JJ = Calc_Cost(qs_opt_full, qs_target, f,
                   kwargs['dx'], kwargs['dt'], kwargs['lamda'])

    J_opt_list.append(J)
    J_opt_FOM_list.append(JJ)

    '''
    Adjoint calculation
    '''
    qs_adj = TI_adjoint(q0_adj, qs, qs_target, M_f, A_f, LU_M_f, wf.Nxi, wf.dx, wf.Nt, wf.dt,
                        scheme=kwargs['adjoint_scheme'], opt_poly_jacobian=Df)

    '''
     Update Control
    '''
    dL_du = Calc_Grad(psi, f, qs_adj, kwargs['lamda'])
    dL_du_norm_square = L2norm_ROM(dL_du, kwargs['dt'])
    dL_du_norm = np.sqrt(dL_du_norm_square)

    dL_du_norm_list.append(dL_du_norm)
    dL_du_norm_ratio_list.append(dL_du_norm / dL_du_norm_list[0])

    if dL_du_norm / dL_du_norm_list[0] < 5e-3:
        print(f"BB acting.....")
        omega_bb = get_BB_step(fOld, fNew, dL_du_Old, dL_du, opt_step, **kwargs)
        if omega_bb < 0:  # Negative BB step not accepted (Thus running Armijo step)
            print(f"WARNING... BB gave negative step length, thus ignoring that and using TWBT for correct step size")
            print(f"TWBT acting.....")
            fNew, J_opt, dL_du, omega_twbt, stag = Update_Control_TWBT(f, q0, qs_target, psi,
                                                                       A_p, J, omega_twbt,
                                                                       dL_du, dL_du_norm_square,
                                                                       **kwargs)
        else:
            fNew = Update_Control_BB(fNew, dL_du, omega_bb)
            stag = False
    else:
        print(f"TWBT acting.....")
        fNew, J_opt, dL_du, omega_twbt, stag = Update_Control_TWBT(f, q0, qs_target, psi,
                                                                   A_p, J, omega_twbt,
                                                                   dL_du, dL_du_norm_square,
                                                                   **kwargs)

    running_time.append(perf_counter() - time_odeint_s)

    # Saving previous controls for Barzilai Borwein step
    fOld = np.copy(f)
    f = np.copy(fNew)
    dL_du_Old = np.copy(dL_du)

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
        break
    elif dL_du_norm / dL_du_norm_list[0] < kwargs['delta_conv']:
        print("\n\n-------------------------------")
        print(
            f"Optimization converged with, "
            f"J_opt : {J}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du_norm / dL_du_norm_list[0]}"
        )
        break
    else:
        if opt_step == 0:
            if stag:
                print("\n\n-------------------------------")
                print(f"Armijo Stagnated !!!!!! due to the step length being too low thus exiting at {opt_step} with "
                      f"J_opt : {J}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du_norm / dL_du_norm_list[0]}")
                break
        else:
            dJ = (J_opt_list[-1] - J_opt_list[-2]) / J_opt_list[0]
            if abs(dJ) == 0:
                print(f"WARNING: dJ has turned close to 0...")
                break
            if stag:
                print("\n-------------------------------")
                print(f"Armijo Stagnated !!!!!! due to the step length being too low thus exiting at {opt_step} with "
                      f"J_opt : {J}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du_norm / dL_du_norm_list[0]}")
                break
            # Convergence criteria for BB
            if JJ > 1e6 or abs(omega_bb) < kwargs['omega_cutoff']:
                print("\n\n-------------------------------")
                print(
                    f"Barzilai Borwein acceleration failed!!!!!! J increased to unrealistic values or the omega went below cutoff, thus exiting "
                    f"at itr: {opt_step} with "
                    f"J_ROM: {J}, J_FOM: {JJ}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du_norm / dL_du_norm_list[0]}")
                f = np.copy(fOld)
                break

# Final state corresponding to the optimal control f
qs_opt = TI_primal(q0, f, A_p, psi, wf.Nxi, wf.Nt, wf.dt)
f_opt = psi @ f

# Compute the cost with the optimal control
J = Calc_Cost(qs_opt, qs_target, f, kwargs['dx'], kwargs['dt'], kwargs['lamda'])
print("\n")
print(f"J with respect to the optimal control for FOM: {J}")

end = time.time()
print("\n")
print("Total time elapsed = %1.3f" % (end - start))
# %%
# Save the convergence lists
np.save(impath + 'J_opt_list.npy', J_opt_list)
np.save(impath + 'running_time.npy', running_time)

# # Save the optimized solution
# np.save(impath + 'qs_opt.npy', qs_opt)
# np.save(impath + 'qs_adj_opt.npy', qs_adj)
# np.save(impath + 'f_opt.npy', f_opt)
# np.save(impath + 'f_opt_low.npy', f)

# %%
# Plot the results
pf = PlotFlow(wf.X, wf.t)

pf.plot1D(qs_org, name="qs_org", immpath=immpath)
pf.plot1D(qs_target, name="qs_target", immpath=immpath)
pf.plot1D(qs_opt, name="qs_opt", immpath=immpath)
pf.plot1D(qs_adj, name="qs_adj_opt", immpath=immpath)
pf.plot1D(f_opt, name="f_opt", immpath=immpath)
pf.plot1D_FOM_converg(J_opt_list, immpath=immpath)
